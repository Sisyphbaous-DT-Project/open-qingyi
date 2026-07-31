#!/usr/bin/env python
"""Numerical cross-check: qingyi_kda.kda_recurrence (pure PyTorch reference)
vs fla-core's official `chunk_kda` kernel.

fp32 cases must pass with max relative error < 1e-4.
bf16 errors are only recorded, not judged.
A state-handoff (chunked continuation) case verifies streaming decoding:
full sequence in one call vs two segments joined by the final state.
"""

import os
import sys

import torch

sys.path.insert(0, "/root/projects/qingyi-kda")

# Force IEEE fp32 in all triton dots so the kernel can be compared against the
# fp32 reference at 1e-4 tolerance. Triton would otherwise default to tf32 on
# sm_89 (rel err ~1e-3). Must be set before triton/fla are imported.
os.environ["TRITON_F32_DEFAULT"] = "ieee"

import triton  # noqa: E402
from fla.ops.kda import chunk_kda  # noqa: E402
import fla.ops.kda.chunk_intra as _chunk_intra  # noqa: E402

# The intra-chunk UT-transform solve hardcodes tf32 on tf32-capable cards
# (fla/ops/kda/chunk_intra.py). Patch the module global to ieee BEFORE the
# first kernel compile (triton resolves globals lazily at compile time).
# This is a test-only precision override, not a change to fla semantics.
_chunk_intra.SOLVE_TRIL_DOT_PRECISION = triton.language.constexpr("ieee")

from qingyi_kda import kda_recurrence  # noqa: E402

DEVICE = "cuda"
REL_TOL = 1e-4


def make_inputs(B, T, H, HV, K, V, dtype, seed, with_state=False, scale=None):
    """Random KDA inputs. g is log-space decay (<= 0), beta is post-sigmoid.

    q/k are l2-normalized per token, matching real KDA usage
    (``use_qk_l2norm_in_kernel=True`` in fla layers). With unnormalized keys
    the delta-rule transition (I - beta k k^T) has eigenvalues of magnitude
    >> 1, the state explodes exponentially, and ANY implementation difference
    (even fp32 rounding order) is amplified beyond recognition -- making
    numerical comparison meaningless.
    """
    gen = torch.Generator(device=DEVICE).manual_seed(seed)

    def rnd(*shape, s=1.0):
        return torch.randn(*shape, generator=gen, device=DEVICE, dtype=torch.float32) * s

    q = torch.nn.functional.normalize(rnd(B, T, H, K), dim=-1).to(dtype)
    k = torch.nn.functional.normalize(rnd(B, T, H, K), dim=-1).to(dtype)
    v = rnd(B, T, HV, V).to(dtype)
    # Per-channel log decay in [-3, 0) -> per-step decay in (e^-3, 1).
    g = (-torch.rand(B, T, HV, K, generator=gen, device=DEVICE) * 3).to(dtype)
    # Post-sigmoid write gate in (0, 1).
    beta = torch.rand(B, T, HV, generator=gen, device=DEVICE).to(dtype)
    # fla requires initial_state in fp32 regardless of the input dtype.
    h0 = rnd(B, HV, K, V, s=0.1) if with_state else None
    return q, k, v, g, beta, h0, scale


def err_stats(a, b):
    """Error metrics of a w.r.t. reference b.

    Returns (abs_err, rel_err) where:
    - abs_err = max |a - b|
    - rel_err = max |a - b| / max |b|  (relative to the peak magnitude of b)

    The peak-scaled relative error is the meaningful metric here: the delta
    rule involves heavy cancellation, so some output/state elements sit at
    ~1e-5 while the tensor magnitude is O(1). A pointwise ratio
    |diff|/(|b|+eps) flags those near-zero elements as huge "relative" errors
    even when both implementations agree to fp32 rounding (abs err ~1e-7).
    """
    a, b = a.float(), b.float()
    diff = (a - b).abs().max().item()
    rel = diff / b.abs().max().clamp_min(1e-12).item()
    return diff, rel


def run_case(name, B, T, H, HV, K, V, dtype, seed, with_state=False, scale=None, judge=True):
    q, k, v, g, beta, h0, scale = make_inputs(B, T, H, HV, K, V, dtype, seed, with_state, scale)

    o_ref, s_ref = kda_recurrence(
        q, k, v, g, beta, scale=scale, initial_state=h0, output_final_state=True
    )
    o_fla, s_fla = chunk_kda(
        q, k, v, g, beta, scale=scale, initial_state=h0, output_final_state=True
    )

    o_abs, o_rel = err_stats(o_ref, o_fla)
    s_abs, s_rel = err_stats(s_ref, s_fla)
    rel = max(o_rel, s_rel)
    status = ""
    if judge:
        status = "PASS" if rel < REL_TOL else "FAIL"
    print(
        f"[{status or 'INFO':4}] {name:28} B={B} T={T:3} H={H} HV={HV} K={K:3} V={V:3} "
        f"{str(dtype).replace('torch.', ''):8} | o: abs={o_abs:.3e} rel={o_rel:.3e} "
        f"| S: abs={s_abs:.3e} rel={s_rel:.3e}"
    )
    return not judge or rel < REL_TOL


def run_continuation_case(seed=7):
    """Full pass vs two segments joined by final_state (both via fla kernel)."""
    B, T1, T2, H, HV, K, V = 2, 120, 80, 4, 4, 64, 64
    q, k, v, g, beta, h0, _ = make_inputs(B, T1 + T2, H, HV, K, V, torch.float32, seed,
                                          with_state=True)

    o_full, s_full = chunk_kda(q, k, v, g, beta, initial_state=h0, output_final_state=True)

    sl = slice(None), slice(0, T1)
    o1, s1 = chunk_kda(q[:, :T1], k[:, :T1], v[:, :T1], g[:, :T1], beta[:, :T1],
                       initial_state=h0, output_final_state=True)
    o2, s2 = chunk_kda(q[:, T1:], k[:, T1:], v[:, T1:], g[:, T1:], beta[:, T1:],
                       initial_state=s1, output_final_state=True)
    del sl

    o_abs, o_rel = err_stats(torch.cat([o1, o2], dim=1), o_full)
    s_abs, s_rel = err_stats(s2, s_full)
    rel = max(o_rel, s_rel)
    status = "PASS" if rel < REL_TOL else "FAIL"
    print(
        f"[{status:4}] {'continuation (fla handoff)':28} B={B} T={T1}+{T2} H={H} HV={HV} "
        f"K={K:3} V={V:3} float32  | o: abs={o_abs:.3e} rel={o_rel:.3e} "
        f"| S: abs={s_abs:.3e} rel={s_rel:.3e}"
    )
    return rel < REL_TOL


def main():
    torch.manual_seed(0)
    ok = True
    print("=" * 110)
    print("fp32 cases (judged, rel < 1e-4): reference kda_recurrence vs fla chunk_kda")
    print("=" * 110)
    ok &= run_case("basic", 2, 128, 4, 4, 64, 64, torch.float32, seed=1)
    ok &= run_case("gva (HV=2H)", 2, 128, 2, 4, 64, 64, torch.float32, seed=2)
    ok &= run_case("single-token decode", 2, 1, 4, 4, 64, 64, torch.float32, seed=3)
    ok &= run_case("odd len 100", 1, 100, 4, 4, 64, 128, torch.float32, seed=4)
    ok &= run_case("odd len 257", 1, 257, 4, 4, 64, 64, torch.float32, seed=5)
    ok &= run_case("initial_state + scale", 2, 150, 4, 4, 64, 64, torch.float32,
                   seed=6, with_state=True, scale=0.25)
    print("=" * 110)
    print("streaming handoff (judged, rel < 1e-4)")
    print("=" * 110)
    ok &= run_continuation_case(seed=7)
    print("=" * 110)
    print("bf16 cases (recorded only, NOT judged)")
    print("=" * 110)
    run_case("bf16 basic", 2, 200, 4, 4, 64, 64, torch.bfloat16, seed=8, judge=False)
    run_case("bf16 initial_state", 2, 200, 4, 4, 64, 64, torch.bfloat16, seed=9,
             with_state=True, judge=False)
    print("=" * 110)
    print("ALL FP32 CASES PASSED" if ok else "SOME CASES FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
