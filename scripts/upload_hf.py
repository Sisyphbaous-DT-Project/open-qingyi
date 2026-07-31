#!/usr/bin/env python
"""Upload a local model folder to the Hugging Face Hub.

Reads the token from the HF_TOKEN environment variable (never hardcoded).

Usage:
    export HF_TOKEN=hf_...
    python scripts/upload_hf.py /path/to/model-folder
    python scripts/upload_hf.py /path/to/model-folder --repo-id user/repo
"""
import argparse
import os

from huggingface_hub import HfApi


def main():
    p = argparse.ArgumentParser(description="Upload a folder to a HF model repo.")
    p.add_argument("folder_path", help="local folder to upload")
    p.add_argument("--repo-id", default="shiershuihesaixiliya/qingyi-kda-0.6b",
                   help="target repo id (default: %(default)s)")
    args = p.parse_args()

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="model", private=False, exist_ok=True)
    print("REPO-OK", flush=True)
    api.upload_folder(folder_path=args.folder_path,
                      repo_id=args.repo_id,
                      repo_type="model")
    print("UPLOAD-DONE", flush=True)


if __name__ == "__main__":
    main()
