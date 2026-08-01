src = open('main.tex', encoding='utf-8').read()

# 1. header comment + review mode
old_head = """% QINGYI-KDA-0.6B workshop paper
% Compile: pdflatex main && bibtex main && pdflatex main && pdflatex main
% Change "review" to "final" for camera-ready, "preprint" for arXiv version.
\\documentclass[11pt]{article}
\\usepackage[preprint]{acl}"""
new_head = """% ARR submission version (anonymous, review mode)
% Compile: pdflatex main && bibtex main && pdflatex main && pdflatex main
\\documentclass[11pt]{article}
\\usepackage[review]{acl}"""
assert old_head in src, 'head not found'
src = src.replace(old_head, new_head)

# 2. anonymous author block
old_author = """\\author{Ronglong Bao\\thanks{~~Code, weights and recipes: \\url{https://github.com/Sisyphbaous-DT-Project/open-qingyi}} \\\\
  DT-Project \\\\
  \\texttt{islonglongy@qq.com}}"""
assert old_author in src, 'author not found'
src = src.replace(old_author, """\\author{Anonymous ARR submission}""")

# 3. drop acknowledgments for review
old_ack = """\\section*{Acknowledgments}

Compute: one rented 32\\,GB GPU; local RTX 4070 Laptop for v1. We thank
the authors of GenDistill and HALO for public recipes and failure
records.

"""
assert old_ack in src, 'ack not found'
src = src.replace(old_ack, '')

open('arr/main.tex', 'w', encoding='utf-8', newline='\n').write(src)
print('written, len', len(src))
print('review mode ->', 'usepackage[review]{acl}' in src)
print('anonymous ->', 'Anonymous ARR submission' in src)
print('acks removed ->', 'Acknowledgments' not in src)
print('no github ->', 'github.com' not in src)
print('no qq ->', 'qq.com' not in src)
