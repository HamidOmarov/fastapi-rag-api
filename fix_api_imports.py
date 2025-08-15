import re, pathlib

p = pathlib.Path("app/api.py")
s = p.read_text(encoding="utf-8")
lines = s.splitlines()

n = len(lines)
i = 0

# --- Shebang / encoding sətirlərini saxla
header = []
while i < n and (lines[i].startswith("#!") or re.search(r"coding[:=]\s*[-\w.]+", lines[i])):
    header.append(lines[i]); i += 1

# --- Docstring varsa götür (yalnız faylın əvvəlində olanda)
doc_block = []
if i < n and re.match(r'^\s*(?P<q>"""|\'\'\')', lines[i]):
    q = '"""' if lines[i].strip().startswith('"""') else "'''"
    doc_block.append(lines[i]); i += 1
    while i < n:
        doc_block.append(lines[i])
        if q in lines[i]:
            i += 1
            break
        i += 1

# --- Qalan hissədən __future__ importlarını çək
future_re = re.compile(r'^\s*from __future__ import ')
futures, rest = [], []
for ln in lines[i:]:
    if future_re.match(ln):
        futures.append(ln.rstrip())
    else:
        rest.append(ln)

# unikallaşdır
seen = set(); futures = [x for x in futures if not (x in seen or seen.add(x))]

# --- Köhnə storage importlarını təmizlə
rest = [ln for ln in rest if not re.match(r'^\s*from app\.storage import ', ln)]

storage_line = 'from app.storage import DATA_DIR, INDEX_DIR, HISTORY_JSON'

out = []
out += header
if header and (doc_block or futures or storage_line): out.append('')
out += doc_block
if doc_block and (futures or storage_line): out.append('')
out += futures
if futures: out.append(storage_line)
else:
    # futures yoxdursa, storage-ı docstringdən sonra qoyuruq
    if doc_block: out.append(storage_line)
    else: out.append(storage_line)
# importlardan sonra boş sətir
out.append('')

# qalan hissə
out += rest

txt = '\n'.join(out).rstrip() + '\n'
p.write_text(txt, encoding="utf-8")
print("OK: rearranged", p)
