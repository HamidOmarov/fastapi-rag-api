import pathlib

def clean_file(path: pathlib.Path):
    b = path.read_bytes()
    # BOM-u (əgər faylın əvvəlindədirsə) kənarlaşdırmaq üçün utf-8-sig ilə oxu
    try:
        s = b.decode("utf-8-sig")
    except UnicodeDecodeError:
        s = b.decode("utf-8", errors="ignore")
    changed = False
    if "\ufeff" in s:           # sətir içi FEFF-ləri də sil
        s = s.replace("\ufeff", "")
        changed = True
    # Əgər başlanğıcda BOM vardısa, utf-8-sig artıq onu çıxarıb; yazarkən BOMsuz yaz
    if changed or b[:3] == b"\xef\xbb\xbf":
        path.write_text(s, encoding="utf-8", newline="\n")
        print(f"cleaned: {path}")
        return 1
    return 0

changed = 0
# app/ altındakı bütün .py fayllar
for p in pathlib.Path("app").rglob("*.py"):
    changed += clean_file(p)

# kökdə .py varsa, onları da yoxla (opsional)
for p in pathlib.Path(".").glob("*.py"):
    changed += clean_file(p)

print("total_changed:", changed)
