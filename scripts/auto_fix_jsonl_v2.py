
import json, re, shutil, pathlib

p = pathlib.Path('data/train_data_templates.jsonl')
bak = p.with_suffix('.jsonl.autofix.bak')
shutil.copy(p, bak)

def normalize_tail(s: str) -> str:
    s = s.rstrip()
    # Consolidar cierres al final a '}]}' (lo correcto para {"templates":[ ... ]})
    s = re.sub(r'\}\s*\]\s*\]\s*$', '}]}', s)
    s = re.sub(r'\}\s*\]\s*\]\s*\]\s*$', '}]}', s)
    s = re.sub(r'\]\s*\}\s*\]\s*$', '}]}', s)
    s = re.sub(r'\}\s*\]\s*$', '}]}', s)
    return s

def add_missing_commas_between_template_objects(s: str) -> str:
    # Dentro de "templates": [ { ... } { "prompt": ... } ]  => falta coma
    # Patrón más seguro: dos llaves de cierre seguidas de otra llave de apertura antes de "prompt"
    s = re.sub(r'\}\}\s*\{\s*"prompt"\s*:', r'}}, {"prompt":', s)

    # También: cierre de plan } seguido de { "plan" sin coma (poco común, pero por si acaso)
    s = re.sub(r'\}\s*\{\s*"prompt"\s*:', r'}, {"prompt":', s)

    # Por si un editor metió espacios raros entre '}' y '{'
    s = re.sub(r'\}\s+\{\s*"prompt"\s*:', r'}, {"prompt":', s)

    return s

bad = []
fixed_lines = []
for ln, line in enumerate(p.read_text(encoding='utf-8').splitlines(), 1):
    t = line
    try:
        json.loads(t)
    except Exception:
        t2 = normalize_tail(t)
        if t2 != t:
            t = t2
        # Intento de insertar comas faltantes entre objetos de templates
        t3 = add_missing_commas_between_template_objects(t)
        try:
            json.loads(t3)
            t = t3
        except Exception as e:
            bad.append((ln, str(e)))
    fixed_lines.append(t)

p.write_text('\n'.join(fixed_lines), encoding='utf-8')

if bad:
    print("Aún fallan estas líneas:")
    for n, err in bad:
        print(f"- Línea {n}: {err}")
    print(f"\nSe creó backup en: {bak}")
else:
    print("OK: todas las líneas parsean tras la reparación.")
    print(f"Backup: {bak}")