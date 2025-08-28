import json, re, shutil, pathlib

p = pathlib.Path('data/train_data_templates.jsonl')
bak = p.with_suffix('.jsonl.badtails.bak')
shutil.copy(p, bak)

def normalize_tail(s: str) -> str:
    s = s.rstrip()

    # Si ya termina en '}]}' no tocamos
    if s.endswith('}]}'):
        return s

    # Colas frecuentes tipo '}]}]', '}]}]]', '}]}}]', '}]}}}]', con o sin espacios
    s = re.sub(r'\}\s*\]\s*\]\s*$', '}]}', s)           # '}]]' -> '}]}'
    s = re.sub(r'\}\s*\]\s*\]\s*\]\s*$', '}]}', s)      # '}]]]' -> '}]}'
    s = re.sub(r'\}\s*\]\s*\}\s*\]\s*$', '}]}', s)      # '}]}]' (mezcla) -> '}]}'
    s = re.sub(r'\}\s*\]\s*\}\s*\}\s*$', '}]}', s)      # '}}}}' al final -> '}]}'
    s = re.sub(r'\]\s*\}\s*\]\s*$', '}]}' , s)          # '] } ]' -> '}]}'
    s = re.sub(r'\}\s*\]\s*$', '}]}' , s)               # si quedó '} ]' -> '}]}' (seguro para cola)
    s = s.rstrip()
    return s

bad_lines = []
fixed_lines = []
for idx, line in enumerate(p.read_text(encoding='utf-8').splitlines(), 1):
    t = line
    try:
        json.loads(t)
    except Exception:
        # Intento de normalización de cola y re-parseo
        t2 = normalize_tail(t)
        try:
            json.loads(t2)
            t = t2
        except Exception as e:
            bad_lines.append((idx, str(e)))
    fixed_lines.append(t)

p.write_text('\n'.join(fixed_lines), encoding='utf-8')

if bad_lines:
    print("Aún fallan estas líneas:")
    for n, err in bad_lines:
        print(f"- Línea {n}: {err}")
else:
    print("OK: todas las líneas parsean tras normalizar colas.")
    print(f"Backup: {bak}")