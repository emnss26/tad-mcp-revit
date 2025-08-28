import json, re, pathlib, sys

SRC = pathlib.Path("data/train_data_templates.jsonl")
BAK = SRC.with_suffix(".jsonl.bak")

# Parches comunes
FIXES = [
    # 1) Cierre de más al final: ]]}  -> ]}}
    (re.compile(r'\]\]\}\s*$'), '}]}' ),

    # 2) Falta coma entre objetos dentro de templates:
    # ...}}{"prompt": ...  -> ...}}}, {"prompt": ...
    (re.compile(r'\}\}\s*\{\s*"prompt"\s*:'), '}}}, {"prompt":'),
    (re.compile(r'\}\}\s*\{\s*"action"\s*:'), '}}}, {"action":'),
    (re.compile(r'\}\}\s*\{\s*"plan"\s*:'),   '}}}, {"plan":'),

    # 3) Falta coma despues de un objeto "plan": {...}}{"prompt":...
    (re.compile(r'("plan"\s*:\s*\{[^{}]*\}\s*)\}\s*\{\s*"prompt"\s*:'), r'\1}}, {"prompt":'),

    # 4) Comas sobrantes antes de cerrar arrays/objetos: ,]  o ,}
    (re.compile(r',\s*\]'), ']'),
    (re.compile(r',\s*\}'), '}'),
    (re.compile(r'}\]\]\s*$'), '}]}'),
    (re.compile(r'}\s*\]\s*\]\s*$'), '}]}'),
]

def try_fix(s: str) -> str:
    """Aplica FIXES secuencialmente solo si siguen fallando los parseos."""
    orig = s
    for _ in range(3):  # hasta 3 rondas de parches
        try:
            json.loads(s)
            return s  # ya válido
        except json.JSONDecodeError as e:
            # aplica parches uno por uno e intenta mejorar
            changed = False
            for rx, rep in FIXES:
                new_s = rx.sub(rep, s)
                if new_s != s:
                    s = new_s
                    changed = True
            if not changed:
                # no hay nada más que arreglar automáticamente
                return s
    return s

def main():
    lines = SRC.read_text(encoding="utf-8").splitlines()
    bad_report = []
    repaired = []
    changed = 0

    for i, line in enumerate(lines, start=1):
        s = line.strip()
        if not s:
            repaired.append(line)
            continue

        try:
            json.loads(s)
            repaired.append(line)
            continue
        except json.JSONDecodeError as e:
            fixed = try_fix(s)
            try:
                json.loads(fixed)
                repaired.append(fixed)
                changed += 1
            except json.JSONDecodeError as e2:
                # deja la original y reporta
                repaired.append(line)
                snippet = s[:200].replace('\n',' ')
                bad_report.append((i, str(e2), snippet))

    # backup y escritura
    BAK.write_text('\n'.join(lines) + '\n', encoding="utf-8")
    SRC.write_text('\n'.join(repaired) + '\n', encoding="utf-8")

    print(f"Archivo original respaldado en: {BAK}")
    print(f"Líneas reparadas automáticamente: {changed}")
    if bad_report:
        print("Aún fallan estas líneas:")
        for num, err, snip in bad_report:
            print(f"- Línea {num}: {err} | {snip}...")

if __name__ == "__main__":
    main()