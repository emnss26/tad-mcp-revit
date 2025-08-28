import json, pathlib

PATH = pathlib.Path('data/train_data_templates.jsonl')
BAD = {  # línea -> columna (1-based) que reportó tu validador
    63: 682, 64: 635, 65: 766, 76: 815, 79: 611, 82: 617, 83: 712,
    89: 637, 96: 589, 102: 599, 106: 596, 111: 686, 120: 543, 122: 916
}

lines = PATH.read_text(encoding='utf-8').splitlines()
for ln, col in BAD.items():
    s = lines[ln-1]
    i = max(0, col-1-80)
    j = min(len(s), col-1+80)
    frag = s[i:j]
    pointer = ' ' * (min(80, col-1-i)) + '^'
    print(f"\n--- LINEA {ln} (len={len(s)}) col={col} ---")
    print(frag)
    print(pointer)