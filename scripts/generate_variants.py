import json
import os
import random
import re
from typing import Dict, List, Tuple

random.seed(42)

INPUT = "data/train_data_templates.jsonl"
OUTPUT = "data/train_data_variants.jsonl"

# Regex to find placeholders including format specifiers (e.g., {name}, {x:.2f})
PLACEHOLDER_RE = re.compile(r"\{[^{}]+\}")

# --- Utility helpers --------------------------------------------------------

def extract_placeholders(text: str) -> List[str]:
    return PLACEHOLDER_RE.findall(text)

def freeze_placeholders(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Replace {placeholders} with unique tokens to protect them during paraphrasing/translation.
    Returns (masked_text, map_token_to_placeholder).
    """
    mapping = {}
    def repl(m):
        ph = m.group(0)
        token = f"__PH_{len(mapping)}__"
        mapping[token] = ph
        return token
    return PLACEHOLDER_RE.sub(repl, text), mapping

def unfreeze_placeholders(text: str, mapping: Dict[str, str]) -> str:
    for token, ph in mapping.items():
        text = text.replace(token, ph)
    return text

def ensure_min_unique(items: List[str], k: int) -> List[str]:
    seen = set()
    out = []
    for it in items:
        key = it.strip().lower()
        if key not in seen:
            out.append(it)
            seen.add(key)
        if len(out) >= k:
            break
    # If still short, add enumerated suffix (keeps exact prompt uniqueness without changing semantics)
    while len(out) < k and out:
        base = out[len(out) % len(out)]
        out.append(base + " ")
    return out

def detect_lang(text: str) -> str:
    """Very light heuristic for ES/EN; defaults to ES if uncertain."""
    t = text.lower()
    spanish_markers = ["nivel", "coloca", "inserta", "dibuja", "crea", "traza", "planta", "vista", "hoja", "muro", "viga", "columna", "ducto", "tubería", "rejilla", "baranda", "escalera"]
    english_markers = ["level", "place", "insert", "draw", "create", "trace", "plan", "view", "sheet", "wall", "beam", "column", "duct", "pipe", "grid", "railing", "stair"]
    if any(w in t for w in spanish_markers) and not any(w in t for w in english_markers):
        return "es"
    if any(w in t for w in english_markers) and not any(w in t for w in spanish_markers):
        return "en"
    # Fallback: if contains accented letters, favor ES
    if any(ch in t for ch in "áéíóúñ¿¡"):
        return "es"
    return "es"

# --- Paraphrase engines -----------------------------------------------------

SYNS_ES = {
    "crea": ["crea", "genera", "añade", "haz", "construye"],
    "coloca": ["coloca", "inserta", "ubica", "sitúa", "pon"],
    "dibuja": ["dibuja", "traza", "modela"],
    "traza": ["traza", "dibuja", "define"],
    "inserta": ["inserta", "coloca", "agrega"],
    "quiero": ["quiero", "necesito", "por favor", "me gustaría"],
    "desde": ["desde", "de", "partiendo de"],
    "hasta": ["hasta", "a", "terminando en"],
    "en_el_nivel": [
        "en el nivel '{level}'",
        "dentro del nivel '{level}'",
        "sobre el nivel '{level}'"
    ],
    "en_la_vista": [
        "en la vista '{view}'",
        "dentro de la vista '{view}'"
    ],
    "llamado": ["llamado", "con el nombre", "nombrado", "que se llame"],
    "tipo": ["tipo", "de tipo", "del tipo"],
}

SYNS_EN = {
    "create": ["create", "generate", "add", "make", "build"],
    "place": ["place", "insert", "put", "drop"],
    "draw": ["draw", "trace", "model"],
    "insert": ["insert", "place", "add"],
    "from": ["from", "starting at"],
    "to": ["to", "ending at"],
    "on_level": [
        "on level '{level}'",
        "at level '{level}'",
        "within level '{level}'"
    ],
    "in_view": [
        "in view '{view}'",
        "inside view '{view}'"
    ],
    "called": ["called", "named"],
    "type": ["type", "of type"],
}

# Common structural patterns that preserve placeholders.
PATTERNS_ES = [
    # Basic direct forms
    "{verb} {obj} {type_clause} {context_clause}{loc_clause}{extra_clause}",
    "{verb_cap} {obj} {type_clause} {loc_clause} {context_clause}{extra_clause}",
    "{prelude} {verb} {obj} {type_clause} {context_clause}{loc_clause}{extra_clause}",
    # Imperative with explicit coordinates or ranges
    "{verb_cap} {obj} {type_clause} {context_clause} {path_clause}{extra_clause}",
    # Vary order of context vs location
    "{verb_cap} {obj} {type_clause} {loc_clause} {context_clause}{extra_clause}",
]

PATTERNS_EN = [
    "{verb_cap} {obj} {type_clause} {context_clause}{loc_clause}{extra_clause}",
    "{verb_cap} {obj} {type_clause} {loc_clause} {context_clause}{extra_clause}",
    "{prelude} {verb} {obj} {type_clause} {context_clause}{loc_clause}{extra_clause}",
    "{verb_cap} {obj} {type_clause} {path_clause}{extra_clause}",
]

# Utility to build clauses safely while keeping placeholders intact
def make_type_clause_es(base: str) -> str:
    # Expect something like: "tipo '{type}'" already in prompt; fallback generic
    if "'{type}'" in base or "{type}" in base:
        return "de tipo '{type}'"
    # Try detect any '{...type...}' placeholder
    if re.search(r"\{[^}]*type[^}]*\}", base):
        return "de tipo '{type}'"
    return ""

def make_type_clause_en(base: str) -> str:
    if "'{type}'" in base or "{type}" in base or re.search(r"\{[^}]*type[^}]*\}", base):
        return "of type '{type}'"
    return ""

def find_context_level_es(base: str) -> str:
    if "{level}" in base:
        return random.choice(SYNS_ES["en_el_nivel"])
    if "{view}" in base:
        return random.choice(SYNS_ES["en_la_vista"])
    return ""

def find_context_level_en(base: str) -> str:
    if "{level}" in base:
        return random.choice(SYNS_EN["on_level"])
    if "{view}" in base:
        return random.choice(SYNS_EN["in_view"])
    return ""

def build_loc_clause_es(base: str) -> str:
    parts = []
    # Look for coordinate groups present in prompt and produce phrasing accordingly
    if all(k in base for k in ("{loc_x", "{loc_y}")):
        parts.append("en la posición ({loc_x:.2f}, {loc_y:.2f}{maybe_z})".replace("{maybe_z}", ", {loc_z:.2f}" if "{loc_z" in base else ""))
    if all(k in base for k in ("{start_x", "{start_y}")) and all(k in base for k in ("{end_x", "{end_y}")):
        parts.append("desde ({start_x:.2f},{start_y:.2f}{maybe_sz}) hasta ({end_x:.2f},{end_y:.2f}{maybe_ez})"
                     .replace("{maybe_sz}", ", {start_z:.2f}" if "{start_z" in base else "")
                     .replace("{maybe_ez}", ", {end_z:.2f}" if "{end_z" in base else ""))
    return (" " + random.choice(parts)) if parts else ""

def build_loc_clause_en(base: str) -> str:
    parts = []
    if all(k in base for k in ("{loc_x", "{loc_y}")):
        parts.append("at ({loc_x:.2f}, {loc_y:.2f}{maybe_z})".replace("{maybe_z}", ", {loc_z:.2f}" if "{loc_z" in base else ""))
    if all(k in base for k in ("{start_x", "{start_y}")) and all(k in base for k in ("{end_x", "{end_y}")):
        parts.append("from ({start_x:.2f},{start_y:.2f}{maybe_sz}) to ({end_x:.2f},{end_y:.2f}{maybe_ez})"
                     .replace("{maybe_sz}", ", {start_z:.2f}" if "{start_z" in base else "")
                     .replace("{maybe_ez}", ", {end_z:.2f}" if "{end_z" in base else ""))
    return (" " + random.choice(parts)) if parts else ""

def build_path_clause_es(base: str) -> str:
    if "{path_str}" in base or "{path}" in base:
        return "siguiendo la ruta {path_str}".replace("{path_str}", "{path_str}" if "{path_str}" in base else "{path}")
    return ""

def build_path_clause_en(base: str) -> str:
    if "{path_str}" in base or "{path}" in base:
        return "following the path {path_str}".replace("{path_str}", "{path_str}" if "{path_str}" in base else "{path}")
    return ""

def build_extra_clause_es(base: str) -> str:
    extras = []
    if "{top_level}" in base:
        extras.append("hasta '{top_level}'")
    if "{sill_height" in base:
        extras.append("con antepecho {sill_height:.2f} m")
    if "{diameter_mm" in base:
        extras.append("de {diameter_mm}mm")
    if "{height" in base and "altura" not in "".join(extras):
        extras.append("con altura {height:.2f} m")
    if "{slope" in base:
        extras.append("con pendiente {slope}°")
    if "{run_width" in base:
        extras.append("con un ancho de {run_width:.2f} m")
    if extras:
        return " " + ", ".join(extras)
    return ""

def build_extra_clause_en(base: str) -> str:
    extras = []
    if "{top_level}" in base:
        extras.append("up to '{top_level}'")
    if "{sill_height" in base:
        extras.append("with a sill height of {sill_height:.2f} m")
    if "{diameter_mm" in base:
        extras.append("with {diameter_mm}mm")
    if "{height" in base and "height" not in "".join(extras):
        extras.append("with height {height:.2f} m")
    if "{slope" in base:
        extras.append("with a {slope}° slope")
    if "{run_width" in base:
        extras.append("with a run width of {run_width:.2f} m")
    if extras:
        return " " + ", ".join(extras)
    return ""

def build_object_from_base_es(base: str) -> str:
    # Heuristic: look for known nouns; fallback to "elemento"
    candidates = [
        ("ducto", "un ducto"),
        ("tubería", "una tubería"),
        ("columna", "una columna"),
        ("viga", "una viga"),
        ("muro", "un muro"),
        ("baranda", "una baranda"),
        ("escalera", "una escalera"),
        ("luminaria", "una luminaria"),
        ("dispositivo", "un dispositivo eléctrico"),
        ("rejilla", "una terminal de aire"),
        ("ventana", "una ventana"),
        ("puerta", "una puerta"),
        ("plafón", "un plafón"),
        ("región rellena", "una región rellena"),
        ("leyenda", "una leyenda"),
        ("vista", "una vista"),
        ("piso", "un piso"),
        ("cimentación", "una cimentación"),
        ("barandal", "un barandal"),
        ("cercha", "una armadura"),
        ("riostra", "un arriostramiento"),
        ("charola", "una charola portacables"),
    ]
    base_l = base.lower()
    for key, label in candidates:
        if key in base_l:
            return label
    return "un elemento"

def build_object_from_base_en(base: str) -> str:
    candidates = [
        ("duct", "a duct"),
        ("pipe", "a pipe"),
        ("column", "a column"),
        ("beam", "a beam"),
        ("wall", "a wall"),
        ("railing", "a railing"),
        ("stair", "a stair"),
        ("fixture", "a light fixture"),
        ("device", "an electrical device"),
        ("grille", "an air terminal"),
        ("window", "a window"),
        ("door", "a door"),
        ("ceiling", "a ceiling"),
        ("region", "a filled region"),
        ("legend", "a legend"),
        ("view", "a view"),
        ("floor", "a floor"),
        ("foundation", "a foundation"),
        ("truss", "a truss"),
        ("brace", "a brace"),
        ("cabletray", "a cable tray"),
    ]
    base_l = base.lower()
    for key, label in candidates:
        if key in base_l:
            return label
    return "an element"

def capitalize_first(s: str) -> str:
    return s[0].upper() + s[1:] if s else s

def paraphrase_es(base_prompt: str, n: int = 10) -> List[str]:
    masked, mapping = freeze_placeholders(base_prompt)
    # Verb selection based on detected action words in the original Spanish prompt
    verb_candidates = []
    low = masked.lower()
    if any(v in low for v in ["coloca", "inserta", "ubica", "sitúa", "pon"]):
        verb_candidates = SYNS_ES["coloca"]
    elif any(v in low for v in ["dibuja", "traza", "modela"]):
        verb_candidates = SYNS_ES["dibuja"]
    elif any(v in low for v in ["crea", "genera", "añade", "haz", "construye"]):
        verb_candidates = SYNS_ES["crea"]
    else:
        verb_candidates = SYNS_ES["crea"] + SYNS_ES["coloca"]
    obj = build_object_from_base_es(low)
    variants = []
    for _ in range(n * 2):  # oversample then dedupe
        verb = random.choice(verb_candidates)
        verb_cap = capitalize_first(verb)
        context_clause = find_context_level_es(low)
        type_clause = make_type_clause_es(low)
        loc_clause = build_loc_clause_es(low)
        path_clause = build_path_clause_es(low)
        extra_clause = build_extra_clause_es(low)
        prelude = random.choice(["", "Por favor,", "Necesito que", "Quiero que"]).strip()
        pattern = random.choice(PATTERNS_ES)
        s = pattern.format(
            verb=verb,
            verb_cap=verb_cap,
            obj=obj,
            type_clause=(" " + type_clause) if type_clause else "",
            context_clause=(" " + context_clause) if context_clause else "",
            loc_clause=loc_clause,
            path_clause=(" " + path_clause) if path_clause else "",
            extra_clause=extra_clause,
            prelude=prelude
        )
        s = re.sub(r"\s+", " ", s).strip()
        variants.append(s)
    # Unmask placeholders
    variants = [unfreeze_placeholders(v, mapping) for v in variants]
    return ensure_min_unique(variants, n)

def paraphrase_en(base_prompt: str, n: int = 10) -> List[str]:
    masked, mapping = freeze_placeholders(base_prompt)
    low = masked.lower()
    if any(v in low for v in ["place", "insert", "put", "drop"]):
        verb_candidates = SYNS_EN["place"]
    elif any(v in low for v in ["draw", "trace", "model"]):
        verb_candidates = SYNS_EN["draw"]
    elif any(v in low for v in ["create", "generate", "add", "make", "build"]):
        verb_candidates = SYNS_EN["create"]
    else:
        verb_candidates = SYNS_EN["create"] + SYNS_EN["place"]
    obj = build_object_from_base_en(low)
    variants = []
    for _ in range(n * 2):
        verb = random.choice(verb_candidates)
        verb_cap = capitalize_first(verb)
        context_clause = find_context_level_en(low)
        type_clause = make_type_clause_en(low)
        loc_clause = build_loc_clause_en(low)
        path_clause = build_path_clause_en(low)
        extra_clause = build_extra_clause_en(low)
        prelude = random.choice(["", "Please", "I need you to", "I'd like you to"]).strip()
        pattern = random.choice(PATTERNS_EN)
        s = pattern.format(
            verb=verb,
            verb_cap=verb_cap,
            obj=obj,
            type_clause=(" " + type_clause) if type_clause else "",
            context_clause=(" " + context_clause) if context_clause else "",
            loc_clause=loc_clause,
            path_clause=(" " + path_clause) if path_clause else "",
            extra_clause=extra_clause,
            prelude=prelude
        )
        s = re.sub(r"\s+", " ", s).strip()
        variants.append(s)
    variants = [unfreeze_placeholders(v, mapping) for v in variants]
    return ensure_min_unique(variants, n)

# Minimal direct phrase "translator" ES<->EN for when the original is only in one language.
# We keep it conservative; placeholders remain masked.
PHRASEBOOK_ES2EN = [
    ("Crea", "Create"),
    ("Coloca", "Place"),
    ("Dibuja", "Draw"),
    ("Inserta", "Insert"),
    ("Quiero", "I want to"),
    ("Traza", "Trace"),
    ("en el nivel", "on level"),
    ("en la vista", "in view"),
    ("llamada", "called"),
    ("llamado", "called"),
    ("tipo", "type"),
    ("desde", "from"),
    ("hasta", "to"),
    ("con pendiente", "with a slope of"),
    ("con altura", "with height"),
    ("con un ancho de", "with a width of"),
    ("en la posición", "at"),
]
PHRASEBOOK_EN2ES = [
    ("Create", "Crea"),
    ("Place", "Coloca"),
    ("Draw", "Dibuja"),
    ("Insert", "Inserta"),
    ("Trace", "Traza"),
    ("on level", "en el nivel"),
    ("in view", "en la vista"),
    ("called", "llamada"),
    ("type", "tipo"),
    ("from", "desde"),
    ("to", "hasta"),
    ("with a slope of", "con pendiente"),
    ("with height", "con altura"),
    ("with a width of", "con un ancho de"),
    ("at", "en la posición"),
]

def naive_translate_es2en(text: str) -> str:
    masked, mapping = freeze_placeholders(text)
    out = masked
    for es, en in PHRASEBOOK_ES2EN:
        out = re.sub(rf"\b{re.escape(es)}\b", en, out)
    return unfreeze_placeholders(out, mapping)

def naive_translate_en2es(text: str) -> str:
    masked, mapping = freeze_placeholders(text)
    out = masked
    for en, es in PHRASEBOOK_EN2ES:
        out = re.sub(rf"\b{re.escape(en)}\b", es, out)
    return unfreeze_placeholders(out, mapping)

# --- Main processing --------------------------------------------------------

def expand_templates(input_path: str = INPUT, output_path: str = OUTPUT,
                     variants_per_lang: int = 10) -> None:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")
    total_in = 0
    total_out = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            action_key = obj.get("action_key") or obj.get("action") or ""
            templates = obj.get("templates") or [{"prompt": obj.get("prompt"), "plan": obj.get("plan")}]
            for tpl in templates:
                prompt = tpl.get("prompt", "")
                plan = tpl.get("plan", obj.get("plan"))
                if not prompt or not plan:
                    continue
                total_in += 1

                # Decide base language and generate both sets
                base_lang = detect_lang(prompt)
                if base_lang == "es":
                    es_variants = paraphrase_es(prompt, n=variants_per_lang)
                    # Build English base via naive translation of the first ES variant, then paraphrase
                    en_seed = naive_translate_es2en(es_variants[0])
                    en_variants = paraphrase_en(en_seed, n=variants_per_lang)
                else:
                    en_variants = paraphrase_en(prompt, n=variants_per_lang)
                    es_seed = naive_translate_en2es(en_variants[0])
                    es_variants = paraphrase_es(es_seed, n=variants_per_lang)

                # Write out ES
                for v in es_variants:
                    rec = {"action_key": action_key, "prompt": v, "plan": plan, "lang": "es"}
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total_out += 1

                # Write out EN
                for v in en_variants:
                    rec = {"action_key": action_key, "prompt": v, "plan": plan, "lang": "en"}
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total_out += 1

    print(f"Expanded {total_in} base templates into {total_out} variants.")
    print(f"Output written to: {output_path}")

if __name__ == "__main__":
    expand_templates()
