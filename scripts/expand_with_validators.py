import json, re, random
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES_FILE = ROOT / "data" / "train_data_templates.jsonl"
TOOLS_FILE     = ROOT / "data" / "tools_catalog.json"
OUTPUT_FILE    = ROOT / "data" / "train_data_expanded.jsonl"

# ---------- utilidades de idioma ----------
ES_LEVEL_PREFIX = ["Nivel", "Planta", "Piso"]
EN_LEVEL_PREFIX = ["Level", "Floor"]

ES_VIEW_SUFFIX  = ["- Arquitectura", "- Estructura", "- MEP", "- General"]
EN_VIEW_SUFFIX  = ["- Arch", "- Str", "- MEP", "- General"]

ES_SHEET_TTL    = ["Planta", "Corte", "Fachada", "Detalles", "General"]
EN_SHEET_TTL    = ["Plan", "Section", "Elevation", "Details", "General"]

# ---------- saneadores ----------
BAD_CHARS = r'[^A-Za-z0-9ÁÉÍÓÚÜÑáéíóúüñ\-\s_\.\(\)\#\:\/]'
BAD_CHARS_RE = re.compile(BAD_CHARS)

def sanitize_name(s: str, maxlen=60):
    s = s.strip()
    s = BAD_CHARS_RE.sub("", s)          # elimina raros
    s = re.sub(r"\s{2,}", " ", s)        # espacios múltiples
    return s[:maxlen]

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def rnd_bool(): return random.choice([True, False])

# ---------- generadores por placeholder (fallbacks seguros) ----------
def gen_level_name(lang="es"):
    n = random.randint(1, 8)
    if lang == "es":
        return f"{random.choice(ES_LEVEL_PREFIX)} {n:02d}"
    else:
        return f"{random.choice(EN_LEVEL_PREFIX)} {n:02d}"

def gen_view_name(lang="es"):
    base = gen_level_name(lang)
    suf  = random.choice(ES_VIEW_SUFFIX if lang=="es" else EN_VIEW_SUFFIX)
    return sanitize_name(f"{base} {suf}")

def gen_sheet_number():
    # A-101, S-201, M-301...
    series = random.choice(["A", "S", "M", "E", "G"])
    num = random.randint(1, 499)
    return f"{series}-{num:03d}"

def gen_sheet_title(lang="es"):
    word = random.choice(ES_SHEET_TTL if lang=="es" else EN_SHEET_TTL)
    return f"{word} {random.randint(1, 5)}"

def gen_elevation():     return round(random.uniform(0.00, 18.00), 2)
def gen_height():        return round(random.uniform(2.40, 6.50), 2)
def gen_offset():        return round(random.uniform(-1.00, 1.00), 2)
def gen_diameter_mm():   return random.choice([100,125,150,200,250,300,350,400])
def gen_spacing_mm():    return random.choice([100,150,200,250,300,350,400])
def gen_count():         return random.randint(2, 10)
def gen_scale():         return random.choice([50, 75, 100, 150, 200])
def gen_angle():         return random.choice([15,30,45,60,75,90,120,180])
def gen_slope_deg():     return random.choice([5,10,15,20,25,30])
def gen_u():             return round(random.uniform(0.30, 3.00), 2)
def gen_v():             return round(random.uniform(0.30, 3.00), 2)

def gen_xy(maxm=50.0):
    return round(random.uniform(0.00, maxm), 2)

def gen_xyz():
    return gen_xy(), gen_xy(), round(random.uniform(0.00, 6.00), 2)

def gen_path_points(nmin=2, nmax=5):
    n = random.randint(nmin, nmax)
    pts = []
    x, y = gen_xy(), gen_xy()
    z = round(random.uniform(0.0,3.0),2)
    for i in range(n):
        pts.append({"x": round(x+random.uniform(-3,3),2),
                    "y": round(y+random.uniform(-3,3),2),
                    "z": z})
    return pts

# Map rápido por nombre de placeholder -> generador
GEN_BY_NAME = {
    # nombres
    "name": lambda lang: sanitize_name(random.choice([
        gen_view_name(lang), gen_sheet_title(lang), gen_level_name(lang)
    ])),
    "new_name": lambda lang: sanitize_name(gen_view_name(lang)),
    "from_": lambda lang: sanitize_name(gen_view_name(lang)),
    "to": lambda lang: sanitize_name(gen_view_name(lang)),
    "level_name": lambda lang: gen_level_name(lang),

    # niveles / vistas
    "level": lambda lang: gen_level_name(lang),
    "base_level": lambda lang: gen_level_name(lang),
    "top_level": lambda lang: gen_level_name(lang),
    "bottom": lambda lang: gen_level_name(lang),
    "top": lambda lang: gen_level_name(lang),
    "view": lambda lang: gen_view_name(lang),
    "view_name": lambda lang: gen_view_name(lang),
    "source": lambda lang: gen_view_name(lang),

    # hojas
    "sheet_number": lambda lang: gen_sheet_number(),
    "number": lambda lang: gen_sheet_number(),

    # disciplinas (en)
    "discipline_en": lambda lang: random.choice(["Architectural","Structural","Mechanical","Electrical","Coordination"]),
    "discipline":    lambda lang: random.choice(["Arquitectura","Estructura","Mecánica","Eléctrica","Coordinación"]),

    # enums comunes
    "category_en": lambda lang: random.choice([
        "Walls","Floors","Doors","Windows","Structural Columns","Ducts","Pipes","Electrical Fixtures","Mechanical Equipment","Views"
    ]),
    "category": lambda lang: random.choice([
        "Muros","Pisos","Puertas","Ventanas","Columnas Estructurales","Ductos","Tuberías","Aparatos Eléctricos","Equipos Mecánicos","Vistas"
    ]),
    "type": lambda lang: random.choice([
        "Genérico 200mm","Muro de concreto 15cm","Viga W610x125","Columna HSS 200x200","Ducto Redondo",
        "PVC 2in","Charola 300mm","Luminaria Empotrable 2x4","Barandal Simple"
    ]),
    "mullion_type": lambda lang: random.choice(["Rectangular 50mm","Tubular 75mm","Cuchilla 100mm"]),
    "titleblock_type": lambda lang: random.choice(["A1 Vertical","A1 Horizontal","A0","A2"]),
    "phase_filter": lambda lang: random.choice(["Mostrar completo","Nuevo","Existente","Demolición","Show All"]),
    "tag_type": lambda lang: random.choice(["Keynote Tag","Door Tag","Window Tag","Generic Tag"]),
    "fill_type": lambda lang: random.choice(["Solid Black","Solid Fill","Diagonal Hatch","Crosshatch"]),

    # numéricos / formatos
    "elevation": lambda lang: gen_elevation(),
    "height": lambda lang: gen_height(),
    "base_offset": lambda lang: gen_offset(),
    "top_offset": lambda lang: gen_offset(),
    "height_offset": lambda lang: round(random.uniform(0.20, 1.50), 2),
    "run_width": lambda lang: round(random.uniform(1.00, 2.40), 2),
    "diameter_mm": lambda lang: gen_diameter_mm(),
    "spacing_mm": lambda lang: gen_spacing_mm(),
    "count": lambda lang: gen_count(),
    "scale": lambda lang: gen_scale(),
    "angle_deg": lambda lang: gen_angle(),
    "slope": lambda lang: gen_slope_deg(),
    "sill_height": lambda lang: round(random.uniform(0.50, 1.40), 2),

    # coordenadas y tamaños
    "origin_x": lambda lang: gen_xy(), "origin_y": lambda lang: gen_xy(), "origin_z": lambda lang: 0.0,
    "width": lambda lang: round(random.uniform(2.0, 15.0), 2),
    "depth": lambda lang: round(random.uniform(2.0, 15.0), 2),

    "start_x": lambda lang: gen_xy(), "start_y": lambda lang: gen_xy(), "start_z": lambda lang: round(random.uniform(0,3),2),
    "end_x":   lambda lang: gen_xy(), "end_y":   lambda lang: gen_xy(), "end_z":   lambda lang: round(random.uniform(0,3),2),

    "loc_x": lambda lang: gen_xy(), "loc_y": lambda lang: gen_xy(), "loc_z": lambda lang: round(random.uniform(0,3),2),

    "point_x": lambda lang: gen_xy(), "point_y": lambda lang: gen_xy(), "point_z": lambda lang: 0.0,

    "direction_name": lambda lang: random.choice(["Norte","Sur","Este","Oeste"]) if lang=="es" else random.choice(["North","South","East","West"]),
    "direction_x": lambda lang: random.choice([1,0,-1,0]),
    "direction_y": lambda lang: random.choice([0,1,0,-1]),
    "direction_z": lambda lang: 0,
    "z_coord":     lambda lang: 0.0,

    # genéricos
    "param_name": lambda lang: random.choice(["Comentarios","Mark","Tipo","Material estructural","Top Offset","Base Offset"]),
    "value":      lambda lang: sanitize_name(random.choice(["OK","Revisar","Aprobado","Fase 2","Eje A-3","Zona Norte"])),
    "schedule_name": lambda lang: sanitize_name(random.choice(["Puertas", "Ventanas", "Muros", "Vigas", "Equipos MEP"])),
    "field_name":    lambda lang: sanitize_name(random.choice(["Cantidad","Costo","Área","Volumen","Peso"])),
    "formula":       lambda lang: 'Cantidad * Costo',
    "workset_name":  lambda lang: sanitize_name(random.choice(["Arquitectura","Estructura","MEP","Anotaciones"])),
    "scope_box_name":lambda lang: sanitize_name(random.choice(["Sector A","Sector B","Núcleo","Zona Patio"])),
    "template_name": lambda lang: sanitize_name(random.choice(["ARQ - Planta","STR - Planta","MEP - Coordinación"])),
    "sheet_name":    lambda lang: sanitize_name(gen_sheet_title(lang)),
    "pattern":       lambda lang: r"Copia de (.+)",
    "replace":       lambda lang: r"\1 - Copia",
    "rebar_type":    lambda lang: sanitize_name(random.choice(["#3","#4","#5","#6","#8"])),
    "rgb_value":     lambda lang: [random.randint(0,255),random.randint(0,255),random.randint(0,255)],
    "names":         lambda lang: [sanitize_name(gen_view_name(lang)) for _ in range(random.randint(1,3))],
    "views":         lambda lang: [sanitize_name(gen_view_name(lang)) for _ in range(random.randint(1,4))],
    "fields":        lambda lang: [random.choice(["Name","Type","Level","Mark","Comments","Workset","Phase"]) for _ in range(random.randint(3,5))],
    "profile_points":lambda lang: gen_path_points(),
    "path":          lambda lang: gen_path_points(),
    "host_element_id": lambda lang: 12345,  # placeholder numérico válido
    "face_index":      lambda lang: random.randint(0,3),
    "offset_mm":       lambda lang: random.choice([25, 50, 75, 100, 125, 150]),
    "offset":          lambda lang: round(random.uniform(0.20, 3.00), 2),

    "from_link_name": lambda lang: sanitize_name(random.choice(["Base Topográfica","Modelo Arquitectura","Estructura Link"]))
}

# ---------- helpers de render ----------
FMT_FIELD_RE = re.compile(r"\{([A-Za-z0-9_]+)(:[^\}]+)?\}")

def fill_placeholders_in_string(s: str, vars_dict: dict):
    """Rellena placeholders con soporte para formato tipo : .2f"""
    def _repl(m):
        key = m.group(1)
        fmt = m.group(2)  # e.g. ':.2f'
        val = vars_dict.get(key, m.group(0))
        if fmt and isinstance(val, (int, float)):
            try:
                return ("{"+fmt+"}").format(val)
            except Exception:
                return str(val)
        return str(val)
    return FMT_FIELD_RE.sub(_repl, s)

def deep_render(obj, vars_dict):
    """Aplica render recursivo a dict/list/str."""
    if isinstance(obj, dict):
        return {k: deep_render(v, vars_dict) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deep_render(v, vars_dict) for v in obj]
    elif isinstance(obj, str):
        return fill_placeholders_in_string(obj, vars_dict)
    else:
        return obj

# ---------- generators guiados por tools_catalog ----------
def load_tools():
    if TOOLS_FILE.exists():
        with open(TOOLS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"tools": []}

def guess_type_by_name(name):
    name_l = name.lower()
    if any(k in name_l for k in ["x","y","z"]):   return "number"
    if any(k in name_l for k in ["width","depth","height","offset","elevation","diameter","spacing","angle","slope","scale"]):
        return "number"
    if any(k in name_l for k in ["count","index"]): return "integer"
    if "name" in name_l or "title" in name_l or "type" in name_l or "level" in name_l or "view" in name_l or "sheet" in name_l:
        return "string"
    return "string"

def sample_value_for(name, lang="es"):
    gen = GEN_BY_NAME.get(name)
    if gen:
        val = gen(lang)
    else:
        # fallback por tipo inferido
        t = guess_type_by_name(name)
        if t == "number":
            val = round(random.uniform(0.1, 10.0), 2)
        elif t == "integer":
            val = random.randint(1, 10)
        else:
            val = sanitize_name(f"{name}_{random.randint(1,999)}")
    # sanea strings
    if isinstance(val, str):
        val = sanitize_name(val)
    return val

def collect_all_placeholders(template_prompt: str, plan_obj: dict):
    """Detecta todas las llaves {var} presentes en prompt y plan para rellenar consistentemente."""
    s = json.dumps(plan_obj, ensure_ascii=False)
    keys = set(m.group(1) for m in FMT_FIELD_RE.finditer(template_prompt))
    keys |= set(m.group(1) for m in FMT_FIELD_RE.finditer(s))
    return sorted(keys)

# ---------- helpers de selección y normalización de prompts ----------
def plan_needs_selection(plan):
    """Detecta si el plan usa selección interactiva."""
    if isinstance(plan, list):
        for step in plan:
            if isinstance(step, dict) and step.get("action") == "ui.prompt_for_selection":
                return True
    return False

# Reemplazos en español
ES_REPLACEMENTS = [
    (r"\bPermíteme seleccionar\b", "Selecciona"),
    (r"\bPermiteme seleccionar\b", "Selecciona"),
    (r"\bVoy a seleccionar\b", "Selecciona"),
    (r"\bCrea la hoja\b", "Crea un plano"),
    (r"\btitleblock\b", "membrete"),
    (r"\bOrdena el schedule\b", "Ordena la tabla"),
    (r"\bschedule\b", "tabla"),
    (r"\bCrea una baranda\b", "Crea un barandal"),
    (r"\bAñade los worksets\b", "Crea los worksets"),
    (r"\bAñade los subproyectos\b", "Crea los subproyectos"),
]

def rewrite_prompt_es(prompt: str, needs_sel: bool) -> str:
    s = prompt

    # reemplazos léxicos
    for pat, rep in ES_REPLACEMENTS:
        s = re.sub(pat, rep, s, flags=re.IGNORECASE)

    # si el plan pide selección, forzamos estilo imperativo
    if needs_sel:
        s = re.sub(r"(?i)^\s*(Permíteme|Permiteme|Voy a)\s+seleccionar(.*)$", r"Selecciona\2", s)
        s = re.sub(r"(?i)\bpermíteme\b", "Selecciona", s)
        s = re.sub(r"(?i)\bpermiteme\b", "Selecciona", s)
        # suaviza estructuras "Quiero ... seleccionar ..."
        s = re.sub(r"(?i)^\s*Quiero\s+(.*?seleccionar)", "Selecciona", s)
        # limpia muletillas
        s = s.replace("el siguiente", "un").replace("la siguiente", "una").replace("que elija", "")

        # Si no comienza con un verbo claro y no contiene "Selecciona", prepende "Selecciona"
        if not re.search(r"(?i)\bSelecciona\b", s):
            s = "Selecciona " + s.lstrip().capitalize()

    # retoques
    s = s.replace("  ", " ").strip()
    return s

# Traducción rápida ES->EN (prompts)
EN_MAP = [
    (r"\bCrea\b", "Create"),
    (r"\bColoca\b", "Place"),
    (r"\bInserta\b", "Insert"),
    (r"\bDibuja\b", "Draw"),
    (r"\bTraza\b", "Draw"),
    (r"\bEtiqueta\b", "Tag"),
    (r"\bOrdena\b", "Sort"),
    (r"\bRenombra\b", "Rename"),
    (r"\bMuestra\b", "Show"),
    (r"\bOculta\b", "Hide"),

    (r"\bplano\b", "sheet"),
    (r"\bhoja\b", "sheet"),
    (r"\bvista\b", "view"),
    (r"\bleyenda\b", "legend"),
    (r"\bsección\b", "section"),
    (r"\belevación\b", "elevation"),
    (r"\bplanta\b", "plan"),

    (r"\bmembrete\b", "titleblock"),
    (r"\btabla\b", "schedule"),
    (r"\bbarandal\b", "railing"),
    (r"\barriostramiento\b", "brace"),
    (r"\bcercha\b", "truss"),
    (r"\bmuros?\b", "walls"),
    (r"\bductos?\b", "ducts"),
    (r"\btuber[ií]as?\b", "pipes"),
    (r"\bworksets?\b", "worksets"),
]

def translate_prompt_es_to_en(prompt_es: str, needs_sel: bool) -> str:
    # primero normalizamos en ES (para partir de un texto limpio)
    s = rewrite_prompt_es(prompt_es, needs_sel)

    # selección a "Select ..."
    s = re.sub(r"(?i)^\s*Selecciona\b", "Select", s)
    s = re.sub(r"(?i)\bSelecciona\b", "Select", s)

    # mapeo léxico básico
    for pat, rep in EN_MAP:
        s = re.sub(pat, rep, s, flags=re.IGNORECASE)

    # espacios
    s = s.replace("  ", " ").strip()
    return s

# ---------- expansión ----------
def expand_templates(variants_per_lang=5):
    tools = load_tools()
    tool_index = {}
    for t in tools.get("tools", []):
        tool_index[t.get("action")] = t

    out = []
    n_read = 0

    with open(TEMPLATES_FILE, "r", encoding="utf-8-sig") as f: # Usar utf-8-sig por si acaso
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            action_key = obj.get("action_key")
            templates = obj.get("templates", [])
            for tpl in templates:
                prompt_base_template = tpl["prompt"]
                plan_template = tpl["plan"]

                placeholder_names = collect_all_placeholders(prompt_base_template, plan_template)
                needs_sel = plan_needs_selection(plan_template)

                # --- Generación en Español ---
                for _ in range(variants_per_lang):
                    lang = "es"
                    values = {k: sample_value_for(k, lang) for k in placeholder_names}
                    
                    prompt_es_raw = fill_placeholders_in_string(prompt_base_template, values)
                    prompt_es_final = rewrite_prompt_es(prompt_es_raw, needs_sel)
                    
                    # Rellenar el plan con los mismos valores
                    response_es = deep_render(plan_template, values)

                    # Asegurarse que el plan es una lista
                    if isinstance(response_es, dict):
                        response_es = [response_es]
                    
                    out.append({"prompt": prompt_es_final, "response": {"plan": response_es}})

                # --- Generación en Inglés ---
                for _ in range(variants_per_lang):
                    lang = "en"
                    # Generamos un nuevo set de valores aleatorios para inglés
                    values_en = {k: sample_value_for(k, lang) for k in placeholder_names}

                    # Rellenamos la plantilla original en español con los valores en inglés
                    prompt_es_filled_with_en_values = fill_placeholders_in_string(prompt_base_template, values_en)
                    
                    # Ahora traducimos ese prompt ya completo
                    prompt_en_final = translate_prompt_es_to_en(prompt_es_filled_with_en_values, needs_sel)
                    
                    # Rellenamos el plan con los valores en inglés
                    response_en = deep_render(plan_template, values_en)

                    # Asegurarse que el plan es una lista
                    if isinstance(response_en, dict):
                        response_en = [response_en]
                        
                    out.append({"prompt": prompt_en_final, "response": {"plan": response_en}})
                
                n_read += 1

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for ex in out:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"✅ Plantillas procesadas: {n_read}")
    print(f"✅ Ejemplos generados: {len(out)}")
    print(f"📄 Archivo: {OUTPUT_FILE}")
    
if __name__ == "__main__":
    random.seed(42)
    expand_templates(variants_per_lang=5)  # 5 ES + 5 EN = 10 por template
