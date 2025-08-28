import os
import json
import random
from pathlib import Path
from faker import Faker

fake = Faker('es_MX')  # español México
fake_en = Faker('en_US')

TEMPLATES_FILE = Path("data/train_data_templates.jsonl")
OUTPUT_FILE = Path("data/train_data_expanded.jsonl")

# Variantes semánticas para frases comunes (ES/EN)
synonyms_es = {
    "crea": ["genera", "haz", "construye", "produce"],
    "coloca": ["inserta", "ubica", "pon", "añade"],
    "planta": ["vista de planta", "plano", "planta arquitectónica"],
    "puerta": ["acceso", "entrada", "hoja"],
    "ventana": ["abertura", "hueco", "cristal"]
}

synonyms_en = {
    "create": ["generate", "make", "build", "produce"],
    "place": ["insert", "put", "drop", "add"],
    "plan": ["floor plan", "layout", "blueprint"],
    "door": ["entry", "access", "leaf"],
    "window": ["opening", "glass", "pane"]
}

# Para nombres aleatorios de vistas, niveles, etc.
def random_name(prefix="L"):
    return f"{prefix}{random.randint(1, 20)}"

def random_level():
    return f"L{random.randint(1, 5)}"

def random_elevation():
    return round(random.uniform(2.5, 6.0), 2)

def replace_synonyms_es(prompt):
    words = prompt.split()
    return " ".join([random.choice(synonyms_es.get(w.lower(), [w])) for w in words])

def replace_synonyms_en(prompt):
    words = prompt.split()
    return " ".join([random.choice(synonyms_en.get(w.lower(), [w])) for w in words])

def generate_variants():
    expanded = []
    with open(TEMPLATES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            templates = obj.get("templates", [])
            for tpl in templates:
                prompt_base = tpl["prompt"]
                plan = tpl["plan"]
                
                for i in range(5):  # 5 variantes en español
                    prompt = replace_synonyms_es(prompt_base)
                    prompt = prompt.replace("{name}", random_name())
                    prompt = prompt.replace("{level}", random_level())
                    prompt = prompt.replace("{elevation}", str(random_elevation()))
                    expanded.append({
                        "prompt": prompt,
                        "response": plan
                    })
                
                for i in range(5):  # 5 variantes en inglés
                    prompt = replace_synonyms_en(prompt_base)
                    prompt = prompt.replace("{name}", random_name())
                    prompt = prompt.replace("{level}", random_level())
                    prompt = prompt.replace("{elevation}", str(random_elevation()))
                    expanded.append({
                        "prompt": prompt,
                        "response": plan
                    })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for ex in expanded:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"✅ Generadas {len(expanded)} variantes en {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_variants()
