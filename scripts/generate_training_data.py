from __future__ import annotations
import os
import json
import random
import argparse
import time
import re
from pathlib import Path
from typing import Any, Dict, List

# --- CARGAR VARIABLES DE ENTORNO ---
try:
    from dotenv import load_dotenv
    dotenv_path = Path(__file__).resolve().parents[1] / '.env'
    if dotenv_path.exists():
        print(f"Cargando variables de entorno desde: {dotenv_path}")
        load_dotenv(dotenv_path=dotenv_path)
    else:
        print("Archivo .env no encontrado en la raíz del proyecto, se usarán las variables ya existentes.")
except ImportError:
    print("ADVERTENCIA: 'python-dotenv' no está instalado. No se pudo cargar el archivo .env.")

# --- CONFIGURACIÓN DE IA ---
try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# --- POOLS DE DATOS PARA GENERACIÓN ---
DATA_POOLS = {
    "es": {
        "levels": ["Nivel 1", "Planta Baja", "Piso 2", "Nivel de Acceso", "Azotea", "Sótano 1", "Mezzanine"],
        "views": ["Planta Nivel 1 - Arquitectura", "Sección Longitudinal A-A", "Alzado Norte", "Vista 3D - Estructura", "Plano de Plafones Nivel 2"],
        "wall_types": ["Muro Básico: Genérico - 200mm", "Muro de Concreto 15cm", "Tablaroca - 1 cara", "Muro Cortina"],
        "door_types": ["Puerta Simple: 0915 x 2134mm", "Puerta Doble Vidriada", "Puerta de Emergencia"],
        "beam_types": ["Viga I de Acero: W460x52", "Trabe de Concreto: 30x60cm", "Viga de Celosía: 12m"],
        "categories": ["Muros", "Puertas", "Ventanas", "Pisos", "Plafones", "Columnas", "Trabes"],
        "categories_en": ["Walls", "Doors", "Windows", "Floors", "Ceilings", "Structural Columns", "Structural Framing"],
        "colors": ["rojo", "verde", "azul", "amarillo", "naranja"],
        "rgb_values": [[255,0,0], [0,255,0], [0,0,255], [255,255,0], [255,165,0]]
    },
    "en": {
        "levels": ["Level 1", "Ground Floor", "Floor 2", "Entry Level", "Roof", "Basement 1", "Mezzanine"],
        "views": ["Level 1 - Architectural Plan", "Longitudinal Section A-A", "North Elevation", "3D View - Structural", "Level 2 Ceiling Plan"],
        "wall_types": ["Basic Wall: Generic - 200mm", "Concrete Wall 15cm", "Drywall - 1 side", "Curtain Wall"],
        "door_types": ["Single-Flush: 0915 x 2134mm", "Double-Glass Door", "Emergency Exit Door"],
        "beam_types": ["Steel W-Section: W460x52", "Concrete Beam: 30x60cm", "Truss Beam: 12m"],
        "categories": ["Walls", "Doors", "Windows", "Floors", "Ceilings", "Columns", "Beams"],
        "categories_en": ["Walls", "Doors", "Windows", "Floors", "Ceilings", "Structural Columns", "Structural Framing"],
        "colors": ["red", "green", "blue", "yellow", "orange"],
        "rgb_values": [[255,0,0], [0,255,0], [0,0,255], [255,255,0], [255,165,0]]
    }
}

# --- FUNCIONES AUXILIARES DE GENERACIÓN Y IA ---

def _get_random_coords(z_range=(0.0, 15.0)):
    return {
        "x": round(random.uniform(-10.0, 50.0), 2),
        "y": round(random.uniform(-10.0, 50.0), 2),
        "z": round(random.uniform(z_range[0], z_range[1]), 2)
    }

def _get_random_value(key: str, lang: str):
    pool = DATA_POOLS[lang]
    if key in pool: return random.choice(pool[key])
    if key.endswith(('_x', '_y', '_z')): return _get_random_coords()[key[-1]]
    if "name" in key: return f"Generated-{lang.upper()}-{random.randint(100, 999)}"
    if "fields" in key: return random.sample(["Marca", "Tipo", "Nivel", "Comentarios", "Área"], k=random.randint(2, 4))
    if "path" in key or "profile" in key: return [_get_random_coords() for _ in range(random.randint(3, 5))]
    return f"VALUE_FOR_{key.upper()}"

def _rephrase_with_google_ai(base_prompt: str, api_key: str) -> List[str]:
    if not genai:
        print(" -- ADVERTENCIA: La IA está deshabilitada. Instala 'google-generativeai' para activarla.")
        return [base_prompt]
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        meta_prompt = f"""
        Eres un asistente de generación de datos para un agente de IA en Autodesk Revit.
        Tu tarea es reformular la siguiente solicitud de usuario de 5 maneras diferentes y naturales:
        - 2 variaciones adicionales en español, como lo diría un arquitecto o ingeniero en México.
        - 3 variaciones en inglés, como lo diría un profesional de AEC de EE. UU.
        REGLAS CRÍTICAS:
        - Tu respuesta debe ser ÚNICAMENTE un objeto JSON con una clave "variations" (una lista de 5 strings).
        - No incluyas explicaciones ni markdown.
        - Mantén todos los valores específicos (nombres, números, coordenadas) exactamente iguales.
        Solicitud original: "{base_prompt}"
        """
        response = model.generate_content(meta_prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
        variations_obj = json.loads(cleaned_response)
        # Devolvemos el prompt original + las 5 variaciones generadas por la IA
        return [base_prompt] + variations_obj.get("variations", [])
    except Exception as e:
        print(f" -- ADVERTENCIA: Falló la generación con Google AI. Usando solo el prompt base. Error: {e}")
        return [base_prompt]

def _rephrase_with_lm_studio(base_prompt: str) -> List[str]:
    # (La función de LM Studio se mantiene igual, no es necesario cambiarla)
    ...

def _fill_template(template: Any, variables: Dict[str, Any]) -> Any:
    if isinstance(template, str):
        try:
            return template.format_map(variables)
        except KeyError:
            return template
    if isinstance(template, dict):
        return {k: _fill_template(v, variables) for k, v in template.items()}
    if isinstance(template, list):
        return [_fill_template(i, variables) for i in template]
    return template

# --- LÓGICA PRINCIPAL DE GENERACIÓN ---

def generate_dataset(templates_path: str, num_variants: int, ai_provider: str | None, api_key: str):
    print(f"Cargando plantillas desde: {templates_path}")
    with open(templates_path, "r", encoding="utf-8-sig") as f:
        templates = [json.loads(line) for line in f if line.strip()]
    
    print(f"Se cargaron {len(templates)} plantillas de acción.")
    print(f"Objetivo: {num_variants} variantes.")
    if ai_provider:
        print(f"Modo IA activado con el proveedor: {ai_provider.upper()}")
    
    dataset = []
    
    # Pre-calculamos todos los placeholders una sola vez
    all_placeholders = set()
    for group in templates:
        for pair in group.get('templates', []):
            json_str = json.dumps(pair)
            all_placeholders.update(re.findall(r'\{(\w+)', json_str))

    # Bucle principal hasta alcanzar el número deseado de variantes
    while len(dataset) < num_variants:
        template_group = random.choice(templates)
        template_pair = random.choice(template_group["templates"])
        
        # Generamos un set de datos aleatorios. No asignamos idioma todavía.
        variables = {ph: _get_random_value(ph, random.choice(["es", "en"])) for ph in all_placeholders}

        try:
            # Siempre rellenamos la plantilla de prompt en español primero
            base_prompt_es = _fill_template(template_pair["prompt"], variables)
            
            all_prompts = []
            if ai_provider == 'google':
                # La IA nos devolverá una mezcla de prompts en español e inglés
                all_prompts = _rephrase_with_google_ai(base_prompt_es, api_key)
            elif ai_provider == 'lmstudio':
                all_prompts = _rephrase_with_lm_studio(base_prompt_es)
            else:
                # Si no hay IA, generamos manualmente uno en español y otro en inglés
                base_prompt_en = _fill_template(template_pair["prompt"].replace("Crea", "Create").replace("nivel", "level"), variables) # Traducción simple como fallback
                all_prompts = [base_prompt_es, base_prompt_en]

            for prompt in all_prompts:
                final_plan_body = _fill_template(template_pair["plan"], variables)
                if not isinstance(final_plan_body, list):
                    final_plan_body = [final_plan_body]

                final_plan_obj = {
                    "version": "tad-dsl/0.2",
                    "context": {"revit_version": "2025", "units": "SI"},
                    "plan": final_plan_body
                }
                
                response_str = json.dumps(final_plan_obj, ensure_ascii=False)
                dataset.append({"prompt": prompt.strip(), "response": response_str})
                
                if len(dataset) % 100 == 0:
                    print(f"  ... {len(dataset)} / {num_variants} variantes generadas ...")
                
                if len(dataset) >= num_variants:
                    break

        except Exception as e:
            # print(f" -- Saltando plantilla '{template_group['action_key']}'. Error: {e}")
            pass
    
    print("Randomizando el dataset final...")
    random.shuffle(dataset)
    return dataset[:num_variants]

def main():
    parser = argparse.ArgumentParser(description="Generador de dataset de entrenamiento a partir de plantillas.")
    parser.add_argument("--templates", default="data/train_data_templates.jsonl", help="Ruta al archivo de plantillas JSONL.")
    parser.add_argument("--output", default="data/train_dataset_6k.jsonl", help="Ruta del archivo de salida del dataset.")
    parser.add_argument("--num-variants", type=int, default=6000, help="Número total de ejemplos a generar.")
    parser.add_argument("--use-ai", choices=['google', 'lmstudio'], default='google', help="Activa el uso de IA para generar variaciones lingüísticas.")
    
    args = parser.parse_args()
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if args.use_ai == 'google' and not api_key:
        print("ERROR: La opción --use-ai 'google' requiere que la variable de entorno GOOGLE_API_KEY esté configurada.")
        return

    start_time = time.time()
    
    final_dataset = generate_dataset(args.templates, args.num_variants, args.use_ai, api_key)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in final_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    end_time = time.time()
    
    print("\n¡Proceso completado!")
    print(f"  ✓ Se generaron {len(final_dataset)} variantes.")
    print(f"  ✓ El dataset se guardó en: {args.output}")
    print(f"  ✓ Tiempo total: {end_time - start_time:.2f} segundos.")

if __name__ == "__main__":
    main()