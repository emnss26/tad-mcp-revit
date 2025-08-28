import json
import sys
import pathlib
import argparse

def main():
    """
    Valida un archivo línea por línea para asegurar que cada una es un objeto JSON válido.
    Maneja automáticamente el problema común de UTF-8 BOM en Windows.
    """
    parser = argparse.ArgumentParser(description="Valida cada línea de un archivo para asegurar que es un JSON válido.")
    parser.add_argument(
        "input_file", 
        help="Ruta al archivo JSONL que se va a validar (ej: data/train_data_templates.jsonl)"
    )
    args = parser.parse_args()

    p = pathlib.Path(args.input_file)

    if not p.exists():
        print(f"Error: El archivo no se encontró en la ruta especificada: '{p}'", file=sys.stderr)
        sys.exit(1)

    bad_lines = 0
    total_lines = 0

    # LA CORRECCIÓN CLAVE: Usar encoding="utf-8-sig" en lugar de "utf-8".
    # Esto maneja automáticamente el BOM (Byte Order Mark) que algunos editores de Windows añaden.
    with open(p, "r", encoding="utf-8-sig") as f:
        for i, line in enumerate(f, start=1):
            s = line.strip()
            if not s: 
                continue
            
            total_lines += 1
            try:
                json.loads(s)
            except json.JSONDecodeError as e:
                bad_lines += 1
                print(f"Linea {i}: JSON invalido -> {e} | {s[:120]}...")

    print("\n--- Reporte de Validación ---")
    if bad_lines == 0:
        print(f"✅ ¡Éxito! Todas las {total_lines} líneas son JSON válidas.")
    else:
        print(f"❌ Se encontraron {bad_lines} línea(s) inválida(s) de un total de {total_lines}.")

if __name__ == "__main__":
    main()