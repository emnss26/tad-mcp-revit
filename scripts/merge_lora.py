import argparse, os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Modelo base (p.ej microsoft/Phi-3-mini-4k-instruct)")
    ap.add_argument("--adapter", required=True, help="Carpeta del LoRA entrenado")
    ap.add_argument("--out", required=True, help="Carpeta destino del modelo fusionado")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto"
    )
    peft = PeftModel.from_pretrained(base, args.adapter)
    merged = peft.merge_and_unload()  # fusiona pesos LoRA en el base (fp16)

    os.makedirs(args.out, exist_ok=True)
    tok.save_pretrained(args.out)
    merged.save_pretrained(args.out)
    print(f"[ok] merged model saved to: {args.out}")

if __name__ == "__main__":
    main()