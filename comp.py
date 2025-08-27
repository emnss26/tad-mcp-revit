import torch, platform
print("Python:", platform.python_version())
print("Torch:", torch.__version__)
print("CUDA disponible:", torch.cuda.is_available())
print("Num GPUs:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"GPU{i}:", torch.cuda.get_device_name(i))
    print("  CC:", torch.cuda.get_device_capability(i))
print("Compilado con CUDA:", torch.version.cuda) # versión de CUDA a la que fue compilado PyTorch
print("BF16:", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None)