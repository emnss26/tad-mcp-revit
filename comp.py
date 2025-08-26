import torch, platform
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("is_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    print("Capability:", torch.cuda.get_device_capability(0))
    a = torch.randn(1024,1024, device="cuda")
    b = torch.mm(a, a.t())
    print("Matmul OK:", b.shape)