import torch
print(torch.__version__, torch.version.cuda)
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))