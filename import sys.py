import sys
import torch

print(f"Python Version: {sys.version.split()[0]}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device_id = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_id)
    vram_gb = torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 3)
    print(f"Locked onto Device [{device_id}]: {device_name}")
    print(f"Dedicated VRAM: {vram_gb:.2f} GB")
    
    # Quick tensor test on GPU
    x = torch.randn((1000, 1000), device="cuda")
    y = torch.matmul(x, x)
    print("CUDA Computation Test: PASSED")
else:
    print("WARNING: CUDA is NOT detected. Running on CPU.")