"""
Simple test: Does PyTorch GPU work on Windows?
"""
import torch
import time

print("=" * 60)
print("WINDOWS GPU TEST")
print("=" * 60)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Simple GPU computation test
    device = 'cuda'
    x = torch.randn(1000, 1000, device=device, dtype=torch.float64)
    y = torch.randn(1000, 1000, device=device, dtype=torch.float64)

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        z = torch.mm(x, y)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print(f"\nGPU matmul test: {elapsed:.4f} sec for 100 iterations")
    print("GPU WORKS!")
else:
    print("CUDA NOT AVAILABLE - check PyTorch installation")
