"""Check if CUDA is actually working."""
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")

    # Test actual GPU computation
    try:
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')

        import time
        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(100):
            z = torch.mm(x, y)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        print(f"\nGPU matmul test: {elapsed:.4f} sec for 100 iterations")
        print(f"Tensor device: {z.device}")
        print("✅ GPU computation works!")
    except Exception as e:
        print(f"❌ GPU computation failed: {e}")
else:
    print("❌ CUDA not available")
