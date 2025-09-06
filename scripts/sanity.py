import sys, platform, os
print("Python:", sys.version.split()[0])
print("Platform:", platform.platform())
print("Env: JAX_PLATFORMS=", os.getenv("JAX_PLATFORMS"),
      "PYTORCH_ENABLE_MPS_FALLBACK=", os.getenv("PYTORCH_ENABLE_MPS_FALLBACK"),
      "PYTORCH_MPS_HIGH_WATERMARK_RATIO=", os.getenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO"))
try:
    import torch
    print("PyTorch:", torch.__version__)
    print("  MPS available:", getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if torch.backends.mps.is_available():
        x = torch.randn(1_000, 1_000, device="mps")
        y = torch.randn(1_000, 1_000, device="mps")
        z = (x @ y).mean().item()
        print("  MPS matmul mean:", round(z, 6))
except Exception as e:
    print("PyTorch check error:", e)
try:
    import jax, jax.numpy as jnp
    print("JAX:", jax.__version__)
    print("  Devices:", [d.device_kind for d in jax.devices()])
    a = jnp.ones((1024,)); b = jnp.ones((1024,))
    print("  JAX dot:", jnp.dot(a, b).block_until_ready())
except Exception as e:
    print("JAX check error:", e)
try:
    import tensorflow as tf
    print("TensorFlow:", tf.__version__)
    print("  GPUs:", tf.config.list_physical_devices('GPU'))
except Exception as e:
    print("TensorFlow check error:", e)
try:
    import mlx.core as mx
    print("MLX:", mx.__version__ if hasattr(mx, "__version__") else "installed")
    print("  Default device:", getattr(mx, "default_device", lambda: "n/a")())
except Exception as e:
    print("MLX check error:", e)
