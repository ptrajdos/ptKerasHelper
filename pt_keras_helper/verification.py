
import os

BACKENDS = ("tensorflow", "torch", "jax")


def test_backend(backend):
    print(f"\n=== Backend: {backend} ===")

    os.environ["KERAS_BACKEND"] = backend

    try:
        import keras

        print(f"Keras version: {keras.__version__}")
        print(f"Selected backend: {keras.backend.backend()}")

        if backend == "tensorflow":
            import tensorflow as tf

            print(f"TF version: {tf.__version__}")
            print("Build:", tf.sysconfig.get_build_info())
            print("CPUs:")
            for device in tf.config.list_physical_devices("CPU"):
                print(f"  {device}")

            print("GPUs:")
            for device in tf.config.list_physical_devices("GPU"):
                print(f"  {device}")

        elif backend == "torch":
            import torch

            print(f"PyTorch version: {torch.__version__}")
            print("CPU: available")
            print(f"CUDA available: {torch.cuda.is_available()}")
            

            if torch.cuda.is_available():
                print("CUDA version:", torch.version.cuda)
                print(f"CUDA devices: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    print("Capability:", torch.cuda.get_device_capability())
                    print("Architectures:", torch.cuda.get_arch_list())
                    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

        elif backend == "jax":
            import jax

            print(f"JAX version: {jax.__version__}")
            print("Devices:")
            for device in jax.devices():
                print(f"  {device}")

    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")


def main():
    for backend in BACKENDS:
        test_backend(backend)


if __name__ == "__main__":
    main()

