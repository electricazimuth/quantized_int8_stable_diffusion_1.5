import torch
import onnxruntime as ort
import numpy as np
import sys
import os
# check ls /usr/local/cuda/include/ for cudnn
# sudo apt install cudnn9-cuda-12-4=9.1.1.17-1 libcudnn9-dev-cuda-12=9.1.1.17-1 libcudnn9-static-cuda-12=9.1.1.17-1 libcudnn9-cuda-12=9.1.1.17-1
def check_cuda_availability():
    """Checks if CUDA is available and configured correctly for ONNX Runtime."""

    print("=" * 30)
    print("ONNX Runtime Debug Script")
    print("=" * 30)

    # 1. Check ONNX Runtime Build Information
    print("\n1. ONNX Runtime Build Information:")
    print(f"   - Version: {ort.__version__}")
    print(f"   - Build Configuration: {ort.get_build_info()}")

    # 2. Check Available Execution Providers
    print("\n2. Available Execution Providers:")
    available_providers = ort.get_available_providers()
    for provider in available_providers:
        print(f"   - {provider}")

    # 3. Check CUDA Execution Provider
    print("\n3. CUDA Execution Provider Check:")
    if 'CUDAExecutionProvider' in available_providers:
        print("   - CUDAExecutionProvider is AVAILABLE.")
        try:
            # Try creating a session with CUDA
            sess_options = ort.SessionOptions()
            sess_options.enable_profiling = True  # Enable profiling for more detailed error messages
            session = ort.InferenceSession("dummy_model.onnx", sess_options, providers=['CUDAExecutionProvider'])
            print("   - Successfully created an ONNX Runtime session with CUDAExecutionProvider.")

            # Try a simple inference (if you have a model)
            # Replace with your actual model input and output names if you have a model
            try:
                input_name = session.get_inputs()[0].name
                output_name = session.get_outputs()[0].name
                input_data = np.random.randn(1, 3, 224, 224).astype(np.float32) # Example input
                output = session.run([output_name], {input_name: input_data})
                print("   - Successfully ran inference with CUDAExecutionProvider.")
            except Exception as e:
                print(f"   - Inference failed: {e}")
                print("     - Check your model and input data.")

        except Exception as e:
            print(f"   - FAILED to create session with CUDAExecutionProvider: {e}")
            print_cuda_troubleshooting_tips()

    else:
        print("   - CUDAExecutionProvider is NOT AVAILABLE.")
        print_cuda_troubleshooting_tips()

    # 4. Check Environment Variables (if CUDA is not available or session creation failed)
    if 'CUDAExecutionProvider' not in available_providers or 'session' not in locals():
        print("\n4. Environment Variable Check (CUDA related):")
        cuda_env_vars = ["CUDA_PATH", "LD_LIBRARY_PATH", "PATH"]
        for env_var in cuda_env_vars:
            value = os.environ.get(env_var)
            print(f"   - {env_var}: {value}")
        print("   - Ensure these environment variables are correctly set to point to your CUDA installation.")

    # 5. Check CUDA and cuDNN versions (if applicable)
    print("\n5. CUDA and cuDNN Version Check (if installed):")
    try:
        
        print(f"   - PyTorch CUDA available: {torch.cuda.is_available()}")
        print(f"   - PyTorch CUDA version: {torch.version.cuda}")
        print(f"   - PyTorch cuDNN version: {torch.backends.cudnn.version()}")
    except ImportError:
        print("   - PyTorch not installed. Skipping PyTorch CUDA/cuDNN check.")

    try:
        import subprocess
        # Check with nvcc (if available)
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   - CUDA version (nvcc): {result.stdout.strip()}")
        else:
            print("   - Could not determine CUDA version using nvcc.")
    except FileNotFoundError:
        print("   - nvcc not found. Skipping CUDA version check with nvcc.")

def print_cuda_troubleshooting_tips():
    """Prints troubleshooting tips for CUDA related issues."""

    print("\nTroubleshooting Tips:")
    print("   1. Verify CUDA Installation:")
    print("      - Ensure you have a compatible CUDA driver and toolkit installed.")
    print("      - Check NVIDIA driver version using `nvidia-smi`.")
    print("   2. Check cuDNN Installation (if applicable):")
    print("      - Make sure cuDNN is installed and compatible with your CUDA version.")
    print("      - cuDNN libraries should be in a directory included in your LD_LIBRARY_PATH.")
    print("   3. Environment Variables:")
    print("      - Set `CUDA_PATH` to your CUDA installation directory (e.g., /usr/local/cuda).")
    print("      - Add CUDA's `bin` directory to your `PATH` (e.g., export PATH=$CUDA_PATH/bin:$PATH).")
    print("      - Add CUDA's `lib64` directory to your `LD_LIBRARY_PATH` (e.g., export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH).")
    print("   4. ONNX Runtime Build:")
    print("      - Ensure you have installed the GPU version of ONNX Runtime (e.g., `pip install onnxruntime-gpu`).")
    print("   5. Compatibility:")
    print("      - Verify that your ONNX Runtime, CUDA, cuDNN, and NVIDIA driver versions are compatible.")
    print("      - Refer to the ONNX Runtime documentation for compatibility matrices.")
    print("   6. Restart:")
    print("      - After making changes to environment variables or installations, restart your system or your Python kernel.")
    print("   7. Check for conflicting installations:")
    print("      - If you have multiple CUDA installations, make sure the correct one is being used.")
    print("   8. Permissions:")
    print("      - Ensure you have the necessary permissions to access the CUDA libraries.")
    print("   9. Device Query (Advanced):")
    print("      - Use `deviceQuery` (part of the CUDA samples) to check if your system can detect and communicate with your GPU.")
    print("   10. Check ONNX Runtime logs:")
    print("      - Look for more detailed error messages in the ONNX Runtime logs (if enabled).")

# Create a dummy ONNX model for testing (if you don't have one)
def create_dummy_model():
    """Creates a simple ONNX model for testing purposes."""
    import onnx
    from onnx import helper, TensorProto

    # Define a simple model (e.g., Add operation)
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [None, None])
    Z = helper.make_tensor_value_info('Z', TensorProto.FLOAT, [None, None])

    node_def = helper.make_node(
        'Add',
        inputs=['X', 'Y'],
        outputs=['Z'],
    )

    graph_def = helper.make_graph(
        [node_def],
        'test-model',
        [X, Y],
        [Z],
    )

    model_def = helper.make_model(graph_def, producer_name='onnx-example')
    onnx.checker.check_model(model_def)  # Check model validity
    onnx.save(model_def, "dummy_model.onnx")

if __name__ == "__main__":
    if not os.path.exists("dummy_model.onnx"):
        create_dummy_model()
    check_cuda_availability()