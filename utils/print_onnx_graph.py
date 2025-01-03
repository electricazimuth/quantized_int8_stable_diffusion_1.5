import onnx
import os

def save_model_details(model_file, output_file):
    """Loads an ONNX model and saves its details to separate files.

    Args:
        model_file: Path to the ONNX model file.
        output_file: Directory to save the output files.
    """


    # Extract model name for file naming
    model_name = os.path.splitext(os.path.basename(model_file))[0]

    # Load the model
    model = onnx.load(model_file)
    # Save the printable graph to a file
    with open(output_file, "w") as f:
        f.write(onnx.helper.printable_graph(model.graph))

# Define model files and output directories
model_data = [
#    {"model_file": "fixedsize/unet/model.onnx", "filename": "graph_original_unet.txt"},
#    {"model_file": "fixedsize/unet_32/model.onnx", "filename": "graph_fixed_size_unet.txt"},
#   {"model_file": "torchpreproc/unet_preprocessed.onnx", "filename": "graph_unet_torch_prep.txt"},
#    {"model_file": "torchexport/unet_fixed_batch_2_torch.onnx", "filename": "graph_unet_torch_expo.txt"},
#    {"model_file": "fixedsize/unet_QFQDQ_PC1_CMMMX_ATU8_WTS8/model.onnx", "filename": "graph_unet_QFQDQ_PC1_CMMMX_ATU8_WTS8.txt"},
#    {"model_file": "fixedsize/unet_QFQDQ_PC1_CMMMX_ATU8_WTS8/model.onnx", "filename": "graph_unet_QFQDQ_PC1_CMMMX_ATU8_WTS8.txt"},
    {"model_file": "fixedsize/unet_dynamic_uint/model.onnx", "filename": "graph_unet_dynamic_uint.txt"},
    {"model_file": "fixedsize/unet_dynamic_uint_sim/model.onnx", "filename": "graph_unet_dynamic_uint_sim.txt"},    
#    {"model_file": "sd_q8_2/unet.onnx", "filename": "sd_q8_2/unet.onnxbackbone.txt"},
#   {"model_file": 'fixedsize/unet_QFQDQ_PC1_CMMMX_ATU8_WTS8/model.onnx', "filename": "graph_fixed_size_static_quant_unet_ig.txt"},
]

# Process each model
for data in model_data:
    print(f"**** START processing {data['model_file']} ****")
    save_model_details(data["model_file"], data["filename"])
    print(f"**** END processing {data['model_file']} ****")
