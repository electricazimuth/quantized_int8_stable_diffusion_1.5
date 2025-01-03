import onnx
from onnxruntime.quantization import shape_inference, QuantFormat, QuantType, quantize_static, CalibrationDataReader, CalibrationMethod

import numpy as np
import glob
from PIL import Image
import yaml
import torch
import gc
import time
import os
import re
import onnxruntime as ort
from pathlib import Path
from tqdm import tqdm
from build_calibration_dataset import print_time, UNetCalibrationDataReader

nowtime = time.time()

# CalibrationDataReader - UNetCalibrationDataReader imported

def get_quant_format_str(quant_format):
  return "QO" if quant_format == QuantFormat.QOperator else "QDQ"

def get_calibration_method_str(calibrate_method):
  if calibrate_method == CalibrationMethod.Percentile:
    return "PCT"
  elif calibrate_method == CalibrationMethod.MinMax:
    return "MMX"
  elif calibrate_method == CalibrationMethod.Entropy:
    return "ENT"
  else:
    return "UNK"  # Handle unknown methods

def get_quant_type_str(quant_type):
  return "U8" if quant_type == QuantType.QUInt8 else "S8"

def preprocess_model(input_path: str, output_path: str):
    """
    Preprocesses the model for quantization.

    Args:
        input_path: Path to the input ONNX model.
        output_path: Path to save the preprocessed model.
    """
    print(f"Preprocessing model from {input_path} to {output_path}...")
    shape_inference.quant_pre_process(input_path, output_path, skip_symbolic_shape=False, save_as_external_data=True)
    print("Model preprocessing complete.")

def filter_stable_diffusion_nodes(node_name):
    """
    Filters nodes in a Stable Diffusion 1.5 model based on their names.

    This function checks if a node name contains specific keywords related to
    time embeddings, convolutional layers, context embedding, or output
    normalization. It's likely used for model optimization, analysis,
    fine-tuning, or applying custom operations to specific parts of the model.

    Args:
        node_name: The name of the node to check.

    Returns:
        True if the node name matches the filter criteria, False otherwise.
    """

    # Define the keywords to look for in node names. These keywords represent
    # Originally from here: https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/diffusers/quantization/utils.py#L41 - used to Disable quantizer by filter function
    # different components or operations within the Stable Diffusion model.
    keywords = [
        "time_emb_proj",  # Time embedding projection
        "time_embedding",  # Time embedding
        "conv_in",  # Input convolutional layer
        "conv_out",  # Output convolutional layer
        "conv_shortcut",  # Convolutional layer in a skip connection
        "add_embedding",  # Adding embeddings
        "pos_embed",  # Positional embedding
        "time_text_embed",  # Combined time and text embedding
        "context_embedder",  # Context embedder
        "norm_out",  # Output normalization layer
    ]

    # Create a regular expression pattern to efficiently search for the keywords.
    # The pattern matches any string that contains any of the keywords,
    # surrounded by any characters before or after.
    # Example: "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.time_emb_proj" would match
    pattern = re.compile(
        r".*(" + "|".join(keywords) + r").*"
    )

    # Check if the node name matches the pattern.
    match = pattern.match(node_name)

    # Return True if a match is found (node name contains a keyword), False otherwise.
    return match is not None
# --- Main Quantization Script ---

model_path = "fixedsize"
unet_onnx_model_file = "fixedsize/unet_preproc_sim/model.onnx"#fixedsize/unet_preproc/model.onnx"

#unet_onnx_model_file = "torchexport/unet_fixed_batch_2_torch.onnx"



if torch.cuda.is_available():
    print("Cuda available")
else:
    print("NO Cuda available")

unet_onnx_model = unet_onnx_model_file # onnx.load(unet_onnx_model_file, load_external_data=True)
do_static = False


prompts_file = 'prompts.yaml'
num_samples = 1

# Preprocess (Optional)
#preprocess_model(unet_onnx_model, "fixedsize/unet_preproc/model.onnx") # <-- doesnt work due to the lack of onnx version version 
#exit()
# Quantize

"""
Accuracy Evaluation: Compare the accuracy of the quantized model against the original FP32 model using a validation dataset.
Debugging: If accuracy drops significantly, use the debugging tools mentioned in the documentation to identify problematic nodes/tensors. You might need to exclude certain nodes from quantization or adjust calibration parameters.
Per-Channel vs. Per-Tensor: Experiment with per_channel=True or per_channel=False to see which yields better results.
Reduce Range: Try reduce_range=True if you're targeting CPUs without VNNI support.
Calibration Method: If MinMax doesn't give good results, try CalibrationMethod.Entropy or CalibrationMethod.Percentile

QuantFormat.QOperator format quantizes the model with quantized operators directly.
QuantFormat.QDQ format quantize the model by inserting QuantizeLinear/DeQuantizeLinear on the tensor.

It is recommended to use QuantFormat.QDQ format from 1.11 with activation_type = QuantType.QInt8 and weight_type = QuantType.QInt8. 
If model is targeted to GPU/TRT, symmetric activation and weight are required. 
If model is targeted to CPU, asymmetric activation and symmetric weight are recommended for balance of performance and accuracy
Symmetric = (int8), Asymmetric = (uint8)
"""


print_time("Starting up")

dataset_folder = "calibration_dataset"
# Create Calibration Data Reader
print(f"Creating Calibration Data Reader from {dataset_folder}...")

data_reader = UNetCalibrationDataReader(dataset_folder)

print_time("Starting quantization process...")
## note: SmoothQuant requires neural-compressor [ ]
extra_options = {"SmoothQuant": True, "SmoothQuantAlpha": 1.0 }
#extra_options = {}
# ONNXRuntime quantization doesn't support data format:"
# "activation_type=QuantType.QInt8 AND weight_type=QuantType.QUInt8
# one of QuantFormat.QOperator or QuantFormat.QDQ
quant_format = QuantFormat.QDQ
per_channel = True #True
# CalibrationMethod.Percentile or CalibrationMethod.MinMax or CalibrationMethod.Entropy 
calibrate_method = CalibrationMethod.MinMax 
if(calibrate_method == CalibrationMethod.Percentile):
   # percentile: Control quantization scaling factors (amax) collecting range, meaning that we will collect the chosen amax in the range of (n_steps * percentile) steps. Recommendation: 1.0
   # Recommendation SmoothQuantAlpha : 0.8 for SDXL, 1.0 for SD 1.5
   extra_options["CalibPercentile"] = 1.0
   

# ONNXRuntime quantization doesn't support this mixed data format: activation_type=QuantType.QInt8 AND weight_type=QuantType.QUInt8
activation_type=QuantType.QInt8
weight_type=QuantType.QInt8


unet_folder = f"unet_QF{get_quant_format_str(quant_format)}_PC{int(per_channel)}_CM{get_calibration_method_str(calibrate_method)}_AT{get_quant_type_str(activation_type)}_WT{get_quant_type_str(weight_type)}"
model_output_file = model_path + "/" + unet_folder+ "/model.onnx"
directory = os.path.dirname(model_output_file)
if directory and not os.path.exists(directory):
    os.makedirs(directory)


# Load your ONNX model
onnx_model = onnx.load(unet_onnx_model)

# Build the list of nodes to exclude
nodes_to_exclude = []
for node in onnx_model.graph.node:
    if filter_stable_diffusion_nodes(node.name):
        nodes_to_exclude.append(node.name)

print("Nodes to exclude:", nodes_to_exclude)

# Unload the model
del onnx_model

# Force garbage collection
gc.collect()


quantize_static(
    model_input=unet_onnx_model,#"unet_preprocessed.onnx",
    model_output=model_output_file,
    calibration_data_reader=data_reader,
    quant_format=quant_format, #QOperator, 
    op_types_to_quantize=None,
    per_channel=per_channel, ## -- try this True..Per-channel quantization can sometimes lead to better accuracy and potentially smaller model size compared to per-tensor quantization.
    reduce_range=False,
    activation_type=activation_type, #first runs QUInt8
    weight_type=weight_type, #first runs QInt8
    nodes_to_quantize=None,
    nodes_to_exclude=nodes_to_exclude,
    use_external_data_format=True, #option used for large size (>2GB) model. Set to False by default.
    calibrate_method=calibrate_method, #Percentile #MinMax # MinMax / Entropy / Percentile
    extra_options=extra_options
)

print_time(f"Finished quantization {model_output_file}")