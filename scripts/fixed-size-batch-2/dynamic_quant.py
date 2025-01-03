import onnx
from onnxruntime.quantization import shape_inference, QuantFormat, QuantType, quantize_dynamic, CalibrationDataReader, CalibrationMethod

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
        "conv",
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



if torch.cuda.is_available():
    print("Cuda available")
else:
    print("NO Cuda available")


# Preprocess (Optional)
#preprocess_model(unet_onnx_model, "fixedsize/unet_preproc/model.onnx") # <-- doesnt work due to the lack of onnx version version 
#exit()
# Quantize


    """Given an onnx model, create a quantized onnx model and save it into a file

def quantize_dynamic(
    model_input: str | Path | onnx.ModelProto,
    model_output: str | Path,
    op_types_to_quantize=None,
    per_channel=False,
    reduce_range=False,
    weight_type=QuantType.QInt8,
    nodes_to_quantize=None,
    nodes_to_exclude=None,
    use_external_data_format=False,
    extra_options=None,
):

    Args:
        model_input: file path of model or ModelProto to quantize
        model_output: file path of quantized model
        op_types_to_quantize:
            specify the types of operators to quantize, like ['Conv'] to quantize Conv only.
            It quantizes all supported operators by default.
        per_channel: quantize weights per channel
        reduce_range:
            quantize weights with 7-bits. It may improve the accuracy for some models running on non-VNNI machine,
            especially for per-channel mode
        weight_type:
            quantization data type of weight. Please refer to
            https://onnxruntime.ai/docs/performance/quantization.html for more details on data type selection
        nodes_to_quantize:
            List of nodes names to quantize. When this list is not None only the nodes in this list
            are quantized.
            example:
            [
                'Conv__224',
                'Conv__252'
            ]
        nodes_to_exclude:
            List of nodes names to exclude. The nodes in this list will be excluded from quantization
            when it is not None.
        use_external_data_format: option used for large size (>2GB) model. Set to False by default.
        extra_options:
            key value pair dictionary for various options in different case. Current used:
                extra.Sigmoid.nnapi = True/False  (Default is False)
                ActivationSymmetric = True/False: symmetrize calibration data for activations (default is False).
                WeightSymmetric = True/False: symmetrize calibration data for weights (default is True).
                EnableSubgraph = True/False :
                    Default is False. If enabled, subgraph will be quantized. Dynamic mode currently is supported. Will
                    support more in the future.
                ForceQuantizeNoInputCheck = True/False :
                    By default, some latent operators like maxpool, transpose, do not quantize if their input is not
                    quantized already. Setting to True to force such operator always quantize input and so generate
                    quantized output. Also the True behavior could be disabled per node using the nodes_to_exclude.
                MatMulConstBOnly = True/False:
                    Default is True for dynamic mode. If enabled, only MatMul with const B will be quantized.
    """


print_time("Starting up dyncamic")


model_path = "fixedsize"
unet_onnx_model = "fixedsize/unet_preproc_sim/model.onnx"#fixedsize/unet_preproc/model.onnx"

unet_folder_out = f"unet_dynamic_uint"
model_output_file = model_path + "/" + unet_folder_out+ "/model.onnx"

directory = os.path.dirname(model_output_file)
if directory and not os.path.exists(directory):
    os.makedirs(directory)

nodes_to_exclude = None
weight_type = QuantType.QUInt8
if( weight_type == QuantType.QInt8):
    onnx_model = onnx.load(unet_onnx_model)
    nodes_to_exclude = []
    for node in onnx_model.graph.node:
        if filter_stable_diffusion_nodes(node.name):
            nodes_to_exclude.append(node.name)

print("Nodes to exclude:", nodes_to_exclude)
   

# Load your ONNX model
##onnx_model = onnx.load(unet_onnx_model)

# Build the list of nodes to exclude
##nodes_to_exclude = []
##for node in onnx_model.graph.node:
##    if filter_stable_diffusion_nodes(node.name):
##        nodes_to_exclude.append(node.name)



# Unload the model
##del onnx_model

# Force garbage collection
gc.collect()
##op_types_to_quantize = ["Attention", "Conv", "MatMul", "Add", "Mul", "Gemm"]

quantize_dynamic(
    model_input=unet_onnx_model,#"unet_preprocessed.onnx",
    model_output=model_output_file,
    op_types_to_quantize=None,
    per_channel=False,
    reduce_range=False,
    weight_type=weight_type,
    nodes_to_quantize=None,
    nodes_to_exclude=nodes_to_exclude,
    use_external_data_format=True,
    extra_options=None
)


# Quantize the model
####         quantize_dynamic(
####             model_input=out_path,
####             model_output=quantized_out_path,
####             weight_type=QuantType.QUInt8,
####             optimize_model=True,
####             per_channel=True,
####             reduce_range=True,
####             extra_options={'WeightSymmetric': False}
####         )
#### 
#### unsigned = True
####     if(unsigned):
####         wtype = QuantType.QUInt8
####         prefix = "dyn_u_" # runs on cpu
####     else:
####         wtype = QuantType.QInt8
####         prefix = "dyn_s_" # NOT_IMPLEMENTED :  ConvInteger(10) '/conv_in/Conv_quant'
#### 
####     #model_output_file = prefix + model_output_file
#### 
####     directory = os.path.dirname(model_output_file)
####     if directory and not os.path.exists(directory):
####         os.makedirs(directory)
#### 
#### 
####     quantize_dynamic(
####         model_input=unet_onnx_model,
####         model_output=model_output_file,
####         op_types_to_quantize=None,
####         per_channel=False,
####         reduce_range=False,
####         weight_type=wtype,
####         nodes_to_quantize=None,
####         nodes_to_exclude=None,
####         use_external_data_format=True,
####         extra_options=None,
####     )
#### 
print_time(f"Finished quantization {model_output_file}")