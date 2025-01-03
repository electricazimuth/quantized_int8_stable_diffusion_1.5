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

#unet_onnx_model = "torchexport/unet_fixed_batch_2_torch.onnx"
unet_onnx_model = "fixedsize/unet_sim/model.onnx"
unet_onnx_modelout = "fixedsize/unet_preproc_sim/model.onnx"

#check out dir
directory = os.path.dirname(unet_onnx_modelout)
if directory and not os.path.exists(directory):
    os.makedirs(directory)

# Preprocess (Optional)
preprocess_model(unet_onnx_model, unet_onnx_modelout) # <-- doesnt work due to the lack of onnx version version 
print(f"Pre processed readty for quantisation {unet_onnx_modelout}")
exit()
# Quantize
