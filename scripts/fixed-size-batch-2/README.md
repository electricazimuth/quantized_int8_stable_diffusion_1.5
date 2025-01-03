# dependancies - CUDnn was not installed

sudo apt install cudnn9-cuda-12-4=9.1.1.17-1 libcudnn9-dev-cuda-12=9.1.1.17-1 libcudnn9-static-cuda-12=9.1.1.17-1 libcudnn9-cuda-12=9.1.1.17-1
sudo apt install cuda-toolkit-12-4

https://github.com/onnx/onnx/blob/main/requirements.txt
onnx                     1.17.0
onnxruntime              1.20.1
onnxruntime-gpu          1.20.1
optimum                  1.23.3
**protobuf                 3.20.2**

# Quantize Stable Diffusion to int8 (ONNX)
These scripts will build calibration dataset and run it on the unet to generate an int8 model.
This uses scripts from the [onnxruntime](https://github.com/microsoft/onnxruntime) project.

First create the onnx files using optimum (so we have vae & text encoder)
```
optimum-cli export onnx \
--model stable-diffusion-v1-5/stable-diffusion-v1-5 --task text-to-image --opset 17 --device cpu --optimize O1 --no-post-process \
--batch_size 2 --sequence_length 77 --width 64 --height 64 --num_channels 4 \
fixedsize
```
## Fixed size model
Stable Diffusion is normally a dynamically size model. Here we're fixing the size, the width and height are 1/8 of the pixel size output, eg 512x512 => 64x64. The final usuage is meant for very under powered devices. Batch size 2 so we can pass both the conditional and unconditional embeddings simultaneously and make use of classifier-free Guidance (CFG) like the original implementation  

```
batch_size 2   
sequence_length 77  
width 64  
height 64  
```
# Custom fixed sizes
Create fully fixed size using torch onnx exporter
```
python scripts/fixed-size-batch-2/torch_unet_export.py
```
Pre process this model

```
python scripts/fixed-size-batch-2/preprocess_unet.py
```
# Check it works
```
python scripts/fixed-size-batch-2/inference.py
```
# Generate the dataset
```
python scripts/fixed-size-batch-2/build_calibration_dataset.py
```


## Try onnxsim

onnxsim --save-as-external-data torchexport/unet_fixed_batch_2_torch.onnx fixedsize/unet_sim/model.onnx

python scripts/fixed-size-batch-2/preprocess_unet.py

## Try dynamic
python scripts/fixed-size-batch-2/dynamic_quant.py
mkdir fixedsize/unet_dynamic_uint_sim
onnxsim --save-as-external-data fixedsize/unet_dynamic_uint/model.onnx fixedsize/unet_dynamic_uint_sim/model.onnx


RKNN

ValueError: The DynamicQuantizeLinear('encoder_hidden_states_QuantizeLinear') will cause the graph to be a dynamic graph! Remove it manually and try again!






**Build the calibration dataset** the script also outputs the images for verification that the pipeline is correct.
The script is gathering the data from steps [0, 2, 4, 6, 10] as they are the ones that change the most, eg using the most variance in the input data to help calibration.
```
python build_calibration_dataset.py
```

**Run quantisation** In the script update the configuration and model path for you preferences and run the script to quantize the model

```
python build_static_quant_from_dataset.py
```

**Inference** The inference script can take in a set of different models for comparison, mainlymeant for testing differnt UNET models but you can add in difference text encioders or vae decoders if you have them. Update the model paths and run the script and it will generate a random image form the prompts file.
```
python inference.py
```
___

# Broken models - 23.12.2024
The resulting models seem to be broken. I'm investigating other approaches.


Generating the fixed size ONNX models I can't find any way to set the IR version. It's always 0. This means ONNXSIM can't run on the models :(
I beleive this is due to some issue in the framework (onnxruntime or optimum) which is doing the processing
```
 -- Warning: onnx version converter is unable to parse input model. (The IR version of the ONNX model may be too old.) 
 .ValidationError: The model does not have an ir_version set properly.
```


# Preprocessing and optimising

**onnxruntime.quantization.preprocess**  
the preprocessor for quantization fails with:  
`[ONNXRuntimeError] : 7 : INVALID_PROTOBUF : Load model from /tmp/pre.quant.uwwo0rkn/symbolic_shape_inferred.onnx failed:Protobuf parsing failed`  
Maybe you'll have a better setup, you can try using this command:
```
python -m onnxruntime.quantization.preprocess --input fixedsize/unet_torch32/export.onnx --output fixedsize/unet_prepp/model.onnx \
--save_as_external_data --all_tensors_to_one_file --external_data_location fixedsize/unet_prepp/data
```
**onnxruntime.transformers.optimizer**  
You can also try the onnxruntime optimizer, but for my downstream tasks these optimized models break my pipelines.
```
python -m onnxruntime.transformers.optimizer \
    --input fixedsize/unet_torch32/export.onnx \
    --output fixedsize/unet_03/model.onnx \
    --model_type unet \
    --use_external_data_format \
    --opt_level 3
```