# Quantize Stable Diffusion to int8 (ONNX)
These scripts will build calibration dataset and run it on the unet to generate an int8 model.
This uses scripts from the [onnxruntime](https://github.com/microsoft/onnxruntime) project.

First create a fixed size UNET using optimum
```
optimum-cli export onnx \
--model stable-diffusion-v1-5/stable-diffusion-v1-5 --task text-to-image --opset 14 --device cpu --optimize O2 --no-post-process \
--batch_size 1 --sequence_length 77 --width 64 --height 64 --num_channels 4 \
fixedsize
```
## Fixed size model
Stable Diffusion is normally a dynamically size model. Here we're fixing the size, the width and height are 1/8 of the pixel size output, eg 512x512 => 64x64. The final usuage is meant for very under powered devices. Having a batch size of 1 means that during inference we need to run the unet twice, once for the unconditional text_embeddings inputs and once for the conditional text_embeddings inputs. It may be that for certain devices a fixed batch size of 2 is more appropriate.   
*[Thought: it might also quantize better with bacth size 2?]*
```
batch_size 1  
sequence_length 77  
width 64  
height 64  
```

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
