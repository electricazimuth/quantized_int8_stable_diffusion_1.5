# Noteworthy Examples

[Olive](https://github.com/microsoft/Olive) - This is a wrapper for the optimum project as far as I can tell. Does exactly the same. [Example script is here](https://github.com/microsoft/Olive/blob/main/examples/stable_diffusion/stable_diffusion.py) No quantisation options.

[Optimum-quanto](https://github.com/huggingface/optimum-quanto) - This add in torch specific (quanto custom) operators and can't be replicated in ONNX currently.
Quantised models when tried to be loaded in throw: `ValueError: Cannot load <class 'diffusers.models.unets.unet_2d_condition.UNet2DConditionModel'> from quanto_out/unet because the following keys are missing:....`

[ml-stable-diffusion](https://github.com/apple/ml-stable-diffusion) - A Core ML specific export and optimisation to int8

[TensorRT](https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/release/0.15.0/diffusers/quantization) The TensorRT-Model-Optimizer version <=0.15.0 has example script sfor SD 1.5 to int8, but when I run them I get float models.., some part of it isn't working, more debugging should uncover the issue.