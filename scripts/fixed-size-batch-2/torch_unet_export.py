import torch
from diffusers import StableDiffusionPipeline

# -- trying to replicate the optimum setup
# optimum-cli export onnx \
# --model stable-diffusion-v1-5/stable-diffusion-v1-5 --task text-to-image --opset 19 --device cpu --optimize O2 --no-post-process \
# --batch_size 2 --sequence_length 77 --width 64 --height 64 --num_channels 4 \
# fixedsize

# Load the Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float32)
unet = pipe.unet

# Example dummy inputs with a fixed batch size of 2
batch_size = 2
latent_shape = (batch_size, 4, 64, 64)  # Example latent shape (adjust if needed)
latent = torch.randn(latent_shape)
timestep = torch.tensor([1000, 1000], dtype=torch.long)  # Shape: (batch_size,)
encoder_hidden_states = torch.randn((batch_size, 77, 768)) # adjust if needed

# Export to ONNX with fixed input shapes
torch.onnx.export(
    unet,
    args=(latent, timestep, encoder_hidden_states),
    f="torchexport/unet_fixed_batch_2_torch.onnx",
    input_names=["sample", "timestep", "encoder_hidden_states"],
    output_names=["out_sample"],
    opset_version=17,  # Use a supported opset version
    save_as_external_data=True
)
# Required inputs (['latent']) are missing from input feed (['sample', 'encoder_hidden_states', 'timestep']).
# INPUTS
#  - torch.Size([2, 4, 64, 64])
#  - torch.Size([2])
#  - torch.Size([2, 77, 768])