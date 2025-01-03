import torch
from diffusers import StableDiffusionPipeline

# Load the Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float32)
unet = pipe.unet

useinfo = True #False


if(useinfo):
#Input shapes:
#  - torch.Size([2, 4, 64, 64])
#  - torch.Size([2])
#  - torch.Size([2, 77, 768])
#Output shapes:
    from torchinfo import summary
    # Load the Stable Diffusion pipeline
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32)
    unet = pipe.unet

    # Create dummy input shapes (adjust as needed)
    batch_size = 2
    input_size = [
        (batch_size, 4, 64, 64),  # latent
        (batch_size,),             # timestep
        (batch_size, 77, 768)    # encoder_hidden_states
    ]

    # Generate model summary
    summary(unet, input_size=input_size)

else:
#Input shapes:
#  - torch.Size([2, 4, 64, 64])
#  - torch.Size([2])
#  - torch.Size([2, 77, 768])
#Output shapes:
    # Hook to capture shapes
    def print_shapes(module, input, output):
        print("Input shapes:")
        for inp in input:
            print(f"  - {inp.shape}")
        print("Output shapes:")
        for outp in output:
            print(f"  - {outp.shape}")

    # Register the hook
    handle = unet.register_forward_hook(print_shapes)

    # Create dummy inputs (adjust shapes as needed)
    batch_size = 2
    latent_shape = (batch_size, 4, 64, 64)
    latent = torch.randn(latent_shape)
    timestep = torch.tensor([950] * batch_size, dtype=torch.int64)
    encoder_hidden_states = torch.randn((batch_size, 77, 768))

    # Perform a dummy forward pass
    with torch.no_grad():
        unet(latent, timestep, encoder_hidden_states)

    # Remove the hook
    handle.remove()



