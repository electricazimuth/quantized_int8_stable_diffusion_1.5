import yaml
import torch
import numpy as np
import os
import onnxruntime as ort
from pathlib import Path
from transformers import CLIPTokenizer
from tqdm import tqdm
from PIL import Image
from diffusers.schedulers import DDIMScheduler
import random

class ImageGenerator:
    def __init__(self, text_encoders, unets, vae_decoders, prompt="A painting of a boat at sea"):
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        
        # Initialize scheduler
        self.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
            set_alpha_to_one=False
        )
        
        # Create ONNX sessions
        if torch.cuda.is_available():
            cuda_options = {'device_id': 0}
            providers = [('CUDAExecutionProvider', cuda_options), 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        print(f"Providers: {providers}")

        self.text_encoders = {te['name']: ort.InferenceSession(te['path'], providers=providers) for te in text_encoders}
        self.unets = {unet['name']: ort.InferenceSession(unet['path'], providers=providers) for unet in unets}
        self.vae_decoders = {vae['name']: ort.InferenceSession(vae['path'], providers=providers) for vae in vae_decoders}
        
        self.prompt = prompt

        # Create output directory
        Path('generated_images').mkdir(exist_ok=True)
        # lock the seed
        np.random.seed(42)
        random.seed(42)

        # Generate and store the initial random latents (fixed size 1x4x64x64)
        self.initial_latents = np.random.randn(1, 4, 64, 64).astype(np.float32)

    def tokenize_prompt(self, prompt):
        """Tokenize a prompt"""
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,  # Fixed sequence length
            truncation=True,
            return_tensors="np",
        )
        input_ids = text_inputs.input_ids.astype(np.int32)
        return input_ids

    def denoise_latents(self, latents, text_embeddings, unet_name, num_inference_steps=15):
        """Perform multiple denoising steps with fixed size inputs"""
        self.scheduler.set_timesteps(num_inference_steps)
        latents = latents * self.scheduler.init_noise_sigma
        
        for i, t in enumerate(self.scheduler.timesteps):
            # Scale the latents according to the timestep
            scaled_latents = self.scheduler.scale_model_input(torch.from_numpy(latents), t).numpy()
            
            # Process unconditional and conditional latents separately due to fixed batch size
            latent_model_input_uncond = scaled_latents.copy()
            latent_model_input_text = scaled_latents.copy()
            
            # Create timestep tensor with shape [1]
            timestep_data = np.array([t]).astype(np.int64)
            
            # Run UNet for unconditional input
            noise_pred_uncond = self.unets[unet_name].run(
                None,
                {
                    "sample": latent_model_input_uncond,
                    "encoder_hidden_states": text_embeddings[:1],  # Only unconditional
                    "timestep": timestep_data
                }
            )[0]
            
            # Run UNet for conditional input
            noise_pred_text = self.unets[unet_name].run(
                None,
                {
                    "sample": latent_model_input_text,
                    "encoder_hidden_states": text_embeddings[1:],  # Only conditional
                    "timestep": timestep_data
                }
            )[0]
            
            # Combine predictions
            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)


            latents = self.scheduler.step(
                torch.from_numpy(noise_pred),
                t,
                torch.from_numpy(latents)
            ).prev_sample.numpy()
        
        return latents

    def generate_image(self, prompt, te_name, unet_name, vae_name):
        """Generate an image using specified models"""
        
        # Tokenize prompt and get text embeddings
        input_ids = self.tokenize_prompt(prompt)
        text_encoder_output = self.text_encoders[te_name].run(
            None,
            {"input_ids": input_ids}
        )[0]
        
        # Create uncond embeddings
        uncond_input = self.tokenize_prompt("")
        uncond_embeddings = self.text_encoders[te_name].run(
            None,
            {"input_ids": uncond_input}
        )[0]
        
        # Combine embeddings (shape will be 2x77x768)
        text_embeddings = np.concatenate([uncond_embeddings, text_encoder_output])
        
        # Use the stored initial latents
        latents = self.initial_latents.copy()
        
        # Perform denoising steps
        final_latents = self.denoise_latents(latents, text_embeddings, unet_name)
        
        # Decode latents using VAE
        scaled_latents = 1 / 0.18215 * final_latents
        decoded = self.vae_decoders[vae_name].run(
            None,
            {"latent_sample": scaled_latents}
        )[0]

        image = self.convert_to_image(decoded)
        
        # Save image
        shortened_prompt = prompt.replace(" ", "")
        if len(shortened_prompt) > 15:
            shortened_prompt = shortened_prompt[:15]

        image_name = f"generated_images/sd_{te_name}_unet_{unet_name}_vae_{vae_name}_{shortened_prompt}.png"
        image.save(image_name)

        return image_name, prompt

    @staticmethod
    def convert_to_image(decoded):
        """Convert VAE decoder output to PIL Image"""
        decoded = decoded.squeeze(0)
        decoded = ((decoded + 1) * 127.5).clip(0, 255).astype(np.uint8)
        decoded = decoded.transpose(1, 2, 0)  # CHW -> HWC
        return Image.fromarray(decoded)

    def generate_images_with_combinations(self):
        """Generate images using all combinations of models"""
        
        for te_name in self.text_encoders.keys():
            for unet_name in self.unets.keys():
                for vae_name in self.vae_decoders.keys():
                    image_name, prompt = self.generate_image(self.prompt, te_name, unet_name, vae_name)
                    print(f"Generated image: {image_name} with prompt: '{prompt}'")
                print("clear unet")

def main():
    text_encoders = [
        {'name': 'teO1', 'path': 'fixedsize/text_encoder/model.onnx'},
    ]
    unets = [
#        {'name': 'unet32', 'path': 'fixedsize/unet_torch32/model.onnx'},
#        {'name': 'unet', 'path': 'fixedsize/unet/model.onnx'},
        {'name': 'unets81', 'path': 'fixedsize/unet_QFQDQ_PC1_CMMMX_ATU8_WTS8/model.onnx'},
    ]
    vae_decoders = [
        {'name': 'vae01', 'path': 'fixedsize/vae_decoder/model.onnx'},
    ]

    # Load prompts
    prompts_file = 'prompts.yaml'
    with open(prompts_file, 'r') as f:
        prompts = yaml.safe_load(f)
        prompt = random.choice(list(prompts.values()))

    print(f"Generating using: {prompt}")
    
    generator = ImageGenerator(text_encoders, unets, vae_decoders, prompt)
    generator.generate_images_with_combinations()

if __name__ == "__main__":
    main()
