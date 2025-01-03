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
import json
import time

class ImageGenerator:
    def __init__(self, text_encoders, unets, vae_decoders, prompt="A painting of a boat at sea", capture_outputs=False):
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.capture_outputs = capture_outputs
        self.output_dir = Path('model_outputs')
        self.nowtime = time.time()
        
        if self.capture_outputs:
            self.output_dir.mkdir(exist_ok=True)
        
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

        # Generate and store the initial latents (fixed size 1x4x64x64)
        self.initial_latents = np.random.randn(1, 4, 64, 64).astype(np.float32)
    def print_time(self, message):

        elapsed_time = time.time() - self.nowtime
        print(f"{elapsed_time:.2f} {message}")
        self.nowtime = time.time()

    def get_tensor_info(self, array):
        """Get detailed information about a numpy array"""
        return {
            'shape': list(array.shape),
            'size': array.size,
            'bytes': array.nbytes,
            'dtype': str(array.dtype),
            'ndim': array.ndim,
            'is_contiguous': array.flags.c_contiguous,
            'memory_layout': 'C' if array.flags.c_contiguous else ('F' if array.flags.f_contiguous else 'non-contiguous')
        }

    def save_model_outputs(self, timestep, unet_name, noise_pred, latents, scaled_latents):
        """Save intermediate outputs for analysis"""
        output_path = self.output_dir / f"{unet_name}_step_{timestep}"
        output_path.mkdir(exist_ok=True)
        
        # Save the raw outputs
        np.save(output_path / "noise_pred.npy", noise_pred)
        np.save(output_path / "latents.npy", latents)
        np.save(output_path / "scaled_latents.npy", scaled_latents)
        
        # Save statistics for quick comparison
        stats = {
            'noise_pred': {
                'stats': {
                    'mean': float(np.mean(noise_pred)),
                    'std': float(np.std(noise_pred)),
                    'min': float(np.min(noise_pred)),
                    'max': float(np.max(noise_pred)),
                    'nan_count': int(np.isnan(noise_pred).sum()),
                    'inf_count': int(np.isinf(noise_pred).sum())
                },
                'tensor_info': self.get_tensor_info(noise_pred)
            },
            'latents': {
                'stats': {
                    'mean': float(np.mean(latents)),
                    'std': float(np.std(latents)),
                    'min': float(np.min(latents)),
                    'max': float(np.max(latents)),
                    'nan_count': int(np.isnan(latents).sum()),
                    'inf_count': int(np.isinf(latents).sum())
                },
                'tensor_info': self.get_tensor_info(latents)
            },
            'scaled_latents': {
                'stats': {
                    'mean': float(np.mean(scaled_latents)),
                    'std': float(np.std(scaled_latents)),
                    'min': float(np.min(scaled_latents)),
                    'max': float(np.max(scaled_latents)),
                    'nan_count': int(np.isnan(scaled_latents).sum()),
                    'inf_count': int(np.isinf(scaled_latents).sum())
                },
                'tensor_info': self.get_tensor_info(scaled_latents)
            },
            'metadata': {
                'timestep': int(timestep),
                'unet_name': unet_name,
                #'timestamp': str(pd.Timestamp.now())
            }
        }
        
        with open(output_path / "tensor_analysis.json", 'w') as f:
            json.dump(stats, f, indent=2)

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
        """Perform multiple denoising steps with batch size 2"""
        self.scheduler.set_timesteps(num_inference_steps)
        latents = latents * self.scheduler.init_noise_sigma
        
        for i, t in enumerate(self.scheduler.timesteps):
            # Scale the latents
            scaled_latents = self.scheduler.scale_model_input(torch.from_numpy(latents), t).numpy()
            
            # Create batch of size 2 by duplicating the latents
            latent_model_input = np.repeat(scaled_latents, 2, axis=0)  # Shape: [2, 4, 64, 64]
            
            # Create timestep tensor with shape [2] to match batch size
            timestep_data = np.array([t, t], dtype=np.int64)
            
            # Run UNet with batch size 2
            noise_pred = self.unets[unet_name].run(
                None,
                {
                    "sample": latent_model_input,  # Shape: [2, 4, 64, 64]
                    "encoder_hidden_states": text_embeddings,  # Shape: [2, 77, 768]
                    "timestep": timestep_data
                }
            )[0]
            
            # Split predictions and combine with guidance scale
            noise_pred_uncond, noise_pred_text = noise_pred[0:1], noise_pred[1:2]
            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

            if self.capture_outputs:
                self.save_model_outputs(t, unet_name, noise_pred, latents, scaled_latents)

            # Step with scheduler
            latents = self.scheduler.step(
                torch.from_numpy(noise_pred),
                t,
                torch.from_numpy(latents)
            ).prev_sample.numpy()
        
        return latents

    def generate_image(self, prompt, te_name, unet_name, vae_name):
        """Generate an image using specified models"""
        try:
            # Tokenize prompt and get text embeddings
            self.print_time("start generation")
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
            self.print_time("finished generation")

            return image_name, prompt
            
        except Exception as e:
            print(f"Error generating image: {str(e)}")
            raise

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
                    self.print_time(f"Generated image: {image_name} with prompt: '{prompt}'")
                print("clear unet")

def main():
    text_encoders = [
        {'name': 'teO1', 'path': 'fixedsize/text_encoder/model.onnx'},
    ]
    unets = [
#        {'name': 'unet_torch_orig', 'path': 'torchexport/unet_fixed_batch_2_torch.onnx'},
#        {'name': 'unet_torch_prep', 'path': "fixedsize/unet_preproc/model.onnx"},
#        {'name': 'unet_sim', 'path': "fixedsize/unet_sim/model.onnx"},
#        {'name': 'unet_preproc_sim', 'path': "fixedsize/unet_preproc_sim/model.onnx"},
#        {'name': 'unet_A_q8', 'path': "fixedsize/unet_QFQDQ_PC0_CMMMX_ATS8_WTS8/model.onnx"},
#        {'name': 'unet_B_q8', 'path': "fixedsize/unet_QFQO_PC0_CMMMX_ATS8_WTS8/model.onnx"}
#        {'name': 'unet_D_q8', 'path': "fixedsize/unet_QFQDQ_PC1_CMMMX_ATS8_WTS8/model.onnx"},
#        {'name': 'unet_dyn', 'path': "fixedsize/unet_dynamic_int/model.onnx"},
        {'name': 'unet_dynu', 'path': "fixedsize/unet_dynamic_uint/model.onnx"},
        {'name': 'unet_dynu_sim', 'path': "fixedsize/unet_dynamic_uint_sim/model.onnx"},
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
    capture_outputs = False
    
    generator = ImageGenerator(text_encoders, unets, vae_decoders, prompt, capture_outputs)
    generator.generate_images_with_combinations()

if __name__ == "__main__":
    main()