import onnxruntime as ort
import numpy as np
import yaml
import random
from tqdm import tqdm
import os
from onnxruntime.quantization import CalibrationDataReader
from transformers import CLIPTokenizer
from diffusers import DDIMScheduler
import torch
from pathlib import Path
from PIL import Image
import time
import shutil

nowtime = time.time()

class UNetCalibrationDataReader(CalibrationDataReader):
    def __init__(self, calibration_dataset_path):
        self.calibration_dataset_path = calibration_dataset_path
        self.data_items = []  # Stores file paths
        self.enum_data = None
        self.current_index = 0
        self.preprocess_data()

    def preprocess_data(self):
        """
        Preprocesses the calibration dataset to create a list of file paths.
        Now handles combined conditional and unconditional data.
        """
        prompt_dirs = [d for d in os.listdir(self.calibration_dataset_path) 
            if os.path.isdir(os.path.join(self.calibration_dataset_path, d))
        ]
        for prompt_dir in prompt_dirs:
            step_dirs = [d for d in os.listdir(os.path.join(self.calibration_dataset_path, prompt_dir))
                if os.path.isdir(os.path.join(self.calibration_dataset_path, prompt_dir, d))
            ]
            for step_dir in step_dirs:
                # Add only one entry per step since we now handle both conditional 
                # and unconditional data together
                self.data_items.append((prompt_dir, step_dir))

    def get_next(self):
        print_time(f"get_next {self.current_index}")
        if self.current_index >= len(self.data_items):
            self.current_index = 0
            return None

        prompt_dir, step_dir = self.data_items[self.current_index]
        data_path = os.path.join(self.calibration_dataset_path, prompt_dir, step_dir)
        
        # Load the data - now expecting batch size 2 data
        sample = np.load(os.path.join(data_path, "sample.npy"))  # Shape: [2, 4, 64, 64]
        timestep = np.load(os.path.join(data_path, "timestep.npy"))  # Shape: [2]
        encoder_hidden_states = np.load(os.path.join(data_path, "encoder_hidden_states.npy"))  # Shape: [2, 77, 768]
        
        batch_data = {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states
        }

        self.current_index += 1
        return batch_data

    def __len__(self):
        return len(self.data_items)

    def set_range(self, start_index: int, end_index: int):
        self.current_index = start_index
        self.end_index = end_index

def tokenize_prompt(prompt):
    """Tokenize a prompt using the CLIP tokenizer."""
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="np",
    )
    return text_inputs.input_ids.astype(np.int32)

def get_text_embeddings(prompt, ort_sess_text_encoder):
    """Get text embeddings for a prompt using specified text encoder."""
    # Get text embeddings
    input_ids = tokenize_prompt(prompt)
    text_encoder_output = ort_sess_text_encoder.run(None, {"input_ids": input_ids})[0]

    # Get uncond embeddings
    uncond_input = tokenize_prompt("")
    uncond_embeddings = ort_sess_text_encoder.run(None, {"input_ids": uncond_input})[0]

    # Concatenate embeddings
    text_embeddings = np.concatenate([uncond_embeddings, text_encoder_output])

    return text_embeddings

def denoise_latents(latents, text_embeddings, ort_sess_unet, num_inference_steps=20):
    """Perform multiple denoising steps with fixed batch size 2"""
    scheduler.set_timesteps(num_inference_steps)
    latents = latents * scheduler.init_noise_sigma

    for i, t in enumerate(scheduler.timesteps):
        print_time(f"Denoise: {i} {t}")
        
        # Scale the latents according to the timestep
        latents_input = scheduler.scale_model_input(torch.from_numpy(latents), t).numpy()
        
        # Create batch of size 2 by duplicating the latents
        #latent_model_input = np.concatenate([latents_input, latents_input])
        latent_model_input = np.repeat(latents_input, 2, axis=0)  # Shape: [2, 4, 64, 64]
        
        # Create timestep tensor with shape [2] for fixed batch size
        timestep = np.array([t, t], dtype=np.int64)
        
        # Save inputs if this is a step to capture
        if i in num_steps_to_capture:
            print(f"capturing data step {i}")
            step_dir = os.path.join(prompt_dir, f"step_{i}")
            os.makedirs(step_dir, exist_ok=True)
            
            # Save inputs with batch size 2
            np.save(os.path.join(step_dir, "sample.npy"), latent_model_input)
            np.save(os.path.join(step_dir, "timestep.npy"), timestep)
            np.save(os.path.join(step_dir, "encoder_hidden_states.npy"), text_embeddings)
        
        # Run UNet with batch size 2
        noise_pred = ort_sess_unet.run(
            None,
            {
                "sample": latent_model_input,
                "encoder_hidden_states": text_embeddings,
                "timestep": timestep,
            },
        )[0]
        
        # Split and combine predictions
        noise_pred_uncond, noise_pred_text = noise_pred[0:1], noise_pred[1:2]
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(
            torch.from_numpy(noise_pred), t, torch.from_numpy(latents)
        ).prev_sample.numpy()

    return latents

def decode_latents(latents, ort_sess_vae):
    """Decode latents using VAE"""
    scaled_latents = 1 / 0.18215 * latents
    decoded = ort_sess_vae.run(None, {"latent_sample": scaled_latents})[0]
    return decoded
            
def convert_to_image(decoded):
    """Convert VAE decoder output to PIL Image"""
    decoded = decoded.squeeze(0)
    decoded = ((decoded + 1) * 127.5).clip(0, 255).astype(np.uint8)
    decoded = decoded.transpose(1, 2, 0)  # CHW -> HWC
    return Image.fromarray(decoded)

def create_calibration_dataset(prompts, ort_sess_text_encoder, ort_sess_unet, ort_sess_vae, output_dir, num_steps_to_capture, total_steps):
    """
    Generates calibration dataset by running the UNet multiple times for each prompt
    and capturing inputs at specified steps.
    """
    os.makedirs(output_dir, exist_ok=True)
    for prompt_idx, (key, prompt) in enumerate(tqdm(prompts.items(), desc="Generating Calibration Data")):
        print_time(f"prompt: {prompt} {prompt_idx}")
        # Encode text prompt
        text_embeddings = get_text_embeddings(prompt, ort_sess_text_encoder)

        # Initial random noise (sample)
        latents = np.random.randn(1, 4, 64, 64).astype(np.float32)

        # Create a directory for the current prompt
        global prompt_dir
        prompt_dir = os.path.join(output_dir, f"prompt_{prompt_idx}")
        os.makedirs(prompt_dir, exist_ok=True)
        
        # Run UNet for multiple steps with CFG
        latents = denoise_latents(latents, text_embeddings, ort_sess_unet, num_inference_steps=total_steps)
        print_time(f"Decode {prompt_idx}")
        decoded = decode_latents(latents, ort_sess_vae)
        
        image = convert_to_image(decoded)
        
        # Save image
        shortened_prompt = prompt.replace(" ", "")
        if len(shortened_prompt) > 15:
            shortened_prompt = shortened_prompt[:15]
        
        image_path = os.path.join(output_dir, "generated_images")
        os.makedirs(image_path, exist_ok=True)
        
        image_name = f"sd_{key}_{shortened_prompt}.png"
        imagefile_path = os.path.join(image_path, image_name)
        print_time(f"Save image {image_name}")
        image.save(imagefile_path)

def clear_directory(target_directory):
    """Deletes all files and folders within the target directory."""
    if not os.path.exists(target_directory):
        Path(target_directory).mkdir(exist_ok=True)
        return

    for root, dirs, files in os.walk(target_directory, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"Error deleting file {file_path}: {e}")

        for dir in dirs:
            dir_path = os.path.join(root, dir)
            try:
                shutil.rmtree(dir_path)
            except OSError as e:
                print(f"Error deleting directory {dir_path}: {e}")

def print_time(message):
    global nowtime
    elapsed_time = time.time() - nowtime
    print(f"{elapsed_time:.2f} {message}")
    nowtime = time.time()

if __name__ == "__main__":
    # Configuration
    prompts_file = "prompts.yaml"
    text_encoder_model_path = "fixedsize/text_encoder/model.onnx"
    vae_model_path = "fixedsize/vae_decoder/model.onnx"
    unet_model_path = "fixedsize/unet_preproc_sim/model.onnx"

    
    output_dir = "calibration_dataset"
    num_steps_to_capture = [0, 2, 4, 6, 10]
    total_steps = 20
    number_prompts = 2
    
    clear_directory(output_dir)
    guidance_scale = 7.5

    # Load prompts from YAML
    with open(prompts_file, "r") as f:
        prompts_dict = yaml.safe_load(f)

    if number_prompts is not None:
        prompts = dict(list(prompts_dict.items())[:number_prompts])

    # Load tokenizer and scheduler
    print_time("Load Tokenizer")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    print_time("Load scheduler")
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        clip_sample=False,
        set_alpha_to_one=False,
    )

    if torch.cuda.is_available():
        cuda_options = {'device_id': 0}
        providers = [('CUDAExecutionProvider', cuda_options), 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']

    # Load ONNX models
    print_time("Load models")
    ort_sess_text_encoder = ort.InferenceSession(text_encoder_model_path, providers=providers)
    print_time("Loaded text encoder")
    ort_sess_unet = ort.InferenceSession(unet_model_path, providers=providers)
    print_time("Loaded unet")
    ort_sess_vae = ort.InferenceSession(vae_model_path, providers=providers) 

    print_time("VAE loaded, start create dataset")
    create_calibration_dataset(prompts, ort_sess_text_encoder, ort_sess_unet, ort_sess_vae, output_dir, num_steps_to_capture, total_steps)
    print_time("Finished")