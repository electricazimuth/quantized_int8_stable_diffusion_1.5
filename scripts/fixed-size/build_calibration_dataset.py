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
        Now handles both conditional and unconditional data.
        """
        prompt_dirs = [d for d in os.listdir(self.calibration_dataset_path) 
            if os.path.isdir(os.path.join(self.calibration_dataset_path, d))
        ]
        for prompt_dir in prompt_dirs:
            step_dirs = [d for d in os.listdir(os.path.join(self.calibration_dataset_path, prompt_dir))
                if os.path.isdir(os.path.join(self.calibration_dataset_path, prompt_dir, d))
            ]
            for step_dir in step_dirs:
                # Add both conditional and unconditional entries
                self.data_items.append((prompt_dir, step_dir, 0))  # 0 for unconditional
                self.data_items.append((prompt_dir, step_dir, 1))  # 1 for conditional

    def get_next(self):
        print_time(f"get_next {self.current_index}")
        if self.current_index >= len(self.data_items):
            self.current_index = 0
            return None

        prompt_dir, step_dir, cond_flag = self.data_items[self.current_index]
        data_path = os.path.join(self.calibration_dataset_path, prompt_dir, step_dir)
        
        # Load the data
        sample = np.load(os.path.join(data_path, "sample.npy"))
        timestep = np.load(os.path.join(data_path, "timestep.npy"))
        encoder_hidden_states = np.load(os.path.join(data_path, "encoder_hidden_states.npy"))
        
        # Extract either conditional or unconditional data
        if cond_flag == 0:
            # Unconditional - take first batch
            sample = sample[:1]  # Shape: [1, 4, 64, 64]
            encoder_hidden_states = encoder_hidden_states[:1]  # Shape: [1, 77, 768]
        else:
            # Conditional - take second batch
            sample = sample[1:2]  # Shape: [1, 4, 64, 64]
            encoder_hidden_states = encoder_hidden_states[1:2]  # Shape: [1, 77, 768]
        
        batch_data = {
            "sample": sample,
            "timestep": timestep,  # Shape: [1]
            "encoder_hidden_states": encoder_hidden_states
        }

        self.current_index += 1
        return batch_data

    def __len__(self):
        return len(self.data_items)

    def set_range(self, start_index: int, end_index: int):
        self.current_index = start_index
        self.end_index = end_index

# Tokenization function
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

# Text encoding function
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
    """Perform multiple denoising steps handling fixed size inputs"""
    scheduler.set_timesteps(num_inference_steps)
    latents = latents * scheduler.init_noise_sigma

    for i, t in enumerate(scheduler.timesteps):
        print_time(f"Denoise: {i} {t}")
        
        # Scale the latents according to the timestep
        latents_input = scheduler.scale_model_input(torch.from_numpy(latents), t).numpy()
        
        # Process unconditional and conditional separately
        latent_model_input_uncond = latents_input.copy()
        latent_model_input_text = latents_input.copy()
        
        # Create timestep tensor with shape [1]
        timestep = np.array([t], dtype=np.int64)
        
        # Save both conditional and unconditional inputs if this is a step to capture
        if i in num_steps_to_capture:
            print(f"capturing data step {i}")
            step_dir = os.path.join(prompt_dir, f"step_{i}")
            os.makedirs(step_dir, exist_ok=True)
            
            # Save combined inputs (both conditional and unconditional)
            sample_combined = np.concatenate([latent_model_input_uncond, latent_model_input_text])
            np.save(os.path.join(step_dir, "sample.npy"), sample_combined)
            np.save(os.path.join(step_dir, "timestep.npy"), timestep)
            np.save(os.path.join(step_dir, "encoder_hidden_states.npy"), text_embeddings)
        
        # Run UNet for unconditional input
        noise_pred_uncond = ort_sess_unet.run(
            None,
            {
                "sample": latent_model_input_uncond,
                "encoder_hidden_states": text_embeddings[:1],
                "timestep": timestep,
            },
        )[0]
        
        # Run UNet for conditional input
        noise_pred_text = ort_sess_unet.run(
            None,
            {
                "sample": latent_model_input_text,
                "encoder_hidden_states": text_embeddings[1:],
                "timestep": timestep,
            },
        )[0]
        
        # Combine predictions with guidance
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(
            torch.from_numpy(noise_pred), t, torch.from_numpy(latents)
        ).prev_sample.numpy()

    return latents

def decode_latents(latents, ort_sess_vae):
        """Decode latents using VAE"""
        # Create VAE decoder session
        scaled_latents = 1 / 0.18215 * latents
        decoded = ort_sess_vae.run(None, {"latent_sample": scaled_latents})[0]
        return decoded
            
        
def convert_to_image(decoded):
        """Convert VAE decoder output to PIL Image"""
        decoded = decoded.squeeze(0)
        decoded = ((decoded + 1) * 127.5).clip(0, 255).astype(np.uint8)
        decoded = decoded.transpose(1, 2, 0)  # CHW -> HWC
        return Image.fromarray(decoded)

# Dataset creation function
def create_calibration_dataset(prompts, ort_sess_text_encoder, ort_sess_unet, ort_sess_vae, output_dir, num_steps_to_capture, total_steps):
    """
    Generates calibration dataset by running the UNet multiple times for each prompt
    and capturing inputs at specified steps.
    """
    os.makedirs(output_dir, exist_ok=True)
    for prompt_idx, (key, prompt) in enumerate(tqdm(prompts.items(), desc="Generating Calibration Data")):
        print_time(f"promtp: {prompt} {prompt_idx}")
        # Encode text prompt
        text_embeddings = get_text_embeddings(prompt, ort_sess_text_encoder)

        # Initial random noise (sample)
        batch_size = 1
        latents = np.random.randn(batch_size, 4, 64, 64).astype(np.float32)

        # Create a directory for the current prompt
        global prompt_dir
        prompt_dir = os.path.join(output_dir, f"prompt_{prompt_idx}")
        os.makedirs(prompt_dir, exist_ok=True)
        # Run UNet for multiple steps with CFG
        latents = denoise_latents(latents, text_embeddings, ort_sess_unet, num_inference_steps=total_steps)
        print_time("Decode {prompt_idx}")
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
    """Deletes all files and folders within the target directory.

    Args:
        target_directory: The path to the directory to clear.
    """
    if not os.path.exists(target_directory):
        Path(target_directory).mkdir(exist_ok=True)
        #print(f"Target directory '{target_directory}' does not exist. Skipping.")
        return

    for root, dirs, files in os.walk(target_directory, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
                #print(f"Deleted file: {file_path}")
            except OSError as e:
                print(f"Error deleting file {file_path}: {e}")

        for dir in dirs:
            dir_path = os.path.join(root, dir)
            try:
                shutil.rmtree(dir_path)
                #print(f"Deleted directory: {dir_path}")
            except OSError as e:
                print(f"Error deleting directory {dir_path}: {e}")

def print_time(message):
    global nowtime
    elapsed_time = time.time() - nowtime
    print(f"{elapsed_time:.2f} {message}")
    nowtime = time.time()

if __name__ == "__main__":
# Run dataset creation
    # Configuration
    prompts_file = "prompts.yaml"

    # Model Paths - update these for your local setup
    text_encoder_model_path = "fixedsize/text_encoder/model.onnx"  #  text encoder ONNX path
    vae_model_path = "fixedsize/vae_decoder/model.onnx"
    unet_model_path = "fixedsize/unet_torch32/model.onnx"  #  UNet ONNX path 
    # - fixed size unet_torch32/model.onnx
    # sample[FLOAT, 1x4x64x64]
    # timestep[INT64, 1]
    # encoder_hidden_states[FLOAT, 1x77x768]

    output_dir = "calibration_dataset"
    num_steps_to_capture = [0, 2, 4, 6, 10]
    total_steps = 20
    number_prompts = 32
    
    clear_directory(output_dir)
    guidance_scale = 7.5  # CFG scale

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

    ort_sess_text_encoder = ort.InferenceSession(text_encoder_model_path, providers=providers)
    print_time("Loaded text encoder")
    ort_sess_unet = ort.InferenceSession(unet_model_path, providers=providers)
    print_time("Loaded unet")
    ort_sess_vae = ort.InferenceSession(vae_model_path, providers=providers) 
    print_time("Loaded vae")
    print_time("Start create dataset")
    create_calibration_dataset(prompts, ort_sess_text_encoder, ort_sess_unet, ort_sess_vae, output_dir, num_steps_to_capture, total_steps)

    print_time("Finished")