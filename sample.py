import os
import gc
import math
import torch
import torch.nn as nn
import requests
import numpy as np
import random
from PIL import Image
from tqdm import tqdm 
from abc import abstractmethod
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print(f"GPU Name: {device_name}")
else:
    print("No GPU available, using CPU.")


#Step 1 : Instantiate the decoder of the real palmprint VQModel

ckpt_path = 'vqmodel_checkpoint.ckpt'
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

# Simplified VQModel class
class VQModel(torch.nn.Module):
    def __init__(self, ddconfig, embed_dim=3,n_embed=8192):
        super().__init__()
        from taming.modules.diffusionmodules.model import Encoder, Decoder
        from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
        # self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=None,
                                        sane_index_shape=False)
        # self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

    # def encode(self, x):  
    #     h = self.quant_conv(self.encoder(x))
    #     return h

    def decode(self, x, force_not_quantize=False):  
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(x)
        else:
            quant = x
        dec = self.decoder(self.post_quant_conv(quant))
        return dec

# Initialize simplified model
vq_model = VQModel(  
    ddconfig={
        'double_z': False,
        'z_channels': 3,
        'resolution': 256,
        'in_channels': 3,
        'out_ch': 3,
        'ch': 128,
        'ch_mult': [1, 2, 4],
        'num_res_blocks': 2,
        'attn_resolutions': [],
        'dropout': 0.0
    },
    embed_dim=3,
    n_embed=8192
)

# Load weights
# Filter out parameters that are not related to the model
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
elif 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint  

filtered_state_dict = {k: v for k, v in state_dict.items() if k in vq_model.state_dict()}
vq_model.load_state_dict(filtered_state_dict, strict=False)

del checkpoint
del state_dict
del filtered_state_dict

vq_model = vq_model.to(device)
for param in vq_model.parameters():
    param.requires_grad = False
# Set the model to evaluation mode
vq_model.eval()

#Step 2 : Instantiate UNet
from ect.training.networks import ECMPrecond

Model = ECMPrecond(
        img_resolution  = 64,                     # Image resolution.
        img_channels    = 4,                       # Number of color channels.
        label_dim       = 0,                # Number of class labels, 0 = unconditional.
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        sigma_min       = 0.003,            # Minimum supported noise level.
        sigma_max       = float('inf'),     # Maximum supported noise level.
        sigma_data      = 0.5,              # Expected standard deviation of the training data.
        model_type      = 'SongUNet',   # Class name of the underlying model.
        model_channels      = 192          # Base multiplier for the number of channels.
    )

ckpt_path_u = 'LECT_Model.ckpt'
checkpoint_u = torch.load(ckpt_path_u, map_location=device, weights_only=True)

if 'model_state_dict' in checkpoint_u:
    state_dict = checkpoint_u['model_state_dict']
elif 'state_dict' in checkpoint_u:
    state_dict = checkpoint_u['state_dict']
else:
    state_dict = checkpoint_u 

filtered_state_dict = {k: v for k, v in state_dict.items() if k in Model.state_dict()}
Model.load_state_dict(filtered_state_dict, strict=False)

del checkpoint_u
del state_dict
del filtered_state_dict

Model = Model.to(device)
for param in Model.parameters():
    param.requires_grad = False

Model.eval()

#Step 4 : Import the generator function
from ect.training.networks import generator_fn


# Step 5: Process images in folders and save results
input_base_path = 'BEZIER'

output_base_path_onestep = 'LCM_onestep'
output_base_path_twostep = 'LCM_twostep'
os.makedirs(output_base_path_onestep, exist_ok=True)
os.makedirs(output_base_path_twostep, exist_ok=True)

image_size = 64
batch_size = 25  # Adjust based on your GPU memory
transform = transforms.Compose([
    transforms.Resize((image_size, image_size),interpolation=Image.BILINEAR),
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
])

class OrderedImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(image_path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, self.image_files[idx]  # Return image and filename

def save_image(img_tensor, fname, folder):
    img = ((img_tensor + 1) / 2 * 255).clamp(0, 255).byte()
    pil = Image.fromarray(img.permute(1, 2, 0).cpu().numpy())
    pil.save(os.path.join(folder, fname))

def process_images_with_dataloader(dataset, output_folder1, output_folder2):
    os.makedirs(output_folder1, exist_ok=True)
    os.makedirs(output_folder2, exist_ok=True)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    for batch_images, filenames in dataloader:
        images_tensor = batch_images.to(device)

        with torch.no_grad():
            noise = torch.randn(images_tensor.shape[0], 3, image_size, image_size, device=images_tensor.device)
            onestep_latents, twostep_latents = generator_fn(Model, noise, images_tensor)
            generated_images_1 = vq_model.decode(onestep_latents).clamp(-1, 1)
            generated_images_2 = vq_model.decode(twostep_latents).clamp(-1, 1)

        with ThreadPoolExecutor(max_workers=16) as executor:
            for img1, img2, fname in zip(generated_images_1, generated_images_2, filenames):
                executor.submit(save_image, img1, fname, output_folder1)
                executor.submit(save_image, img2, fname, output_folder2)

def process_all_folders():
    total_folders = 4000  
    progress_bar = tqdm(total=total_folders, desc="Processing folders")

    for folder_idx in range(total_folders):
        folder_name = f"{folder_idx:04d}"
        input_folder = os.path.join(input_base_path, folder_name)
        output_folder1 = os.path.join(output_base_path_onestep, folder_name)
        output_folder2 = os.path.join(output_base_path_twostep, folder_name)

        if os.path.isdir(input_folder):
            dataset = OrderedImageDataset(folder_path=input_folder, transform=transform)
            process_images_with_dataloader(dataset, output_folder1, output_folder2)

        progress_bar.update(1)

    progress_bar.close()

if __name__ == "__main__":
    process_all_folders()