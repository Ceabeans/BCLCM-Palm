import os
import gc
import math
import torch
import copy
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
import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides')

torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print(f"GPU Name: {device_name}")
else:
    print("No GPU available, using CPU.")

#Step 1 : Define DataLoader

class PalmDataset(Dataset):
    def __init__(self, palm_dir, label_dir, palm_size=(256, 256), label_size=(64, 64)):
        self.palm_dir = palm_dir
        self.label_dir = label_dir
        self.palm_images = sorted([
            f for f in os.listdir(palm_dir)
            if os.path.isfile(os.path.join(palm_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.label_images = sorted([
            f for f in os.listdir(label_dir)
            if os.path.isfile(os.path.join(label_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        assert len(self.palm_images) == len(self.label_images), "The number of images does not match!"

        self.palm_size = palm_size
        self.label_size = label_size

        # Define transform for color images
        self.palm_transform = transforms.Compose([
            transforms.Resize(self.palm_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # RGB
        ])
        
        # Define transform for grayscale images
        self.label_transform = transforms.Compose([
            transforms.Resize(self.label_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Gray
        ])

    def __len__(self):
        return len(self.palm_images)

    def __getitem__(self, idx):
        palm_path = os.path.join(self.palm_dir, self.palm_images[idx])
        label_path = os.path.join(self.label_dir, self.label_images[idx])

        palm_image = Image.open(palm_path).convert("RGB")
        label_image = Image.open(label_path).convert("L")
        
        palm_tensor = self.palm_transform(palm_image)
        label_tensor = self.label_transform(label_image)

        if palm_tensor.shape[0] != 3:
            raise ValueError(f"palm_image not RGB, shape = {palm_tensor.shape}")
        if label_tensor.shape[0] != 1:
            raise ValueError(f"label_image not grey, shape = {label_tensor.shape}")

        return palm_tensor, label_tensor


real_image_folder = 'Data/trainB'
bezier_image_folder='Data/binary_images_fake_bezier'

dataset = PalmDataset(real_image_folder, bezier_image_folder)

def custom_collate_fn(batch):
    palm_images = torch.stack([item[0] for item in batch])
    label_images = torch.stack([item[1] for item in batch])
    return palm_images, label_images

train_loader = DataLoader(dataset, 
                          batch_size=32, 
                          shuffle=True, 
                          num_workers=16, 
                          pin_memory=True, 
                          collate_fn=custom_collate_fn
                         )

#Step 2 : Instantiate the encoder of the real palmprint VQModel

# Load model parameters
ckpt_path = 'vqmodel_checkpoint.ckpt'
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

# Simplified VQModel class
class VQModel(torch.nn.Module):
    def __init__(self, ddconfig, embed_dim=3,n_embed=8192):
        super().__init__()
        from taming.modules.diffusionmodules.model import Encoder, Decoder
        # from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
        self.encoder = Encoder(**ddconfig)
        # self.decoder = Decoder(**ddconfig)
        # self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
        #                                 remap=None,
        #                                 sane_index_shape=False)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        # self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

    def encode(self, x):  
        h = self.quant_conv(self.encoder(x))
        return h

    # def decode(self, x, force_not_quantize=False): 
    #     if not force_not_quantize:
    #         quant, emb_loss, info = self.quantize(x)
    #     else:
    #         quant = x
    #     dec = self.decoder(self.post_quant_conv(quant))
    #     return dec

# Initialize simplified model
vq_model = VQModel(  # Change the name of the instantiated object to lowercase to avoid confusion with the class name
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

# Load only the model state dict
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

vq_model.eval()

#Step 3 : Instantiate UNet
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

Model.to(device)
Model.train()

#Step 4 : Instantiate loss function
from ect.training.loss import ECMLoss
loss_fn = ECMLoss(q=2)

#Step 5 : Train Model and save the model

# class EMA:
#     def __init__(self, model, beta=0.9999):
#         self.beta = beta
#         self.model = model
#         self.ema_model = copy.deepcopy(model)
#         self.ema_model.eval()  # EMA 模型不训练
#         for p in self.ema_model.parameters():
#             p.requires_grad_(False)

#     @torch.no_grad()
#     def update(self):
#         for p_ema, p in zip(self.ema_model.parameters(), self.model.parameters()):
#             p_ema.data.mul_(self.beta).add_(p.data, alpha=1. - self.beta)

#     def state_dict(self):
#         return self.ema_model.state_dict()

# ema = EMA(Model, beta=0.9993)

optimizer = torch.optim.RAdam(Model.parameters(), lr=1e-04)
# cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
num_epochs = 100
accum_steps = 4  # Gradient accumulation steps


for epoch in tqdm(range(num_epochs), desc='Epochs'):
    stage = (epoch + 1) // 10
    loss_fn.update_schedule(stage)

    optimizer.zero_grad()  # Initialize gradients (before each epoch)

    for step, (palm_images, label_images) in enumerate(train_loader):
        palm_images = palm_images.to(device)
        label_images = label_images.to(device)

        with torch.no_grad():
            latent = vq_model.encode(palm_images).detach()

        loss = loss_fn(net=Model, images=latent, map=label_images)
        loss = loss.mean() / accum_steps  # Average loss (important)

        loss.backward()

        # Accumulate gradients and update parameters
        if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
            optimizer.step()
            # ema.update()
            optimizer.zero_grad()

    # cosine_scheduler.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item() * accum_steps:.4f}')  # 显示真实 loss

    if (epoch + 1) % 10 == 0:
        torch.save(Model.state_dict(), 'LECT_Model.ckpt')
        print("Model parameters saved to LECT_Model.ckpt")