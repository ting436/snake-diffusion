from typing import Any
import torch.nn as nn
import torch
from diffusers import UNet2DConditionModel, AutoencoderKL

class GameModel(nn.Module):
    def __init__(self, actions_count: int, context_length: int = 64) -> None:
        super().__init__()
        self.unet = UNet2DConditionModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4", 
            subfolder="unet",
            cache_dir="./diffusion/.cache/model"
        )
        self.vae = AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4", 
            subfolder="vae",
            cache_dir="./diffusion/.cache/vae"
        )
        self.context_length = context_length
        self.actions_embedding = nn.Embedding(actions_count, 768)
        self.context_length = context_length

    def __call__(self, actions: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
        # actions - [batch_size, length]
        # frames - [batch_size, length, channels, height, width]
        latents = torch.stack(
            [ 
                self.vae.encode(frames[i]).latent_dist.sample() for i in range(min(frames.shape[0], self.context_length))
            ],
            dim=0
        )
        
        act_emb = self.actions_embedding(actions[:, :self.context_length])
        return self.unet(
            sample=latents.flatten(1, 2),
            timestep=1,
            encoder_hidden_states=act_emb,
            return_dict=False
        )[0]

if __name__ == "__main__":
    model = GameModel(4)
    actions = torch.randint(low=0,high=3, size=(2,5))
    frames = torch.rand((2,5,3,120,160))
    model(actions, frames).shape

