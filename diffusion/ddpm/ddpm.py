import torch.nn as nn
import torch
from typing import List, Tuple, Optional
import numpy as np
import torch.nn.functional as F

class DDPM(nn.Module):
    def __init__(
        self,
        T: int,
        eps_model: nn.Module,
        context_length: int,
        device: str
    ):
        super().__init__()
        self.T = T
        self.eps_model = eps_model.to(device)
        self.device = device
        self.context_length = context_length
        beta_schedule = torch.linspace(1e-4, 0.02, T + 1, device=device)
        alpha_t_schedule = 1 - beta_schedule
        bar_alpha_t_schedule = torch.cumprod(alpha_t_schedule.detach().cpu(), 0).to(device)
        sqrt_bar_alpha_t_schedule = torch.sqrt(bar_alpha_t_schedule)
        sqrt_minus_bar_alpha_t_schedule = torch.sqrt(1 - bar_alpha_t_schedule)
        self.register_buffer("beta_schedule", beta_schedule)
        self.register_buffer("alpha_t_schedule", alpha_t_schedule)
        self.register_buffer("bar_alpha_t_schedule", bar_alpha_t_schedule)
        self.register_buffer("sqrt_bar_alpha_t_schedule", sqrt_bar_alpha_t_schedule)
        self.register_buffer("sqrt_minus_bar_alpha_t_schedule", sqrt_minus_bar_alpha_t_schedule)
        self.criterion = nn.MSELoss()

    def forward(self, imgs: torch.Tensor, prev_frames: torch.Tensor, prev_actions: torch.Tensor):
        assert prev_frames.shape[1] == prev_actions.shape[1] == self.context_length
        t = torch.randint(low=1, high=self.T+1, size=(imgs.shape[0],), device=self.device)
        imgs = F.pad(imgs, (2, 2, 2, 2))
        noise = torch.randn_like(imgs, device=self.device)
        batch_size, channels, width, height = imgs.shape
        noise_imgs = self.sqrt_bar_alpha_t_schedule[t].view((batch_size, 1, 1 ,1)) * imgs \
            + self.sqrt_minus_bar_alpha_t_schedule[t].view((batch_size, 1, 1, 1)) * noise
        prev_frames = F.pad(prev_frames, (2, 2, 2, 2))
        noise_imgs = torch.concat([noise_imgs[:, None, :, :, :], prev_frames], dim=1).flatten(1,2)

        pred_noise = self.eps_model(noise_imgs, t.unsqueeze(1), prev_actions)

        return self.criterion(pred_noise, noise)
    
    def sample(
        self,
        size: Tuple[int],
        prev_frames: torch.Tensor,
        prev_actions: torch.Tensor, 
        range_t: Optional[List[int]] = None # range(self.T, 0, -1)
    ):
        self.eval()
        with torch.no_grad():
            x_t = torch.randn(1, *size, device=self.device)
            x_t = F.pad(x_t, (2, 2, 2, 2))
            prev_frames = F.pad(prev_frames, (2, 2, 2, 2))
            if range_t is None:
                range_t = range(self.T, 0, -1)
            for t in range_t:
                z = torch.randn_like(x_t, device=self.device) if t > 0 else 0
                t_tensor = torch.tensor([t], device=self.device).repeat(x_t.shape[0], 1)
                big_x_t = torch.concat([x_t[:, None, :, :, :], prev_frames], dim=1).flatten(1,2)
                pred_noise = self.eps_model(big_x_t, t_tensor, prev_actions)
                
                x_t = 1 / torch.sqrt(self.alpha_t_schedule[t]) * \
                    (x_t - pred_noise * (1 - self.alpha_t_schedule[t]) / self.sqrt_minus_bar_alpha_t_schedule[t]) + \
                    torch.sqrt(self.beta_schedule[t]) * z
            return x_t
        
    def ddim_sample(
        self,
        size: Tuple[int],
        prev_frames: torch.Tensor,
        prev_actions: torch.Tensor, 
        steps: int
    ):
        self.eval()
        with torch.no_grad():
            x_t = torch.randn(1, *size, device=self.device)
            x_t = F.pad(x_t, (2, 2, 2, 2))
            prev_frames = F.pad(prev_frames, (2, 2, 2, 2))
            step_size = self.T // steps
            range_t = range(self.T, -1, -step_size)
            next_range_t = range_t[1:]
            range_t = range_t[:-1]
            for i, j in zip(range_t, next_range_t):
                t_tensor = torch.tensor([i], device=self.device).repeat(x_t.shape[0], 1)

                big_x_t = torch.concat([x_t[:, None, :, :, :], prev_frames], dim=1).flatten(1,2)
                pred_noise = self.eps_model(big_x_t, t_tensor, prev_actions)
                alpha = self.bar_alpha_t_schedule[i]
                next_alpha = self.bar_alpha_t_schedule[j]
                x0 = (x_t - torch.sqrt(1 - alpha) * pred_noise) / torch.sqrt(alpha)
                new_xt = torch.sqrt(1 - next_alpha) * pred_noise

                x_t = torch.sqrt(next_alpha) * x0 + new_xt
            return x_t
        
    # def sample(
    #     self,
    #     size: Tuple[int],
    #     prev_frames: torch.Tensor,
    #     prev_actions: torch.Tensor, 
    #     num_steps=4
    # ):
    #     self.eval()
    #     with torch.no_grad():
    #         step_size = 1000 // num_steps
    #         timesteps = list(range(1000-step_size, -1, -step_size))
            
    #         x_t = torch.randn(1, *size, device=self.device)
    #         x_t = F.pad(x_t, (2, 2, 2, 2))
    #         prev_frames = F.pad(prev_frames, (2, 2, 2, 2))
    #         for t in timesteps:
    #             t_tensor = torch.tensor([t], device=self.device).repeat(x_t.shape[0], 1)
    #             big_x_t = torch.concat([x_t[:, None, :, :, :], prev_frames], dim=1).flatten(1,2)
    #             noise_pred = self.eps_model(big_x_t, t_tensor, prev_actions)

    #             alpha_t = self.alpha_t_schedule[t]
    #             alpha_prev = self.alpha_t_schedule[max(0, t-step_size)]
    #             beta_t = 1 - alpha_t/alpha_prev
                
    #             # Modified update step
    #             x_prev = (1/torch.sqrt(alpha_prev)) * (x_t - 
    #                     (beta_t/torch.sqrt(1-alpha_t)) * noise_pred)
                
    #             if t > 0:
    #                 noise = torch.randn_like(x_t)
    #                 x_prev = x_prev + torch.sqrt(beta_t) * noise
                    
    #             x_t = x_prev
    #         return x_t
        
if __name__ == "__main__":
    size = (60, 60)
    input_channels = 3
    context_length = 4
    actions_count = 5
    T = 1000
    batch_size = 3

    from diffusion.modules_v2 import UNet

    unet = UNet((input_channels) * (context_length + 1), 3, T, actions_count, context_length)
    ddpm = DDPM(
        T=T,
        eps_model=unet,
        context_length=context_length,
        device="cpu"
    )

    img = torch.randn((batch_size, input_channels, *size))
    prev_frames = torch.randn((batch_size, context_length, input_channels, *size))
    # frames = torch.concat([img[:, None, :, :, :], prev_frames], dim=1).flatten(1,2)

    prev_actions = torch.randint(low=0, high=actions_count, size=(batch_size, context_length))
    t = torch.randint(1, T + 1, (batch_size,))
    ddpm.forward(img, prev_frames, prev_actions)
    # unet.forward(frames, t.unsqueeze(1), prev_actions)