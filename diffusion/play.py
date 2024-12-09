from typing import Tuple
import os

import pygame
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

from ddpm import DDPM
from unet import UNet

class Game:
    def __init__(
        self,
        size: Tuple[int, int],
        ddmp: DDPM,
        context_length: int,
        fps: int,
        default_img_path: str,
        device: str,
        T: int
    ) -> None:
        self.ddpm = ddmp
        _, self.height, self.width = size
        self.fps = fps
        self.context_length = context_length
        self.size = size

        import torchvision.transforms as transforms
        self.default_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.5,.5,.5), (.5,.5,.5))
        ])(Image.open(default_img_path).convert('RGB')).to(device)
        self.device = device
        self.T = T

    def run(self) -> None:
        pygame.init()

        screen = pygame.display.set_mode((self.width, self.height))
        clock = pygame.time.Clock()

        def draw_game(obs):
            assert obs.ndim == 4 and obs.size(0) == 1
            img = (obs[0] * 127.5 + 127.5).long().clip(0,255).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
            # pygame_image = np.array(img.resize((self.width, self.height), resample=Image.NEAREST)).transpose((1, 0, 2))
            surface = pygame.surfarray.make_surface(img)
            screen.blit(surface, (0, 0))

        obs, actions = [self.default_img.clone().to(self.device)] * self.context_length, [4] * self.context_length

        for i in range(3):
            action = 0
            pygame.event.pump()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        obs, actions = [self.default_img.clone().to(self.device)] * self.context_length, [4] * self.context_length
                    elif event.key == pygame.K_UP:
                        actions = actions[1:] + [1]
                    elif event.key == pygame.K_DOWN:
                        actions = actions[1:] + [2]
                    elif event.key == pygame.K_RETURN:
                        actions = actions[1:] + [3]
            
            next_obs = self.ddpm.sample(
                self.size,
                torch.stack(obs).unsqueeze(0),
                torch.tensor(actions, dtype=torch.int).to(self.device).unsqueeze(0)
                # [self.T // 2, 1]
            )

            draw_game(next_obs)
            for j in range(self.context_length):
                img = (obs[j] * 127.5 + 127.5).long().clip(0,255).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
                plt.imsave(f'test_{i}_{j}.jpg', img)
            obs = obs[1:] + [next_obs[0][:, 2:-2, 2:-2]]

            pygame.display.flip()  # update screen
            clock.tick(self.fps)  # ensures game maintains the given frame rate
            
            plt.imsave(f'test_{i}.jpg', pygame.surfarray.array3d(screen).transpose(1,0,2))

        pygame.quit()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # For Mac OS
    if torch.backends.mps.is_available():
        device = "mps"
    PATH_TO_READY_MODEL = "./test_models/diffusion/model_3.pth"
    T = 1000
    # T = 5
    CONTEXT_LENGTH = 4
    ACTIONS_COUNT = 5
    import old_unet
    ddpm = DDPM(
        T = T,
        eps_model=UNet(
            in_channels=3 * (CONTEXT_LENGTH + 1),
            out_channels=3,
            T=T+1,
            actions_count=ACTIONS_COUNT,
            seq_length=CONTEXT_LENGTH
        ),
        context_length=CONTEXT_LENGTH,
        device=device
    )
    ddpm.load_state_dict(torch.load(PATH_TO_READY_MODEL, map_location=device))

    dir_path = os.path.dirname(os.path.realpath(__file__))
    game = Game(
        size=(3, 60,60),
        ddmp=ddpm,
        context_length=CONTEXT_LENGTH,
        fps=15,
        default_img_path=os.path.join(dir_path, "default.jpg"),
        device=device,
        T=T
    )
    game.run()
