from typing import Tuple
import os

import pygame
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

from edm.edm import EDM
import edm.modules as modules
from data import SequencesDataset
import random

class Game:
    def __init__(
        self,
        size: Tuple[int, int],
        ddmp: EDM,
        context_length: int,
        fps: int,
        device: str,
        dataset: SequencesDataset
    ) -> None:
        self.ddpm = ddmp
        _, self.height, self.width = size
        self.fps = fps
        self.context_length = context_length
        self.size = size
        self.device = device
        self.dataset = dataset

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

        def reset() -> Tuple[torch.Tensor, torch.Tensor]:
            index = random.randint(0, len(self.dataset) - 1)
            _, last_imgs, actions = dataset[index]

            obs = last_imgs.to(device)
            actions = actions.to(device)
            return obs, actions
        obs, actions = reset()
        number_frame = 0
        while True:
            action = 0
            pygame.event.pump()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        obs, actions = reset()
                    elif event.key == pygame.K_UP:
                        actions = torch.concat((actions[1:], torch.tensor(1, device=self.device).unsqueeze(0)))
                    elif event.key == pygame.K_DOWN:
                        actions = torch.concat((actions[1:], torch.tensor(2, device=self.device).unsqueeze(0)))
                    elif event.key == pygame.K_RETURN:
                        actions = torch.concat((actions[1:], torch.tensor(3, device=self.device).unsqueeze(0)))
            
            next_obs = self.ddpm.sample(
                self.size,
                obs.unsqueeze(0),
                torch.tensor(actions, dtype=torch.int).to(self.device).unsqueeze(0),
                num_steps=10
            )

            draw_game(next_obs)
            obs = torch.concat((obs[1:],next_obs[0][:, 2:-2, 2:-2].unsqueeze(0)))

            pygame.display.flip()  # update screen
            clock.tick(self.fps)  # ensures game maintains the given frame rate
            number_frame += 1
            if number_frame % 300:
                obs, actions = reset()
                draw_game(obs[-1].unsqueeze(0))

        pygame.quit()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # For Mac OS
    if torch.backends.mps.is_available():
        device = "mps"
    PATH_TO_READY_MODEL = "./test_models/diffusion/model_12_edm.pth"
    CONTEXT_LENGTH = 4
    ACTIONS_COUNT = 5
    ddpm = EDM(
        p_mean=-1.2,
        p_std=1.2,
        sigma_data=0.5,
        model=modules.UNet(
            in_channels=3 * (CONTEXT_LENGTH + 1),
            out_channels=3,
            T=None,
            actions_count=ACTIONS_COUNT,
            seq_length=CONTEXT_LENGTH
        ),
        context_length=CONTEXT_LENGTH,
        device=device
    )
    ddpm.load_state_dict(torch.load(PATH_TO_READY_MODEL, map_location=device)["model"])
    import torchvision.transforms as transforms
    transform_to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5,.5,.5), (.5,.5,.5))
    ])

    dataset = SequencesDataset(
        images_dir="./training_data/snapshots",
        actions_path="./training_data/actions",
        seq_length=CONTEXT_LENGTH,
        transform=transform_to_tensor
    )

    dir_path = os.path.dirname(os.path.realpath(__file__))
    game = Game(
        size=(3, 60,60),
        ddmp=ddpm,
        context_length=CONTEXT_LENGTH,
        fps=15,
        device=device,
        dataset=dataset
    )
    # os.environ["SDL_VIDEODRIVER"] = "dummy"
    game.run()
