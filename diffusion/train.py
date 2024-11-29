from typing import Optional

import torch
from ddpm import DDPM
import tqdm
from torch.utils.data.dataloader import DataLoader
from unet import UNet
from data import SequencesDataset

def train(
    model: DDPM,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: str,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    save_every_epoch: Optional[int] = None
):
    training_losses = []
    val_losses = []
    for epoch in range(epochs):
        model.train(True)
        training_loss = 0
        val_loss = 0
        pbar = tqdm.tqdm(train_dataloader)
        for index, (imgs, previous_frames, previous_actions) in enumerate(pbar):
            optimizer.zero_grad()
            
            imgs = imgs.to(device)
            previous_frames = previous_frames.to(device)
            previous_actions = previous_actions.to(device)
    
            loss = model.forward(imgs, previous_frames, previous_actions)
    
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            pbar.set_description(f"loss for epoch {epoch}: {training_loss / (index + 1):.4f}")
        model.eval()
        with torch.no_grad():
            for (imgs, previous_frames, previous_actions) in val_dataloader:
                imgs = imgs.to(device)
                previous_frames = previous_frames.to(device)
                previous_actions = previous_actions.to(device)
    
                loss = model(imgs, previous_frames, previous_actions)
        
                val_loss += loss.item()
        training_losses.append(training_loss / len(val_dataloader))
        val_losses.append(val_loss / len(val_dataloader))
        if save_every_epoch is not None and epoch > 0 and epoch % save_every_epoch == 0:
            torch.save(model.state_dict(), f"model_{epoch}.pth")
    return training_losses, val_losses

if __name__ == "__main__":
    T = 1000
    input_channels = 3
    context_length = 3
    actions_count = 5
    T = 5
    batch_size = 3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # For Mac OS
    if torch.backends.mps.is_available():
        device = "mps"

    ddpm = DDPM(
        T = T,
        eps_model=UNet(
            in_channels=input_channels * (context_length + 1),
            out_channels=3,
            T=T+1,
            actions_count=actions_count,
            seq_length=context_length
        ),
        context_length=context_length,
        device=device
    )

    import torchvision.transforms as transforms
    from torch.utils.data import random_split

    transform_to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5,.5,.5), (.5,.5,.5))
    ])

    dataset = SequencesDataset(
        images_dir="snake_agent/q_learning/snapshots",
        actions_path="snake_agent/q_learning/actions",
        seq_length=context_length,
        transform=transform_to_tensor
    )

    total_size = len(dataset)
    train_size = int(0.8 * total_size)  # 80% for training
    valid_size = total_size - train_size  # 20% for validation

    # Split the dataset
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, valid_size]
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
            
    obs = next(iter(train_dataloader))[1]
    img = Image.fromarray((obs[0,0] * 127.5 + 127.5).long().clip(0,255).permute(1,2,0).detach().cpu().numpy().astype(np.uint8))
    plt.imsave(f'test_0.jpg', img)
    img = Image.fromarray((obs[0,1] * 127.5 + 127.5).long().clip(0,255).permute(1,2,0).detach().cpu().numpy().astype(np.uint8))
    plt.imsave(f'test_1.jpg', img)
    img = Image.fromarray((obs[0,2] * 127.5 + 127.5).long().clip(0,255).permute(1,2,0).detach().cpu().numpy().astype(np.uint8))
    plt.imsave(f'test_2.jpg', img)

    # _, val_losses = train(
    #     model=ddpm,
    #     optimizer=torch.optim.Adam(params=ddpm.parameters(), lr=2e-4),
    #     epochs=1,
    #     device=device,
    #     train_dataloader=train_dataloader,
    #     val_dataloader=val_dataloader
    # )