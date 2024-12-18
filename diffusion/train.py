from typing import Optional, Callable
import random

import torch
import tqdm
from torch.utils.data.dataloader import DataLoader
from data import SequencesDataset

def train(
    model: torch.nn.Module,
    epochs: int,
    device: str,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    save_every_epoch: Optional[int] = None,
    existing_model_path: Optional[str] = None,
    save_imgs: Callable[[torch.nn.Module, torch.Tensor, torch.Tensor, torch.Tensor, int], None] = None
):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=2e-4)
    epoch_range = range(1, epochs + 1)
    if existing_model_path is not None:
        parameters = torch.load(existing_model_path, map_location=device)
        model.load_state_dict(parameters["model"])
        optimizer.load_state_dict(parameters["optimizer"])
        epoch_range = range(parameters["epoch"] + 1, epochs + 1)

    training_losses = []
    val_losses = []
    for epoch in epoch_range:
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
            index_save = random.randint(0, len(val_dataloader) - 1)
            for index, (imgs, previous_frames, previous_actions) in enumerate(val_dataloader):
                imgs = imgs.to(device)
                previous_frames = previous_frames.to(device)
                previous_actions = previous_actions.to(device)
    
                loss = model(imgs, previous_frames, previous_actions)
        
                val_loss += loss.item()
                pbar.set_description(f"val loss for epoch {epoch}: {val_loss / (index + 1):.4f}")
                if index == index_save and save_imgs is not None:
                    save_imgs(model, previous_frames[0], previous_actions[0], imgs[0], epoch)
        training_losses.append(training_loss / len(val_dataloader))
        val_losses.append(val_loss / len(val_dataloader))
        if save_every_epoch is not None and epoch > 0 and epoch % save_every_epoch == 0:
            torch.save({ "model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch }, f"model_{epoch}.pth")
    return training_losses, val_losses

def _get_ddpm_model(context_length: int, device: str) -> torch.nn.Module:
    T = 1000
    input_channels = 3
    
    actions_count = 5
    
    import ddpm.modules_v3
    import ddpm.ddpm
    return ddpm.ddpm.DDPM(
        T = T,
        eps_model=ddpm.modules_v3.UNet(
            in_channels=input_channels * (context_length + 1),
            out_channels=3,
            T=T+1,
            actions_count=actions_count,
            seq_length=context_length
        ),
        context_length=context_length,
        device=device
    )

def _get_edm_model(context_length: int, device: str) -> torch.nn.Module:
    input_channels = 3
    
    actions_count = 5
    
    import edm.modules
    import edm.edm
    unet = edm.modules.UNet((input_channels) * (context_length + 1), 3, None, actions_count, context_length)
    return edm.edm.EDM(
        p_mean=-1.2,
        p_std=1.2,
        sigma_data=0.5,
        model=unet,
        context_length=context_length,
        device=device
    )

if __name__ == "__main__":
    context_length = 4
    batch_size = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # For Mac OS
    if torch.backends.mps.is_available():
        device = "mps"

    # model = _get_ddpm_model(context_length=context_length, device=device)
    model = _get_edm_model(context_length=context_length, device=device)
    import torchvision.transforms as transforms
    from torch.utils.data import random_split

    transform_to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5,.5,.5), (.5,.5,.5))
    ])

    dataset = SequencesDataset(
        images_dir="./training_data/snapshots",
        actions_path="./training_data/actions",
        seq_length=context_length,
        transform=transform_to_tensor
    )
    from torch.utils.data import Subset
    dataset = Subset(dataset, range(10))
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

    import edm.edm
    import numpy as np
    import matplotlib.pyplot as plt
    def _save_edm_imgs(
        model: edm.edm.EDM,
        previous_frames: torch.Tensor,
        previous_actions: torch.Tensor,
        real_imgs: torch.Tensor,
        epoch: int
    ):
        def get_np_img(tensor: torch.Tensor) -> np.ndarray:
            return (tensor * 127.5 + 127.5).long().clip(0,255).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
        new_img = model.sample(real_imgs.shape, previous_frames.unsqueeze(0), previous_actions.unsqueeze(0), num_steps=10)
        # new_img = real_imgs.clone().unsqueeze(0)

        height_row = 5
        col_width = 5
        cols = len(previous_frames) + 1
        fig, axes = plt.subplots(2, cols, figsize=(col_width * cols, height_row * 2))
        for row in range(2):
            for i in range(len(previous_frames)):
                axes[row, i].imshow(get_np_img(previous_frames[i]))
            axes[row, cols - 1].imshow(get_np_img(new_img[0]) if row == 0 else get_np_img(real_imgs))
        plt.subplots_adjust(wspace=0, hspace=0)
        
        # Save the combined figure
        plt.savefig(f"val_images/{epoch}.png", bbox_inches='tight', pad_inches=0)
        plt.close()

    _, val_losses = train(
        model=model,
        epochs=2,
        device=device,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        save_every_epoch=1,
        save_imgs=_save_edm_imgs
    )