import torch
from edm.edm import EDM
import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import edm.modules as modules
from data import SequencesDataset
from train import train
import torchvision.transforms as transforms
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import tqdm

input_channels = 3
context_length = 4
actions_count = 5
batch_size = 1
num_workers = 2
device = "cuda" if torch.cuda.is_available() else "cpu"
# For Mac OS
if torch.backends.mps.is_available():
    device = "mps"
ROOT_PATH = "./"
def local_path(path):
    return os.path.join(ROOT_PATH, path)
MODEL_PATH = local_path("model_19_edm.pth")

edm = EDM(
    p_mean=-1.2,
    p_std=1.2,
    sigma_data=0.5,
    model=modules.UNet((input_channels) * (context_length + 1), 3, None, actions_count, context_length),
    context_length=context_length,
    device=device
)
edm.load_state_dict(torch.load(MODEL_PATH, map_location=device)["model"])

transform_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((.5,.5,.5), (.5,.5,.5))
])

dataset = SequencesDataset(
    images_dir=local_path("snapshots"),
    actions_path=local_path("actions"),
    seq_length=context_length,
    transform=transform_to_tensor
)

length = len(dataset)
length_session = 80
count = 1
index = random.randint(0, length - 1)
img, last_imgs, actions = dataset[index]

img = img.to(device)
last_imgs = last_imgs.to(device)
actions = actions.to(device)
gen_imgs = last_imgs.clone()
import datetime
start = datetime.datetime.now()
edm.sample(img.shape, gen_imgs[-context_length:].unsqueeze(0), actions.unsqueeze(0), num_steps=10)[0][:, 2:-2, 2:-2]
print(f"delay is {(datetime.datetime.now() - start).total_seconds()}")