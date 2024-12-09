from typing import Tuple

import torch
import torch.nn as nn
from modules import PositionalEmbedding, ResnetBlock, ResnetsBlock, DownBlock, UpBlock

class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        T: int,
        actions_count: int,
        seq_length: int,
        steps=(2, 2, 2, 2),
        channels = (64, 64, 64, 64),
        cond_channels = 256,
        attn_step_indexes = [False, False, False, False],
        is_debug = False
    ):
        super().__init__()
        assert len(steps) == len(channels) == len(attn_step_indexes)
        self.time_embedding = PositionalEmbedding(T=T, output_dim=cond_channels)
        self.actions_embedding = nn.Sequential(
            nn.Embedding(actions_count, cond_channels // seq_length),
            nn.Flatten()
        )
        self.cond_embedding = nn.Sequential(
            nn.Linear(cond_channels, cond_channels),
            nn.ReLU(),
            nn.Linear(cond_channels, cond_channels),
        )

        self.first_conv = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)
        down_res_blocks = []
        self.downsample_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        up_res_blocks = []
        for (index, step) in enumerate(steps):
            in_ch = step * [channels[index]]
            out_ch = in_ch.copy()
            out_ch[-1] = channels[index + 1] if index < len(steps) - 1 else channels[index]
            down_res_blocks.append(ResnetsBlock(
                in_channels_list=in_ch,
                out_channels_list=out_ch,
                cond_channels=cond_channels,
                has_attn=index == attn_step_indexes[index]
            ))
            self.downsample_blocks.append(DownBlock())
            self.upsample_blocks.append(UpBlock(out_ch[-1]))
            in_ch = step * [channels[index]]
            out_ch = in_ch.copy()
            in_ch[0] = 2 * (channels[index + 1] if index < len(steps) - 1 else channels[index])
            up_res_blocks.append(ResnetsBlock(
                in_channels_list=in_ch,
                out_channels_list=out_ch,
                cond_channels=cond_channels,
                has_attn=index == attn_step_indexes[index]
            ))
        self.downres_blocks = nn.ModuleList(down_res_blocks)
        self.upres_blocks = nn.ModuleList(reversed(up_res_blocks))
        self.backbone = ResnetsBlock(
            [channels[-1]] * 2,
            [channels[-1]] * 2,
            cond_channels=cond_channels,
            has_attn=True
        )
        self.is_debug = is_debug
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=channels[0], out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, prev_actions: torch.Tensor):
        assert x.shape[0] == prev_actions.shape[0]
        time_emb = self.time_embedding(t)
        actions_emb = self.actions_embedding(prev_actions)
        cond = self.cond_embedding(time_emb + actions_emb)

        x = self.first_conv(x)
        hx = []
        for index, downres_block in enumerate(self.downres_blocks):
            x = downres_block(x, cond)
            hx.append(x)
            x = self.downsample_blocks[index](x)                
        x = self.backbone(x, cond)

        for index, up_block in enumerate(self.upres_blocks):
            x = self.upsample_blocks[len(self.upres_blocks) - index - 1](x)
            x = up_block(torch.cat([x, hx[len(self.upres_blocks) - index - 1]], 1), cond)
        x = self.out(x)

        return x
    
if __name__ == "__main__":
    size = (60, 60)
    input_channels = 3
    context_length = 4
    actions_count = 5
    T = 1000
    batch_size = 2
    unet = UNet(
        (input_channels) * (context_length + 1),
        3,
        T,
        actions_count,
        context_length
    )
    img = torch.randn((batch_size, input_channels, *size))
    prev_frames = torch.randn((batch_size, input_channels, context_length, *size))
    frames = torch.concat([img[:, :, None, :, :], prev_frames], dim=2).flatten(1,2)

    prev_actions = torch.randint(low=0, high=actions_count, size=(batch_size, context_length))
    t = torch.randint(1, T + 1, (batch_size,))
    unet.forward(frames, t.unsqueeze(1), prev_actions)
