import torch
import torch.nn as nn
import math
import einops

class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.ln = nn.Sequential(
            nn.ReLU(),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x):
        return self.ln(x)
    
class PositionalEmbedding(nn.Module):
    def __init__(self, T: int, output_dim: int) -> None:
        super().__init__()
        self.output_dim = output_dim
        position = torch.arange(T+1).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, output_dim, 2) * (-math.log(10000.0) / output_dim))
        pe = torch.zeros(T+1, output_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        return self.pe[x].reshape(x.shape[0], self.output_dim)
    
class MultiheadAttention(nn.Module):
    def __init__(self, n_heads: int, emb_dim: int, input_dim: int) -> None:
        super().__init__()
        assert emb_dim % n_heads == 0
        head_dim = emb_dim // n_heads
        self.K_W = nn.Parameter(torch.rand(n_heads, input_dim, head_dim))
        self.K_b = nn.Parameter(torch.rand(n_heads, head_dim))
        self.Q_W = nn.Parameter(torch.rand(n_heads, input_dim, head_dim))
        self.Q_b = nn.Parameter(torch.rand(n_heads, head_dim))
        self.V_W = nn.Parameter(torch.rand(n_heads, input_dim, head_dim))
        self.V_b = nn.Parameter(torch.rand(n_heads, head_dim))
        self.O_W = nn.Parameter(torch.rand(n_heads, head_dim, input_dim))
        self.O_b = nn.Parameter(torch.rand(input_dim))
        self.norm = nn.LayerNorm([input_dim])
        self.mlp = nn.Sequential(
            nn.LayerNorm([input_dim]),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim)
        )

    def forward(self, x, t, actions):
        _, input_dim, h, w = x.shape
        x = einops.rearrange(x, "b c h w -> b (h w) c")
        res = x
        res = self.norm(x)
        k = einops.einsum(res, self.K_W, "b size ch, n_h ch h_dim -> b size n_h h_dim")
        k = k + self.K_b
        q = einops.einsum(res, self.Q_W, "b size ch, n_h ch h_dim -> b size n_h h_dim")
        q = q + self.Q_b
        qk = einops.einsum(q, k, "batch s1 n_h h_dim, batch s2 n_h h_dim -> batch n_h s1 s2")
        qk = qk / torch.sqrt(torch.tensor(input_dim, dtype=torch.float))
        qk = qk.softmax(-1)
        v = einops.einsum(res, self.V_W, "b size ch, n_h ch h_dim -> b size n_h h_dim")
        v = v + self.V_b
        res = einops.einsum(qk, v, "batch n_h size size, batch size n_h h_dim -> batch size n_h h_dim")
        res = einops.einsum(res, self.O_W, "batch size n_h h_dim, n_h h_dim ch -> batch size ch")
        res = res + self.O_b
        res = res + x
        res = self.mlp(res) + res
        res = einops.rearrange(res, "b (h w) c -> b c h w", h=h, w=w)
        return res

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, is_residual=False, is_debug=False):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
        )
        self.conv_2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        )
        self.actions_emb = MLP(input_dim=time_emb_dim, output_dim=out_channels)
        self.time_emb = MLP(input_dim=time_emb_dim, output_dim=out_channels)
        if in_channels != out_channels:
            self.conv_3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        else:
            self.conv_3 = nn.Identity()
        self.is_debug = is_debug
        self.is_residual = is_residual

    def forward(self, x, t, prev_actions):
        h = self.conv_1(x)
        if t is None:
            return self.conv_3(x) + self.conv_2(h)
        t = self.time_emb(t)
        batch_size, emb_dim = t.shape
        t = t.view(batch_size, emb_dim, 1, 1)
        prev_actions = self.actions_emb(prev_actions)
        prev_actions = prev_actions.view(batch_size, emb_dim, 1, 1)
        conv_res = self.conv_2(h + t + prev_actions)
        if self.is_residual:
            return self.conv_3(x) + conv_res
        else:
            return self.conv_2(h + t)        

class DownBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2)
    
    def forward(self, x, t, actions):
        return self.pool(x)
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upscale = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)
    
    def forward(self, x, t, actions):
        return self.upscale(x)

class SequenceWithTimeEmbedding(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.models = nn.ModuleList(blocks)
    
    def forward(self, x, t, actions):
        for model in self.models:
            x = model(x, t, actions)
        return x

class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        T: int,
        actions_count: int,
        seq_length: int,
        steps=(1, 2, 4),
        hid_size = 128,
        attn_step_indexes = [1],
        has_residuals=True,
        num_resolution_blocks=2,
        is_debug = False
    ):
        super().__init__()

        time_emb_dim = hid_size * 4
        self.time_embedding = nn.Sequential(
            PositionalEmbedding(T=T, output_dim=hid_size),
            nn.Linear(hid_size, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.actions_embedding = nn.Sequential(
            nn.Embedding(actions_count, hid_size),
            nn.Flatten(),
            nn.Linear(hid_size * seq_length, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.first_conv = nn.Conv2d(in_channels, steps[0] * hid_size, 3, padding=1)
        self.down_blocks = nn.ModuleList()
        prev_hid_size = steps[0] * hid_size
        for (index, step) in enumerate(steps):
            res_blocks = []
            for block in range(num_resolution_blocks):
                res_blocks.append(
                    ResnetBlock(
                        in_channels=prev_hid_size if block == 0 else step * hid_size,
                        out_channels=step * hid_size,
                        time_emb_dim=time_emb_dim,
                        is_residual=has_residuals
                    )
                )
                if index in attn_step_indexes:
                    res_blocks.append(
                        MultiheadAttention(
                            n_heads=4,
                            emb_dim=step * hid_size,
                            input_dim=step * hid_size
                        )
                    )
            self.down_blocks.append(
                SequenceWithTimeEmbedding(res_blocks)
            )
            if index != len(steps) - 1:
                self.down_blocks.append(DownBlock())
            prev_hid_size = step * hid_size
        if len(attn_step_indexes) > 0:
            self.backbone = SequenceWithTimeEmbedding([
                ResnetBlock(steps[-1] * hid_size, steps[-1] * hid_size, time_emb_dim=time_emb_dim),
                MultiheadAttention(n_heads=4, emb_dim=steps[-1] * hid_size, input_dim=steps[-1] * hid_size),
                ResnetBlock(steps[-1] * hid_size, steps[-1] * hid_size, time_emb_dim=time_emb_dim),
            ])
        else:
            self.backbone = SequenceWithTimeEmbedding([
                ResnetBlock(steps[-1] * hid_size, steps[-1] * hid_size, time_emb_dim=time_emb_dim),
                ResnetBlock(steps[-1] * hid_size, steps[-1] * hid_size, time_emb_dim=time_emb_dim),
            ])

        self.up_blocks = nn.ModuleList()
        reverse_steps = list(reversed(steps))
        for (index, step) in enumerate(reverse_steps):
            res_blocks = []
            for block in range(num_resolution_blocks):
                next_hid_size = reverse_steps[index + 1] * hid_size if index != len(steps) - 1 else step * hid_size
                res_blocks.append(
                    ResnetBlock(
                        in_channels=prev_hid_size * 2 if block == 0 else next_hid_size,
                        out_channels=next_hid_size,
                        time_emb_dim=time_emb_dim,
                        is_residual=has_residuals
                    )
                )
                if len(reverse_steps) - index - 1 in attn_step_indexes:
                    res_blocks.append(
                        MultiheadAttention(
                            n_heads=4,
                            emb_dim=next_hid_size,
                            input_dim=next_hid_size
                        )
                    )
            self.up_blocks.append(
                SequenceWithTimeEmbedding(res_blocks)
            )
            if index != len(steps) - 1:
                self.up_blocks.append(UpBlock(next_hid_size, next_hid_size))
            prev_hid_size = next_hid_size

        self.is_debug = is_debug
        self.out = nn.Sequential(*[
            nn.GroupNorm(8, steps[0] * hid_size),
            nn.ReLU(),
            nn.Conv2d(in_channels=steps[0] * hid_size, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        ])

    def forward(self, x: torch.Tensor, t: torch.Tensor, prev_actions: torch.Tensor):
        assert x.shape[0] == prev_actions.shape[0]
        time_emb = self.time_embedding(t)
        actions_emb = self.actions_embedding(prev_actions)

        x = self.first_conv(x)
        hx = []
        for down_block in self.down_blocks:
            x = down_block(x, time_emb, actions_emb)
            if not isinstance(down_block, DownBlock):
                hx.append(x)
        x = self.backbone(x, time_emb, actions_emb)

        ind = len(hx) - 1
        for up_block in self.up_blocks:
            if not isinstance(up_block, UpBlock):
                x = up_block(torch.cat([x, hx[ind]], 1), time_emb, actions_emb)
                ind -= 1
            else:
                x = up_block(x, time_emb, actions_emb)
        x = self.out(x)

        return x
    
if __name__ == "__main__":
    size = (64, 64)
    input_channels = 3
    context_length = 4
    actions_count = 5
    T = 5
    batch_size = 3
    unet = UNet(
        (input_channels) * (context_length + 1),
        3,
        T,
        actions_count,
        context_length,
        steps=(1,2,3)
    )
    img = torch.randn((batch_size, input_channels, *size))
    prev_frames = torch.randn((batch_size, input_channels, context_length, *size))
    frames = torch.concat([img[:, :, None, :, :], prev_frames], dim=2).flatten(1,2)

    prev_actions = torch.randint(low=0, high=actions_count, size=(batch_size, context_length))
    t = torch.randint(1, T + 1, (batch_size,))
    unet.forward(frames, t.unsqueeze(1), prev_actions)
