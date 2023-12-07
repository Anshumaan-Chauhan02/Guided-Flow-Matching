#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries

# In[ ]:


import torch
import torch.nn as nn
import matplotlib.pyplot as plot
import math


# ### Upsampling/Downsampling 

# In[ ]:


def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)


def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)


# ### Encodings

# In[ ]:


class TimePositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        if len(time.size()) == 0:
            time = torch.tensor([time.item()]).to("cuda")
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConditionPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.pe = torch.zeros(1, dim).to("cuda")

        i = torch.arange(dim // 2)
        # i = i.reshape(1, -1).t()

        p = -torch.arange(0, dim, 2) / dim
        exp = torch.pow(1e4, p)

        self.pe[0, 0::2] = torch.sin(i * exp)
        self.pe[0, 1::2] = torch.cos(i * exp)

    def forward(self, text):
        N, D = text.shape

        text = nn.functional.normalize(text, dim=1)

        output = torch.empty(N, D).to("cuda")

        pos_emb = self.pe
        output = text + pos_emb

        return output


# ### ConvOp Block

# In[ ]:


class ConvOp(nn.Module):
    def __init__(self, h_x, w_x, in_channels, out_channels, emb_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.GELU(), nn.Linear(emb_dim, in_channels))
        self.cross_attn_block = Cross_Norm(h_x, w_x, in_channels)
        self.init_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=7,
            padding=3,
            groups=in_channels,
        )
        self.conv_block = nn.Sequential(
            nn.GroupNorm(1, in_channels),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels * 2,
                kernel_size=3,
                padding=(1, 1),
            ),
            nn.GELU(),
            nn.GroupNorm(1, out_channels * 2),
            nn.Conv2d(
                in_channels=out_channels * 2,
                out_channels=out_channels,
                kernel_size=3,
                padding=(1, 1),
            ),
        )

        self.res_conv = (
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t_emb, c_emb):
        h = self.init_conv(x)

        t_e = self.time_mlp(t_emb)
        t_e = t_e[(...,) + (None,) * 2]

        if c_emb is not None:
            h = self.cross_attn_block(h, c_emb)
            h = h + t_e
        else:
            h = h + t_e

        h = self.conv_block(h)

        return h + self.res_conv(x)


# ### Residual Blocks

# In[ ]:


class Residual_Norm(nn.Module):
    def __init__(self, input_dim, linear_attn=True):
        super().__init__()
        self.norm = nn.GroupNorm(1, input_dim)
        self.attn = Attention(input_dim, linear_attn=linear_attn)
        # self.attn = Attention(input_dim, linear_attn=linear_attn)

    def forward(self, x):
        input = x
        x = self.norm(x)
        x = self.attn(x)
        return x + input


class Cross_Norm(nn.Module):
    def __init__(self, h_x, w_x, input_dim, embed_dim=256, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(1, input_dim)
        self.attn = CrossAttention(h_x, w_x, input_dim, embed_dim=256, num_heads=4)
        # self.attn = Attention(input_dim, linear_attn=linear_attn)

    def forward(self, x, c_emb):
        input = x
        x = self.norm(x)
        x = self.attn(x, c_emb)
        return x + input


# ### Self Attention

# In[ ]:


class Attention(nn.Module):
    def __init__(self, input_dim, embed_dim=256, num_heads=4, linear_attn=True):
        super().__init__()
        self.n_head = num_heads
        self.emb_dim = embed_dim
        self.linear_attn = linear_attn
        self.hidden_dim = self.emb_dim * self.n_head
        self.key = nn.Conv2d(input_dim, self.hidden_dim, kernel_size=3, padding=(1, 1))
        self.query = nn.Conv2d(
            input_dim, self.hidden_dim, kernel_size=3, padding=(1, 1)
        )
        self.value = nn.Conv2d(
            input_dim, self.hidden_dim, kernel_size=3, padding=(1, 1)
        )
        if linear_attn:
            self.proj = nn.Sequential(
                nn.Conv2d(self.hidden_dim, input_dim, kernel_size=3, padding=(1, 1)),
                nn.GroupNorm(1, input_dim),
            )
        else:
            self.proj = nn.Conv2d(
                self.hidden_dim, input_dim, kernel_size=3, padding=(1, 1)
            )

    def forward(self, x):
        N, C, H, W = x.shape
        output = torch.empty((N, C, H, W)).to("cuda")

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = torch.reshape(Q, (N, self.n_head, self.emb_dim, H * W))
        K = torch.reshape(K, (N, self.n_head, self.emb_dim, H * W))
        V = torch.reshape(V, (N, self.n_head, self.emb_dim, H * W))

        new_K = torch.transpose(torch.transpose(K, 1, 2), 2, 3)
        new_Q = torch.transpose(Q, 1, 2)
        new_V = torch.transpose(V, 1, 2)

        # softmax = torch.nn.Softmax(dim=-1)

        if not self.linear_attn:
            dot_product = (torch.matmul(new_Q, new_K)) / (
                (self.emb_dim / self.n_head) ** (0.5)
            )
            attention_scores = dot_product.softmax(-1)
            Y = torch.matmul(attention_scores, new_V)
        else:  # linear_attn
            Q_sm = new_Q.softmax(-2)
            K_sm = new_K.softmax(-1)

            KV = torch.matmul(new_V, K_sm)

            Y = torch.matmul(KV, Q_sm)

        new_Y = torch.cat([Y[:, :, h, :] for h in range(self.n_head)], dim=1)
        new_Y = torch.reshape(new_Y, (N, self.hidden_dim, H, W))
        output = self.proj(new_Y)

        return output


# ### Cross Attention

# In[ ]:


class CrossAttention(nn.Module):
    def __init__(self, h_x, w_x, input_dim, embed_dim=256, num_heads=4):
        super().__init__()
        self.n_head = num_heads
        self.emb_dim = embed_dim
        self.hidden_dim = self.emb_dim * self.n_head
        self.key = nn.Conv2d(input_dim, self.hidden_dim, kernel_size=3, padding=(1, 1))
        self.query = nn.Conv2d(
            input_dim, self.hidden_dim, kernel_size=3, padding=(1, 1)
        )
        self.value = nn.Conv2d(
            input_dim, self.hidden_dim, kernel_size=3, padding=(1, 1)
        )
        self.proj = nn.Conv2d(self.hidden_dim, input_dim, kernel_size=3, padding=(1, 1))
        self.linear1 = nn.Linear(256, int(input_dim * h_x * w_x))

    def forward(self, x, c_emb):
        N, C, H, W = x.shape
        output = torch.empty((N, C, H, W)).to("cuda")

        c_emb = self.linear1(c_emb)
        c_emb = torch.nn.functional.normalize(c_emb, dim=-1)

        c_emb = torch.reshape(c_emb, (N, C, H, W))
        Q = self.query(x)
        K = self.key(c_emb)
        V = self.value(c_emb)

        Q = torch.reshape(Q, (N, self.n_head, self.emb_dim, H * W))
        K = torch.reshape(K, (N, self.n_head, self.emb_dim, H * W))
        V = torch.reshape(V, (N, self.n_head, self.emb_dim, H * W))

        new_K = torch.transpose(torch.transpose(K, 1, 2), 2, 3)
        new_Q = torch.transpose(Q, 1, 2)
        new_V = torch.transpose(V, 1, 2)

        softmax = torch.nn.Softmax(dim=-1)

        dot_product = (torch.matmul(new_Q, new_K)) / (
            (self.emb_dim / self.n_head) ** (0.5)
        )
        attention_scores = softmax(dot_product)

        Y = torch.matmul(attention_scores, new_V)

        new_Y = torch.cat([Y[:, :, h, :] for h in range(self.n_head)], dim=1)
        new_Y = torch.reshape(new_Y, (N, self.hidden_dim, H, W))
        output = self.proj(new_Y)

        return output


# ### UNet Model

# In[ ]:


class UNet(nn.Module):
    def __init__(self, conditional_gen=False):
        super().__init__()

        im_dim = 32
        self.img_dim = im_dim
        text_dim = 256
        self.text_emb_dim = text_dim
        dim_mult = [1, 2, 4, 8]
        self.time_emb = nn.Sequential(
            TimePositionEmbeddings(im_dim),
            nn.Linear(im_dim, im_dim),
            nn.GELU(),
            nn.Linear(im_dim, im_dim),
        )

        # self.cond_emb = nn.Sequential(
        #     ConditionPositionEmbeddings(text_dim),
        #     nn.Linear(text_dim, im_dim),
        #     nn.ReLU()
        # )

        init_dim = 32 // 3 * 2
        self.in_conv = nn.Conv2d(3, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: im_dim * m, dim_mult)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.downs = nn.ModuleList([])
        idx = 0
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ConvOp(
                            32 / dim_mult[idx],
                            32 / dim_mult[idx],
                            dim_in,
                            dim_out,
                            emb_dim=im_dim,
                        ),
                        ConvOp(
                            32 / dim_mult[idx],
                            32 / dim_mult[idx],
                            dim_out,
                            dim_out,
                            emb_dim=im_dim,
                        ),
                        Residual_Norm(dim_out),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )
            idx += 1

        bridge_dim = dims[-1]
        self.bridge_block1 = ConvOp(
            32 / dim_mult[-1], 32 / dim_mult[-1], bridge_dim, bridge_dim, im_dim
        )
        self.bridge_attn = Residual_Norm(dim_out, linear_attn=False)
        self.bridge_block2 = ConvOp(
            32 / dim_mult[-1], 32 / dim_mult[-1], bridge_dim, bridge_dim, im_dim
        )

        self.ups = nn.ModuleList([])
        idx -= 1
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        ConvOp(
                            32 / dim_mult[idx],
                            32 / dim_mult[idx],
                            dim_out * 2,
                            dim_in,
                            emb_dim=self.img_dim,
                        ),
                        ConvOp(
                            32 / dim_mult[idx],
                            32 / dim_mult[idx],
                            dim_in,
                            dim_in,
                            emb_dim=self.img_dim,
                        ),
                        Residual_Norm(dim_in),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )
            idx -= 1
        self.final_ConvOp = ConvOp(32, 32, im_dim, im_dim, emb_dim=im_dim)
        self.final_conv2d = nn.Conv2d(im_dim, 3, 1)

    def forward(self, t, states):
        if len(states) == 2:
            x = states[0]
            c = states[1]
            c_emb = c
        else:
            x = states
            c = None
            c_emb = None

        t_emb = self.time_emb(t)

        skip_conn_vals = []

        x = self.in_conv(x)

        # down
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t_emb, c_emb)
            x = block2(x, t_emb, c_emb)
            x = attn(x)
            skip_conn_vals.append(x)
            x = downsample(x)

        # bridge
        x = self.bridge_block1(x, t_emb, c_emb)
        x = self.bridge_attn(x)
        x = self.bridge_block2(x, t_emb, c_emb)

        # up
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, skip_conn_vals.pop()), dim=1)
            x = block1(x, t_emb, c_emb)
            x = block2(x, t_emb, c_emb)
            x = attn(x)
            x = upsample(x)

        x = self.final_ConvOp(x, t_emb, c_emb)
        if c is not None:
            return self.final_conv2d(x), c
        else:
            return self.final_conv2d(x)

