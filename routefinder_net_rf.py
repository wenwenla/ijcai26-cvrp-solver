import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv as GN
from utils import select_node_embedding
from einops import rearrange


class RMSNorm(nn.Module):
    """From https://github.com/meta-llama/llama-models"""

    def __init__(self, dim: int, eps: float = 1e-5, **kwargs):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    

class ParallelGatedMLP(nn.Module):
    """From https://github.com/togethercomputer/stripedhyena"""

    def __init__(
        self,
        hidden_size: int = 128,
        inner_size_multiple_of: int = 256,
        mlp_activation: str = "silu",
        model_parallel_size: int = 1,
    ):
        super().__init__()

        multiple_of = inner_size_multiple_of
        self.act_type = mlp_activation
        if self.act_type == "gelu":
            self.act = F.gelu
        elif self.act_type == "silu":
            self.act = F.silu
        else:
            raise NotImplementedError

        self.multiple_of = multiple_of * model_parallel_size

        inner_size = int(2 * hidden_size * 4 / 3)
        inner_size = self.multiple_of * (
            (inner_size + self.multiple_of - 1) // self.multiple_of
        )

        self.l1 = nn.Linear(
            in_features=hidden_size,
            out_features=inner_size,
            bias=False,
        )
        self.l2 = nn.Linear(
            in_features=hidden_size,
            out_features=inner_size,
            bias=False,
        )
        self.l3 = nn.Linear(
            in_features=inner_size,
            out_features=hidden_size,
            bias=False,
        )

    def forward(self, z):
        z1, z2 = self.l1(z), self.l2(z)
        return self.l3(self.act(z1) * z2)


class TransformerBlock(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.norm_attn = RMSNorm(128)
        self.mha = nn.MultiheadAttention(128, 8, batch_first=True)
        self.norm_ffn = RMSNorm(128)
        self.ffn = ParallelGatedMLP()

    def forward(self, x):
        x_norm = self.norm_attn(x)
        h = self.mha(x_norm, x_norm, x_norm, need_weights=False)[0] + x
        h_norm = self.norm_ffn(h)
        h = h + self.ffn(h_norm)
        return h


class Encoder(nn.Module):

    """
    Input: Graphs (batch_size, n_nodes, 3)
    Output: Node embeddings (batch_size, n_nodes, 128)
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        self.city_init = nn.Linear(6, 128)  # locs: 2, demand:1, tw:3
        self.depot_init = nn.Linear(7, 128) # locs:2, dl: 1, flags: 4
        
        self.net = nn.Sequential(
            *[TransformerBlock() for _ in range(6)]
        )

    def forward(self, td):
        locs = td['locs']  # 2
        demands = td['demand'] # 1
        tw = td['time_windows'] # 3
        dl = td['distance_limit'] # 1

        node_fea = torch.cat([locs, demands, tw, dl], dim=-1)

        node_x = node_fea[..., 1:, :6]
        
        flags = td['flags'][:, None, :]
        depot_x = torch.cat([node_fea[..., 0:1, :2], node_fea[..., 0:1, -1:], flags], dim=-1)

        node_init = self.city_init(node_x)
        depot_init = self.depot_init(depot_x)

        init_embedding = torch.cat([depot_init, node_init], dim=-2)
        node_embedding = self.net(init_embedding)
        graph_embedding = node_embedding.mean(1)
        return node_embedding, graph_embedding


class Decoder(nn.Module):

    def __init__(self, ft=False, tanh_alpha=10, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ft = ft
        self.tanh_alpha = tanh_alpha
        self.linear_first = nn.Linear(128 * 3, 128)
        self.ln1 = nn.LayerNorm(128)

        self.lstm = nn.LSTM(128, 128, batch_first=True)
        self.ln2 = nn.LayerNorm(128)
        if self.ft:
            self.context_trans = nn.Sequential(
                nn.Linear(128, 8 * 128),
                nn.ReLU(),
                nn.Linear(8 * 128, 128)
            )
        self.Wq = nn.Linear(128, 128, bias=False)
        self.Wk = nn.Linear(128, 128, bias=False)
        self.Wv = nn.Linear(128, 128, bias=False)
        self.linear_proj = nn.Linear(128, 128, bias=False)
        self.num_heads = 8

        self.logitk = nn.Linear(128, 128)
        self.cache = None
        self.now_step = 0

    def build_cache(self, node_embeddings):
        batch_size = node_embeddings.shape[0]
        self.prev_h = torch.zeros(1, batch_size, 128).to(self.logitk.weight.device)
        self.prev_c = torch.zeros(1, batch_size, 128).to(self.logitk.weight.device)

        self.cache = (
            self.Wk(node_embeddings),
            self.Wv(node_embeddings),
            self.logitk(node_embeddings)
        )
        self.now_step = 0

    def forward(self, td, node_embeddings, graph_embeddings):
        self.now_step += 1
        # if self.now_step % 100 == 0:
        #     # assert False
        #     self.prev_c = self.prev_c.detach()
        #     self.prev_h = self.prev_h.detach()

        first_node_emb = select_node_embedding(node_embeddings, td['first_node'])
        cur_node_emb = select_node_embedding(node_embeddings, td['current_node'])
        
        context = torch.cat([graph_embeddings, first_node_emb, cur_node_emb], dim=1)
        context = self.linear_first(context)
        context = self.ln1(context)

        out, (self.prev_h, self.prev_c) = self.lstm(context.unsqueeze(1), (self.prev_h, self.prev_c))
        context_out = out[:, -1, :]
        context_out = self.ln2(context_out)
        # if self.ft:
        #     context_out = context_out + self.context_trans(context_out)
        context_q = self.Wq(context_out)
        context_k = self.cache[0]
        context_v = self.cache[1]

        # make head
        q = rearrange(context_q, 'b (1 e h) -> b h 1 e', h=8)
        k = rearrange(context_k, 'b n (e h) -> b h n e', h=8)
        v = rearrange(context_v, 'b n (e h) -> b h n e', h=8)
        mask = rearrange(td['mask'], 'b n -> b 1 1 n')
        att_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)  # need mask, yes
        # recover
        att_out = rearrange(att_out, 'b h l e -> b l (h e)')
        context_emb = self.linear_proj(att_out)  # batch_size, 1, 128

        logitK = rearrange(self.cache[2], 'b n e -> b e n')
        compatibility = torch.bmm(context_emb, logitK) / math.sqrt(128) # batch_size, 1, n
        compatibility = compatibility.squeeze(1)  # batch_size, n

        # mask
        compatibility = self.tanh_alpha * torch.tanh(compatibility)
        compatibility[td['mask'] == 1] = -torch.inf

        return compatibility