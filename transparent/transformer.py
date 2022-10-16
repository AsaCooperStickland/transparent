"""# Defining Transformer"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import einops
from numpy import ndarray
from typing import Dict, List

# A helper class to get access to intermediate activations (inspired by Garcon)
# It's a dummy module that is the identity function by default
# I can wrap any intermediate activation in a HookPoint and get a convenient
# way to add PyTorch hooks


class HookPoint(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fwd_hooks = []
        self.bwd_hooks = []

    def give_name(self, name: str) -> None:
        # Called by the model at initialisation
        self.name = name

    def add_hook(self, hook, dir='fwd'):
        # Hook format is fn(activation, hook_name)
        # Change it into PyTorch hook format (this includes input and output,
        # which are the same for a HookPoint)
        def full_hook(module, module_input, module_output):
            return hook(module_output, name=self.name)
        if dir == 'fwd':
            handle = self.register_forward_hook(full_hook)
            self.fwd_hooks.append(handle)
        elif dir == 'bwd':
            handle = self.register_full_backward_hook(full_hook)
            self.bwd_hooks.append(handle)
        else:
            raise ValueError(f"Invalid direction {dir}")

    def remove_hooks(self, dir='fwd'):
        if (dir == 'fwd') or (dir == 'both'):
            for hook in self.fwd_hooks:
                hook.remove()
            self.fwd_hooks = []
        if (dir == 'bwd') or (dir == 'both'):
            for hook in self.bwd_hooks:
                hook.remove()
            self.bwd_hooks = []
        if dir not in ['fwd', 'bwd', 'both']:
            raise ValueError(f"Invalid direction {dir}")

    def forward(self, x):
        return x

# Embed & Unembed


class Embed(nn.Module):
    def __init__(self, d_vocab: int, d_model: int) -> None:
        super().__init__()
        self.W_E = nn.Parameter(
            torch.randn(
                d_model,
                d_vocab) /
            np.sqrt(d_model))

    def forward(self, x):
        return torch.einsum('dbp -> bpd', self.W_E[:, x])


class Unembed(nn.Module):
    def __init__(self, d_vocab: int, d_model: int) -> None:
        super().__init__()
        self.W_U = nn.Parameter(
            torch.randn(
                d_model,
                d_vocab) /
            np.sqrt(d_vocab))

    def forward(self, x):
        return (x @ self.W_U)

# Positional Embeddings


class PosEmbed(nn.Module):
    def __init__(self, max_ctx: int, d_model: int) -> None:
        super().__init__()
        self.W_pos = nn.Parameter(torch.randn(
            max_ctx, d_model) / np.sqrt(d_model))

    def forward(self, x):
        return x + self.W_pos[:x.shape[-2]]

# LayerNorm


class LayerNorm(nn.Module):
    def __init__(self, d_model, epsilon=1e-4, model=[None]):
        super().__init__()
        self.model = model
        self.w_ln = nn.Parameter(torch.ones(d_model))
        self.b_ln = nn.Parameter(torch.zeros(d_model))
        self.epsilon = epsilon

    def forward(self, x):
        if self.model[0].use_ln:
            x = x - x.mean(axis=-1)[..., None]
            x = x / (x.std(axis=-1)[..., None] + self.epsilon)
            x = x * self.w_ln
            x = x + self.b_ln
            return x
        else:
            return x

# Attention


class Attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_head: int, n_ctx: int, model) -> None:
        super().__init__()
        self.model = model
        self.W_K = nn.Parameter(torch.randn(
            num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_Q = nn.Parameter(torch.randn(
            num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_V = nn.Parameter(torch.randn(
            num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_O = nn.Parameter(torch.randn(
            d_model, d_head * num_heads) / np.sqrt(d_model))
        self.register_buffer('mask', torch.tril(torch.ones((n_ctx, n_ctx))))
        self.d_head = d_head
        self.hook_k = HookPoint()
        self.hook_q = HookPoint()
        self.hook_v = HookPoint()
        self.hook_z = HookPoint()
        self.hook_attn = HookPoint()
        self.hook_attn_pre = HookPoint()

    def forward(self, x):
        k = self.hook_k(torch.einsum('ihd,bpd->biph', self.W_K, x))
        q = self.hook_q(torch.einsum('ihd,bpd->biph', self.W_Q, x))
        v = self.hook_v(torch.einsum('ihd,bpd->biph', self.W_V, x))
        attn_scores_pre = torch.einsum('biph,biqh->biqp', k, q)
        attn_scores_masked = torch.tril(
            attn_scores_pre) - 1e10 * (1 - self.mask[:x.shape[-2], :x.shape[-2]])
        attn_matrix = self.hook_attn(F.softmax(self.hook_attn_pre(
            attn_scores_masked / np.sqrt(self.d_head)), dim=-1))
        z = self.hook_z(torch.einsum('biph,biqp->biqh', v, attn_matrix))
        z_flat = einops.rearrange(z, 'b i q h -> b q (i h)')
        out = torch.einsum('df,bqf->bqd', self.W_O, z_flat)
        return out

# MLP Layers


class MLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int, act_type: str, model) -> None:
        super().__init__()
        self.model = model
        self.W_in = nn.Parameter(
            torch.randn(
                d_mlp,
                d_model) /
            np.sqrt(d_model))
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(
            torch.randn(
                d_model,
                d_mlp) /
            np.sqrt(d_model))
        self.b_out = nn.Parameter(torch.zeros(d_model))
        self.act_type = act_type
        # self.ln = LayerNorm(d_mlp, model=self.model)
        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()
        assert act_type in ['ReLU', 'GeLU', 'SoLU']
        if act_type == 'SoLU':
            #self.use_ln = True
            self.ln = nn.LayerNorm(d_mlp)
            self.hook_solu = HookPoint()

    def forward(self, x):
        x = self.hook_pre(torch.einsum(
            'md,bpd->bpm', self.W_in, x) + self.b_in)
        if self.act_type == 'ReLU':
            x = F.relu(x)
        elif self.act_type == 'GeLU':
            x = F.gelu(x)
        elif self.act_type == 'SoLU':
            x = x * F.softmax(x, dim=-1)
            x = self.hook_solu(x)
            x = self.ln(x)
        x = self.hook_post(x)
        x = torch.einsum('dm,bpm->bpd', self.W_out, x) + self.b_out
        return x

# Transformer Block


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, d_mlp: int, d_head: int,
                 num_heads: int, n_ctx: int, act_type: str, model) -> None:
        super().__init__()
        self.model = model
        # self.ln1 = LayerNorm(d_model, model=self.model)
        self.attn = Attention(d_model, num_heads, d_head,
                              n_ctx, model=self.model)
        # self.ln2 = LayerNorm(d_model, model=self.model)
        self.mlp = MLP(d_model, d_mlp, act_type, model=self.model)
        self.hook_attn_out = HookPoint()
        self.hook_mlp_out = HookPoint()
        self.hook_resid_pre = HookPoint()
        self.hook_resid_mid = HookPoint()
        self.hook_resid_post = HookPoint()

    def forward(self, x):
        x = self.hook_resid_mid(
            x + self.hook_attn_out(self.attn((self.hook_resid_pre(x)))))
        x = self.hook_resid_post(x + self.hook_mlp_out(self.mlp((x))))
        return x
# Full transformer


class Transformer(nn.Module):
    def __init__(self, num_layers: int, d_vocab: int, d_model: int, d_mlp: int, d_head: int,
                 num_heads: int, n_ctx: int, act_type: str, use_cache: bool=False, use_ln: bool=True) -> None:
        super().__init__()
        self.cache = {}
        self.use_cache = use_cache

        self.embed = Embed(d_vocab, d_model)
        self.pos_embed = PosEmbed(n_ctx, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(
            d_model, d_mlp, d_head, num_heads, n_ctx, act_type, model=[self]) for i in range(num_layers)])
        # self.ln = LayerNorm(d_model, model=[self])
        self.unembed = Unembed(d_vocab, d_model)
        self.use_ln = use_ln
        self.d_model = d_model
        self.d_head = d_head
        self.d_mlp = d_mlp
        self.num_heads = num_heads
        self.act_type = act_type

        for name, module in self.named_modules():
            if isinstance(module, HookPoint):
                module.give_name(name)

    def forward(self, x):
        x = self.embed(x)
        x = self.pos_embed(x)
        for block in self.blocks:
            x = block(x)
        # x = self.ln(x)
        x = self.unembed(x)
        return x

    def set_use_cache(self, use_cache):
        self.use_cache = use_cache

    def hook_points(self):
        return [module for name, module in self.named_modules()
                if 'hook' in name]

    def get_entropy(self):
        names = [hook.name for hook in self.hook_points() if hook.name.endswith('hook_attn')]
        attention_list = [self.cache[name] for name in names]
        entropy_list = [torch.special.entr(attention)
                        for attention in attention_list]
        entropy_list = [entropy.mean(-1).mean(-1).mean(-1).mean(-1)
                        for entropy in entropy_list]
        return entropy_list

    def get_sparsity(self):
        if self.act_type == "SoLU":
            names = [hook.name for hook in self.hook_points() if hook.name.endswith('hook_solu')]
            activation_list = [self.cache[name] for name in names]
            activation_list = [torch.count_nonzero(F.relu(activation - 1e-5), dim=-1) for activation in activation_list]
        elif self.act_type == "GeLU":
            names = [hook.name for hook in self.hook_points() if hook.name.endswith('hook_post')]
            activation_list = [self.cache[name] for name in names]
            activation_list = [torch.count_nonzero(F.relu(activation - 1e-5), dim=-1) for activation in activation_list]
        else:
            names = [hook.name for hook in self.hook_points() if hook.name.endswith('hook_post')]
            activation_list = [self.cache[name] for name in names]
            activation_list = [torch.count_nonzero(activation, dim=-1) for activation in activation_list]
        
        activation_list = [1.0 - activation.float().mean(-1).mean(-1) /
                           self.d_mlp for activation in activation_list]
        return activation_list

    def get_activation_l1_norm(self):
        if self.act_type == "SoLU":
            raise NotImplementedError
        else:
            names = [hook.name for hook in self.hook_points() if hook.name.endswith('hook_post')]
            activations = torch.stack([self.cache[name] for name in names])
            activations = torch.norm(activations, p=1, dim=-1)
        return activations.mean(-1).mean(-1).mean(-1)

    def remove_all_hooks(self):
        for hp in self.hook_points():
            hp.remove_hooks('fwd')
            hp.remove_hooks('bwd')

    def cache_all(self, cache, incl_bwd=False, l1_norm=0.0):
        # Caches all activations wrapped in a HookPoint
        def save_hook(tensor, name):
            if l1_norm > 0.0 and name.endswith('hook_post'):
                cache[name] = tensor
            else:
                cache[name] = tensor.detach()

        def save_hook_back(tensor, name):
            cache[name + '_grad'] = tensor[0].detach()
        
        for hp in self.hook_points():
            hp.add_hook(save_hook, 'fwd')
            if incl_bwd:
                hp.add_hook(save_hook_back, 'bwd')

    def get_permutation_dict(self) -> Dict[str, ndarray]:
        params_store = {}
        for name, p in self.named_parameters():
            stored_p = p.detach().cpu().numpy()
            if any([x in name for x in ['K', 'Q', 'V']]):
                for i in range(self.num_heads):
                    name_head = f'{name}.head{i}'
                    params_store[name_head] = stored_p[i, :, :].transpose()
            elif 'W_O' in name:
                # stored_p = stored_p.reshape(self.d_model, self.d_head, self.num_heads)
                stored_p = einops.rearrange(
                    stored_p, 'd (i h) -> d i h', i=self.num_heads, h=self.d_head)
                for i in range(self.num_heads):
                    name_head = f'{name}.head{i}'
                    params_store[name_head] = stored_p[:, i, :].transpose()
            else:
                params_store[name] = stored_p.transpose()
        return params_store

    def insert_new_params(self, new_params: Dict[str, ndarray]) -> None:
        new_state_dict = {}
        for name, p in self.named_parameters():
            if any([x in name for x in ['K', 'Q', 'V']]):
                new_p = []
                for i in range(self.num_heads):
                    name_head = f'{name}.head{i}'
                    new_p.append(new_params[name_head].transpose())
                p = np.stack(new_p, axis=0)
            elif 'W_O' in name:
                new_p = []
                for i in range(self.num_heads):
                    name_head = f'{name}.head{i}'
                    new_p.append(new_params[name_head].transpose())
                p = np.stack(new_p, axis=1)
                p = einops.rearrange(p, 'd i h -> d (i h)',
                                     i=self.num_heads, h=self.d_head)
                # p = p.reshape(self.d_model, self.d_head * self.num_heads)
            else:
                p = new_params[name].transpose()
            new_state_dict[name] = torch.tensor(p)
        self.load_state_dict(new_state_dict, strict=False)
