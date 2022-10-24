"""# Permutation Utils"""

from collections import defaultdict
from typing import Dict, Optional, NamedTuple

import numpy as np
# from jax import random
import torch
from scipy.optimize import linear_sum_assignment
from numpy import ndarray


class PermutationSpec(NamedTuple):
    perm_to_axes: dict
    axes_to_perm: dict

# def mlp_permutation_spec(num_hidden_layers: int) -> PermutationSpec:
#   """We assume that one permutation cannot appear in two axes of the same weight array."""
#   assert num_hidden_layers >= 1
#   return PermutationSpec(
#       perm_to_axes={
#           f"P_{i}": [(f"Dense_{i}/kernel", 1), (f"Dense_{i}/bias", 0), (f"Dense_{i+1}/kernel", 0)]
#           for i in range(num_hidden_layers)
#       },
#       axes_to_perm={
#           "Dense_0/kernel": (None, "P_0"),
#           **{f"Dense_{i}/kernel": (f"P_{i-1}", f"P_{i}")
#              for i in range(1, num_hidden_layers)},
#           **{f"Dense_{i}/bias": (f"P_{i}", )
#              for i in range(num_hidden_layers)},
#           f"Dense_{num_hidden_layers}/kernel": (f"P_{num_hidden_layers-1}", None),
#           f"Dense_{num_hidden_layers}/bias": (None, ),
#       })


def permutation_spec_from_axes_to_perm(axes_to_perm: dict) -> PermutationSpec:
    perm_to_axes = defaultdict(list)
    for wk, axis_perms in axes_to_perm.items():
        for axis, perm in enumerate(axis_perms):
            if perm is not None:
                perm_to_axes[perm].append((wk, axis))
    return PermutationSpec(perm_to_axes=dict(
        perm_to_axes), axes_to_perm=axes_to_perm)


def mlp_permutation_spec(num_hidden_layers: int) -> PermutationSpec:
    """We assume that one permutation cannot appear in two axes of the same weight array."""
    assert num_hidden_layers >= 1
    return permutation_spec_from_axes_to_perm({
        "fc1.weight": (None, "P_0"),
        **{f"fc{i+1}.weight": (f"P_{i-1}", f"P_{i}")
           for i in range(1, num_hidden_layers)},
        **{f"fc{i+1}.bias": (f"P_{i}", )
           for i in range(num_hidden_layers)},
        f"fc{num_hidden_layers+1}.weight": (f"P_{num_hidden_layers-1}", None),
        f"fc{num_hidden_layers+1}.bias": (None, ),
    })


def vgg16_permutation_spec() -> PermutationSpec:
    return permutation_spec_from_axes_to_perm({
        "Conv_0/kernel": (None, None, None, "P_Conv_0"),
        **{f"Conv_{i}/kernel": (None, None, f"P_Conv_{i-1}", f"P_Conv_{i}")
           for i in range(1, 13)},
        **{f"Conv_{i}/bias": (f"P_Conv_{i}", )
           for i in range(13)},
        **{f"LayerNorm_{i}/scale": (f"P_Conv_{i}", )
           for i in range(13)},
        **{f"LayerNorm_{i}/bias": (f"P_Conv_{i}", )
           for i in range(13)},
        "Dense_0/kernel": ("P_Conv_12", "P_Dense_0"),
        "Dense_0/bias": ("P_Dense_0", ),
        "Dense_1/kernel": ("P_Dense_0", "P_Dense_1"),
        "Dense_1/bias": ("P_Dense_1", ),
        "Dense_2/kernel": ("P_Dense_1", None),
        "Dense_2/bias": (None, ),
    })


def resnet20_permutation_spec() -> PermutationSpec:
    def conv(name, p_in, p_out): return {
        f"{name}/kernel": (None, None, p_in, p_out)}

    def norm(name, p): return {f"{name}/scale": (p, ), f"{name}/bias": (p, )}

    def dense(name, p_in, p_out): return {
        f"{name}/kernel": (p_in, p_out), f"{name}/bias": (p_out, )}

    # This is for easy blocks that use a residual connection, without any
    # change in the number of channels.
    def easyblock(name, p): return {
        **conv(f"{name}/conv1", p, f"P_{name}_inner"),
        **norm(f"{name}/norm1", f"P_{name}_inner"),
        **conv(f"{name}/conv2", f"P_{name}_inner", p),
        **norm(f"{name}/norm2", p)
    }

    # This is for blocks that use a residual connection, but change the number
    # of channels via a Conv.
    def shortcutblock(name, p_in, p_out): return {
        **conv(f"{name}/conv1", p_in, f"P_{name}_inner"),
        **norm(f"{name}/norm1", f"P_{name}_inner"),
        **conv(f"{name}/conv2", f"P_{name}_inner", p_out),
        **norm(f"{name}/norm2", p_out),
        **conv(f"{name}/shortcut/layers_0", p_in, p_out),
        **norm(f"{name}/shortcut/layers_1", p_out),
    }

    return permutation_spec_from_axes_to_perm({
        **conv("conv1", None, "P_bg0"),
        **norm("norm1", "P_bg0"),
        #
        **easyblock("blockgroups_0/blocks_0", "P_bg0"),
        **easyblock("blockgroups_0/blocks_1", "P_bg0"),
        **easyblock("blockgroups_0/blocks_2", "P_bg0"),
        #
        **shortcutblock("blockgroups_1/blocks_0", "P_bg0", "P_bg1"),
        **easyblock("blockgroups_1/blocks_1", "P_bg1"),
        **easyblock("blockgroups_1/blocks_2", "P_bg1"),
        #
        **shortcutblock("blockgroups_2/blocks_0", "P_bg1", "P_bg2"),
        **easyblock("blockgroups_2/blocks_1", "P_bg2"),
        **easyblock("blockgroups_2/blocks_2", "P_bg2"),
        #
        **dense("dense", "P_bg2", None),
    })


def nanda_transformer_permutation_spec(act_type: str) -> PermutationSpec:
    def norm(name, p): return {f"{name}.ln.weight": (p, ), f"{name}.ln.bias": (p, )}

    def attention(block_name, weight_name, head, p_in, p_out): return {
        f"{block_name}.{weight_name}.head{head}": (p_in, p_out)}

    def dense(block_name, weight_name, bias_name, p_in, p_out): return {f"{block_name}.{weight_name}": (p_in, p_out),
                                                                        f"{block_name}.{bias_name}": (p_out, )}
    heads = 4
    layers = 4

    def attention_block(name, p, i): return {
        **attention(f"{name}", "W_Q", i, p, f"P_{name}_wqk_head{i}"),
        **attention(f"{name}", "W_K", i, p, f"P_{name}_wqk_head{i}"),
        **attention(f"{name}", "W_V", i, p, f"P_{name}_wvo_head{i}"),
        **attention(f"{name}", "W_O", i, f"P_{name}_wvo_head{i}", p),
        # **attention(f"{name}", "W_Q", i, p, f"P_{name}_head{i}"),
        # **attention(f"{name}", "W_K", i, p, f"P_{name}_head{i}"),
        # **attention(f"{name}", "W_V", i, p, f"P_{name}_head{i}"),
        # **attention(f"{name}", "W_O", i, f"P_{name}_head{i}", p),
    }

    def dense_block(name, p): 
        if act_type == "solu":
            return {
            **dense(f"{name}", "W_in", "b_in", p, f"p_{name}_mlp"),
            **dense(f"{name}", "W_out", "b_out", f"p_{name}_mlp", p),
            **norm(f"{name}", f"p_{name}_mlp")
            }
        else:
            return {
            **dense(f"{name}", "W_in", "b_in", p, f"p_{name}_mlp"),
            **dense(f"{name}", "W_out", "b_out", f"p_{name}_mlp", p)
        }
    all_blocks = {"embed.W_E": (None, f"p_residual")}
    all_blocks["pos_embed.W_pos"] = (f"p_residual", None)
    for layer in range(layers):
        for head in range(heads):
            all_blocks.update(attention_block(
                f"blocks.{layer}.attn", f"p_residual", head))
        all_blocks.update(dense_block(f"blocks.{layer}.mlp", f"p_residual"))
    all_blocks["unembed.W_U"] = (None, f"p_residual")  # no need to transpose
    return permutation_spec_from_axes_to_perm(all_blocks)


def simple_nanda_transformer_permutation_spec() -> PermutationSpec:
    # conv = lambda name, p_in, p_out: {f"{name}/kernel": (None, None, p_in, p_out)}
    def norm(name, p): return {f"{name}.w_ln": (p, ), f"{name}.b_ln": (p, )}

    def attention(block_name, weight_name, head, p_in, p_out): return {
        f"{block_name}.{weight_name}.head{head}": (p_in, p_out)}

    def dense(block_name, weight_name, bias_name, p_in, p_out): return {f"{block_name}.{weight_name}": (p_in, p_out),
                                                                        f"{block_name}.{bias_name}": (p_out, )}
    heads = 4
    layers = 4

    def attention_block(name, p, i): return {
        **attention(f"{name}", "W_Q", i, p, f"P_{name}_head{i}"),
        **attention(f"{name}", "W_K", i, p, f"P_{name}_head{i}"),
        **attention(f"{name}", "W_V", i, p, f"P_{name}_head{i}"),
        **attention(f"{name}", "W_O", i, f"P_{name}_head{i}", p),
    }

    def dense_block(name, p): return {
        **dense(f"{name}", "W_in", "b_in", p, f"p_{name}_mlp"),
        **dense(f"{name}", "W_out", "b_out", f"p_{name}_mlp", p)
    }
    all_blocks = {"embed.W_E": (None, f"p_residual")}
    all_blocks["pos_embed.W_pos"] = (f"p_residual", None)
    for layer in range(layers):
        # for head in range(heads):
        #   all_blocks.update(attention_block(f"blocks.{layer}.attn", f"p_residual", head))
        all_blocks.update(dense_block(f"blocks.{layer}.mlp", f"p_residual"))
    all_blocks["unembed.W_U"] = (None, f"p_residual")  # no need to transpose
    return permutation_spec_from_axes_to_perm(all_blocks)


def get_permuted_param(ps: PermutationSpec, perm: Dict[str, ndarray],
                       k: str, params: Dict[str, ndarray], except_axis: Optional[int]=None) -> ndarray:
    """Get parameter `k` from `params`, with the permutations applied."""
    w = params[k]
    for axis, p in enumerate(ps.axes_to_perm[k]):
        # Skip the axis we're trying to permute.
        if axis == except_axis:
            continue
        # print(k)
        # None indicates that there is no permutation relevant to that axis.
        if p is not None:
            # if 'embed' in k:
            #   print(k)
            #   print(w.shape)
            #   print(perm[p])
            #   print(p, 'p')
            #   print(axis, 'axis')
            w = np.take(w, perm[p], axis=axis)
            # if 'embed' in k:
            #   print(w.shape, 'after')

    return w


def apply_permutation(ps: PermutationSpec, perm: Dict[str, ndarray], params: Dict[str, ndarray],
                      skipped_keys: Optional[list]=None) -> Dict[str, ndarray]:
    """Apply a `perm` to `params`."""
    if skipped_keys:
        new_params = {k: get_permuted_param(
            ps, perm, k, params) for k in params.keys() if k not in skipped_keys}
        new_params.update({k: params[k] for k in skipped_keys})
        return new_params
    else:
        return {k: get_permuted_param(ps, perm, k, params)
                for k in params.keys()}


def weight_matching(rng: int, ps: PermutationSpec, params_a: Dict[str, ndarray],
                    params_b: Dict[str, ndarray], max_iter: int=100, 
                    init_perm: Optional[Dict[str, ndarray]]=None) -> Dict[str, ndarray]:
    """Find a permutation of `params_b` to make them match `params_a`."""
    perm_sizes = {p: params_a[axes[0][0]].shape[axes[0][1]]
                  for p, axes in ps.perm_to_axes.items()}

    perm = {p: np.arange(n) for p, n in perm_sizes.items()
            } if init_perm is None else init_perm
    perm_names = list(perm.keys())
    np.random.seed(rng)
    for iteration in range(max_iter):
        progress = False
        for p_ix in np.random.permutation(len(perm_names)):
            p = perm_names[p_ix]
            n = perm_sizes[p]
            A = np.zeros((n, n))
            for wk, axis in ps.perm_to_axes[p]:
                w_a = params_a[wk]
                # print(p, n, 'perm_name, perm_size')
                # print(wk, 'name')
                # print(axis, 'axis')
                w_b = get_permuted_param(
                    ps, perm, wk, params_b, except_axis=axis)
                w_a = np.moveaxis(w_a, axis, 0).reshape((n, -1))
                w_b = np.moveaxis(w_b, axis, 0).reshape((n, -1))
                A += w_a @ w_b.T

            ri, ci = linear_sum_assignment(A, maximize=True)
            assert (ri == np.arange(len(ri))).all()

            oldL = np.vdot(A, np.eye(n)[perm[p]])
            newL = np.vdot(A, np.eye(n)[ci, :])
            print(f"{iteration}/{p}: {newL - oldL}")
            progress = progress or newL > oldL + 1e-12

            perm[p] = np.array(ci)

        if not progress:
            print(
                f'not making progress iteration {iteration}, {p_ix}, newL {newL}, oldL {oldL}')
            break

    return perm


def test_weight_matching():
    """If we just have a single hidden layer then it should converge after just one step."""
    ps = mlp_permutation_spec(num_hidden_layers=1)
    rng = 123  # random.PRNGKey(123)
    torch.random.manual_seed(rng)
    num_hidden = 10
    shapes = {
        "Dense_0/kernel": (2, num_hidden),
        "Dense_0/bias": (num_hidden, ),
        "Dense_1/kernel": (num_hidden, 3),
        "Dense_1/bias": (3, )
    }
    params_a = {k: torch.randn(shape).numpy() for k, shape in shapes.items()}
    params_b = {k: torch.randn(shape).numpy() for k, shape in shapes.items()}
    perm = weight_matching(rng, ps, params_a, params_b)
    print(perm)
