import torch
from torch.nn import functional as F
from tqdm import tqdm, trange
from copy import deepcopy
import numpy as np
from collections import Counter


def convert_to_tokens(indices, tokenizer, extended=False, extra_values_pos=None, strip=True):
    if extended:
        res = [
            tokenizer.convert_ids_to_tokens([idx])[0]
            if idx < len(tokenizer)
            else (
                f"[pos{idx-len(tokenizer)}]"
                if idx < extra_values_pos
                else f"[val{idx-extra_values_pos}]"
            )
            for idx in indices
        ]
    else:
        res = tokenizer.convert_ids_to_tokens(indices)
    if strip:
        res = list(map(lambda x: x[1:] if x[0] == "Ġ" else "#" + x, res))
    return res


def top_tokens(
    v,
    tokenizer,
    k=100,
    only_english=False,
    only_ascii=True,
    with_extra_info=False,
    exclude_brackets=False,
    # extended=True,
    # extra_values=None,
):
    # if tokenizer is None:
    #     tokenizer = my_tokenizer
    v = deepcopy(v)
    ignored_indices = []
    if only_ascii:
        ignored_indices = [
            key for val, key in tokenizer.vocab.items() if not val.strip("Ġ").isascii()
        ]
    if only_english:
        ignored_indices = [
            key
            for val, key in tokenizer.vocab.items()
            if not (val.strip("Ġ").isascii() and val.strip("Ġ[]").isalnum())
        ]
    if exclude_brackets:
        ignored_indices = set(ignored_indices).intersection(
            {
                key
                for val, key in tokenizer.vocab.items()
                if not (val.isascii() and val.isalnum())
            }
        )
        ignored_indices = list(ignored_indices)
    v[:, ignored_indices] = float("-inf")
    # extra_values_pos = len(v)
    # if extra_values is not None:
    #     v = torch.cat([v, extra_values])
    # print(v)
    T = 0.1
    probs = F.softmax(v / T, dim=-1)
    # print(probs)
    # print(probs.sum())
    entropy = torch.special.entr(probs).mean(-1)
    # print(entropy)
    prob_values, _ = torch.topk(probs, k=k, dim=-1)
    # print(values_)
    # print(indices_)
    values, indices = torch.topk(v, k=k, dim=-1)
    # print(values, 'no probs')
    # print(indices, 'no probs')
    # res = convert_to_tokens(
    #     indices, tokenizer, extended=extended, extra_values_pos=extra_values_pos
    # )
    if with_extra_info:
        return (
            indices.cpu().numpy(),
            values.cpu().numpy(),
            prob_values.cpu().numpy(),
            entropy.cpu().numpy(),
        )
    return (indices.cpu().numpy(),)


def top_matrix_tokens(
    mat,
    tokenizer,
    k=100,
    rel_thresh=None,
    thresh=None,
    sample_entries=10000,
    alphabetical=True,
    only_english=False,
    exclude_brackets=False,
    with_values=True,
    extended=True,
):
    mat = deepcopy(mat)
    ignored_indices = []
    if only_english:
        ignored_indices = [
            key
            for val, key in tokenizer.vocab.items()
            if not (val.isascii() and val.strip("[]").isalnum())
        ]
    if exclude_brackets:
        ignored_indices = set(ignored_indices).intersection(
            {
                key
                for val, key in tokenizer.vocab.items()
                if not (val.isascii() and val.isalnum())
            }
        )
        ignored_indices = list(ignored_indices)
    mat[ignored_indices, :] = -np.inf
    mat[:, ignored_indices] = -np.inf
    cond = torch.ones_like(mat).bool()
    if rel_thresh:
        cond &= mat > torch.max(mat) * rel_thresh
    if thresh:
        cond &= mat > thresh
    entries = torch.nonzero(cond)
    if sample_entries:
        entries = entries[
            np.random.randint(len(torch.nonzero(cond)), size=sample_entries)
        ]
    res_indices = sorted(
        entries, key=lambda x: x[0] if alphabetical else -mat[x[0], x[1]]
    )
    res = [
        *map(
            partial(convert_to_tokens, extended=extended, tokenizer=tokenizer),
            res_indices,
        )
    ]

    if with_values:
        res_ = []
        for (x1, x2), (i1, i2) in zip(res, res_indices):
            res_.append((x1, x2, mat[i1][i2].item()))
        res = res_
    return res