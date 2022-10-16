import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from torch.utils.data.dataloader import DataLoader
from argparse import Namespace

from transparent.transformer import Transformer


def cross_entropy_high_precision(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # Shapes: batch x vocab, batch
    # Cast logits to float64 because log_softmax has a float32 underflow on overly
    # confident data and can only return multiples of 1.2e-7 (the smallest float x
    # such that 1+x is different from 1 in float32). This leads to loss spikes
    # and dodgy gradients
    logprobs = F.log_softmax(logits.to(torch.float64), dim=-1)
    prediction_logprobs = torch.gather(logprobs, index=labels[:, None], dim=-1)
    loss = -torch.mean(prediction_logprobs)
    return loss


def train_epoch(model: Transformer, train_loader: DataLoader, scheduler, optimizer, train_sparsity: list,
                train_entropy: list, train_losses: list, args: Namespace) -> None:
    '''Train on dataset in `train_loader` for one epoch.'''
    for batch_idx, batch in enumerate(train_loader):
        # print(batch['input_ids'][0].shape)
        # print(batch['input_ids'][0])
        # print(tokenizer.decode(batch["input_ids"][0]))
        data = batch['input_ids'].to(args.device)
        # print(data.shape)
        logits = model(data)[:, :-1]
        batch_size, sequence_length, vocab = logits.shape
        labels = batch['labels'].to(device)[:, 1:]
        logits = logits.reshape(batch_size * sequence_length, vocab)
        labels = labels.reshape(batch_size * sequence_length)
        train_loss = cross_entropy_high_precision(logits, labels)
        with torch.no_grad():
            sparsity = get_mean(model.get_sparsity())
        train_sparsity.append(sparsity)
        entropy = get_mean(model.get_entropy())
        train_entropy.append(entropy)
        train_losses.append(train_loss.item())
        if batch_idx % 100 == 0:
            print(
                f"{batch_idx}_train{np.log(train_loss.item()):.4f}_entropy{entropy:.4f}_sparsity{sparsity:.4f}"
            )
        if args.l1_norm_penalty > 0.0 and model.training:
            activation_l1_norm = model.get_activation_l1_norm()
            train_loss += args.l1_norm_penalty * activation_l1_norm
            if batch_idx % 100 == 0:
                print(np.log(train_loss.item()))
        if args.fisher_penalty_weight > 0.0 and model.training:
            outdx = Categorical(logits=logits).sample().detach()
            f_loss = cross_entropy_high_precision(logits, outdx)
            grads = torch.autograd.grad(
                f_loss,
                [p for n, p in model.named_parameters() if p.requires_grad == True],
                retain_graph=True,
                create_graph=True,
            )
            gr_norm_sq = 0.0
            for gr in grads:
                if gr is not None:
                    gr_norm_sq += (gr**2).sum()

            train_loss += args.fisher_penalty_weight * gr_norm_sq
        else:
            pass
        train_loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()


def test_epoch(model: Transformer, test_loader: DataLoader, args: Namespace, get_stats: bool=True) -> tuple:
    '''Test on dataset in `test_loader` for one epoch. Optionally collect stats
    about model internals.'''
    loss_store = []
    entropy_store = []
    sparse_store = []
    for batch_idx, batch in enumerate(test_loader):
        data = batch['input_ids'].to(args.device)
        logits = model(data)[:, :-1]
        batch_size, sequence_length, vocab = logits.shape
        labels = batch['labels'].to(args.device)[:, 1:]
        logits = logits.reshape(batch_size * sequence_length, vocab)
        labels = labels.reshape(batch_size * sequence_length)
        test_loss = cross_entropy_high_precision(logits, labels).item()
        if get_stats:
            sparsity = get_mean(model.get_sparsity())
            sparse_store.append(sparsity)
            entropy = get_mean(model.get_entropy())
            entropy_store.append(entropy)
        loss_store.append(test_loss)
    n = len(test_loader)
    if get_stats:
        return (sum(loss_store) / n, sum(entropy_store) / n,
                sum(sparse_store) / n)
    else:
        return (sum(loss_store) / n,)


def get_mean(torch_list: list) -> float:
    torch_list = [x.item() for x in torch_list]
    return sum(torch_list) / len(torch_list)
