"""
Train a simple transformer language model on wikitext, to recording various model internals,
e.g. activation sparsity, attention entropy, with option to add fisher penalty and sparsity
penalty.
"""
import os
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np
from pathlib import Path
import pickle
from transformers.optimization import get_linear_schedule_with_warmup

from transparent.transformer import Transformer
from transparent.data_utils import get_tokenized_wikitext
from transparent.training_utils import train_epoch, test_epoch

drive_root = Path('./savedlms/')
try:
    os.mkdir(drive_root)
except FileExistsError:
    print(f'{drive_root} exists already.')

def main():
    parser = ArgumentParser(
        description="Train language models over a range of learning rates."
    )
    parser.add_argument(
        "--share-embeddings", action="store_true",
        help="Use the same embedding initialization as model 0 for model 1."
    )
    
    parser.add_argument(
        "--act-type", device=str,
        help="Activation function for transformer.",
    )
    parser.add_argument(
        "--device", device=str,
        help="Device to perform computation on (e.g. 'cuda').",
    )
    parser.add_argument(
        "--l1-norm-penalty", device=float, default=0.0,
        help="Weight to give activation l1 norm penalty.",
    )
    parser.add_argument(
        "--fisher-penalty-weight", device=float, default=0.0,
        help="Weight to give activation l1 norm penalty.",
    )
    args = parser.parse_args()

    # Transformer hyper-params
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    train_loader, validation_loader = get_tokenized_wikitext(args)

    act2act = {'solu': 'SoLU', 'gelu': 'GeLU', 'relu': 'ReLU'}
    act_device = act2act[args.act_type]

    num_layers = 4
    d_vocab = len(tokenizer)
    d_model = 256
    d_mlp = 1024
    num_heads = 4
    d_head = 64
    n_ctx = 128
    use_ln = False
    train_model = True
    save_models = True
    save_every = 8
    num_epochs = 6
    weight_decay = 0.0
    num_training_steps = len(train_loader) * num_epochs
    
    # Training loop
    seed = 1234
    # data_store = []
    # plot_labels = []
    if train_model:
        for learning_rate in [0.0001, 0.001]:
            torch.manual_seed(seed) # use same model init for every lr etc.
            for run_id in range(2):
                drive_path = drive_root / f'lm_l1pen_{args.l1_norm_penalty}_fpen{args.fisher_penalty_weight}_{args.act_device}_seed{seed}_lr_{learning_rate}_wd_{weight_decay}_{run_id}'
                try:
                    os.mkdir(drive_path)
                except FileExistsError:
                    pass
                if os.path.exists(drive_path / 'data.pkl') and not args.share_embeddings:
                    print(f'Skipping {drive_path}')
                else:
                    print(f'Training {drive_path}')
                    model = Transformer(
                        num_layers=num_layers,
                        d_vocab=d_vocab,
                        d_model=d_model,
                        d_mlp=d_mlp,
                        d_head=d_head,
                        num_heads=num_heads,
                        n_ctx=n_ctx,
                        act_device=act_device,
                        use_cache=True,
                        use_ln=use_ln,
                    )
                    model.to(args.device)
                    if args.share_embeddings:
                        if run_id == 0:
                            embed_state_dict = {name: weight for name, weight in model.state_dict().items()
                                                if "embed" in name}
                            print(f"Reusing from model 0: {embed_state_dict.keys()}")
                            continue
                        else:
                            model.load_state_dict(embed_state_dict, strict=False)
                            drive_path = drive_path / "model0embeds"
                            try:
                                os.mkdir(drive_path)
                            except FileExistsError:
                                print(f"Warning, already created {drive_path}")
                    
                    model.cache_all(model.cache, l1_norm=args.l1_norm_penalty)

                    optimizer = optim.AdamW(
                        model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.98)
                    )
                    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, 
                                                                num_training_steps=num_training_steps)
                    run_name = f"run_{int(time.time())}"
                    print(f"Run name {run_name}")
                    if save_models:
                        #os.mkdir(root / run_name)
                        save_dict = {
                            "model": model.state_dict(),
                        }
                        torch.save(save_dict, drive_path / "init.pth")
                    train_losses = []
                    test_losses = []
                    train_entropy = []
                    test_entropy = []
                    train_sparsity = []
                    test_sparsity = []
                    for epoch in range(num_epochs):
                        model.train()
                        train_epoch(model, train_loader, scheduler, optimizer, train_sparsity, 
                                    train_entropy, train_losses, args)
                        model.eval()
                        test_loss, entropy, sparsity = test_epoch(model, validation_loader, args)
                        test_losses.append(test_loss)
                        test_entropy.append(entropy)
                        test_sparsity.append(sparsity)
                        print(
                            f"{epoch}_test{np.log(test_loss):.4f}_entropy{entropy:.4f}_sparsity{sparsity:.4f}"
                        )  
                        if (save_models) and (epoch % save_every == 0):
                            save_dict = {
                                "model": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                "train_loss": train_losses[-1],
                                "test_loss": test_loss,
                                "epoch": epoch,
                            }
                            torch.save(save_dict, drive_path / f"{epoch}.pth")
                            print(f"Saved model to {drive_path/f'{epoch}.pth'}")
                    if save_models:
                        save_dict = {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "train_loss": train_losses[-1],
                            "test_loss": test_loss,
                            "train_losses": train_losses,
                            "test_losses": test_losses,
                            "epoch": epoch,
                        }
                        torch.save(save_dict, drive_path / f"final.pth")
                        print(f"Saved model to {drive_path/f'final.pth'}")
            
                    with open(drive_path / 'data.pkl', 'wb') as file:
                        pickle.dump([train_losses, test_losses, train_sparsity, 
                                     test_sparsity, train_entropy, test_entropy], file)
    

if __name__ == "__main__":
    main()
