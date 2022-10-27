""" Permuting trained language models"""
import os
import json
import torch
import pickle
from argparse import ArgumentParser
import numpy as np
from pathlib import Path
from copy import deepcopy
from collections import defaultdict
from transformers import AutoTokenizer

from easy_transformer import EasyTransformer

from transparent.transformer import Transformer
from transparent.training_utils import test_epoch, test_epoch_kl
from transparent.data_utils import get_tokenized_wikitext, get_tokenized_code, get_tokenized_openwebtext
from transparent.permutation_utils import nanda_transformer_permutation_spec, weight_matching, apply_permutation


def linear_interporation(state_dict1, state_dict2, alpha=1):

    new_dict = {}
    for key in state_dict1.keys():
        if 'mask' in key:
            new_dict[key] = state_dict1[key]
        else:
            new_dict[key] = alpha * state_dict1[key] + (1 - alpha) * state_dict2[key]
    return new_dict
    
def linear_mode_connectivity(model, state_dict1, state_dict2, dataloader, args, bins=10):

    original_weight = deepcopy(model.state_dict())
    all_loss = []
    
    for i in range(bins+1):
        alpha = i/bins
        # print(alpha)
        new_state_dict = linear_interporation(state_dict1, state_dict2, alpha)
        model.load_state_dict(new_state_dict)
        model.to(args.device)
        model.eval()
        test_loss = test_epoch(model, dataloader, args, get_stats=False)
        all_loss.append(test_loss)
        # print(test_loss)
        model.to('cpu')
    
    # Accuracy
    # top_acc = (all_accuracy[0] + all_accuracy[-1]) / 2
    # bottom_acc = np.min(np.array(all_accuracy))
    
    # Loss
    top_loss = np.max(np.array(all_loss))
    bottom_loss = (all_loss[0] + all_loss[-1]) / 2 
    
    model.load_state_dict(original_weight)
    
    return top_loss - bottom_loss, all_loss

def get_model(args, model_path, tokenizer, run_id, results_store):
    if 'stanford' in args.pretrained_model:
        model = EasyTransformer.from_pretrained(f'{args.pretrained_model}-{run_id}')
    else:
        seed = 1234
        weight_decay = 0.0
        num_layers = 4
        d_vocab = len(tokenizer)
        d_model = 256
        d_mlp = 1024
        num_heads = 4
        d_head = 64
        n_ctx = 128
        act2act = {'solu': 'SoLU', 'gelu': 'GeLU', 'relu': 'ReLU'}
        act_type = act2act[args.act_type]
        use_ln = False
        drive_path = model_path / (f'lm_l1pen_{args.l1_norm_penalty}_fpen{args.fisher_penalty_weight}' +
                                   f'_{args.act_type}_seed{seed}_lr_{learning_rate}_wd_{weight_decay}_{run_id}')
        if run_id == 1 and args.share_embeddings:
            drive_path = drive_path / "model0embeds"
        with open(drive_path / "data.pkl", "rb") as file:
            extra_data = pickle.load(file)
            results_store["entropy"].append(extra_data[-1][-1])
            results_store["sparsity"].append(extra_data[-3][-1])
        save_dict = torch.load(drive_path / f"final.pth", map_location=args.device)
        model_dict = save_dict["model"]
        model = Transformer(
                        num_layers=num_layers,
                        d_vocab=d_vocab,
                        d_model=d_model,
                        d_mlp=d_mlp,
                        d_head=d_head,
                        num_heads=num_heads,
                        n_ctx=n_ctx,
                        act_type=act_type,
                        use_cache=True,
                        use_ln=use_ln,
                    )
        model.load_state_dict(model_dict)
    return model

def load_and_evaluate_model(results_store, learning_rate, model_path, args, 
                            validation_loader, code_loader, owp_loader, tokenizer):
    
    saved_models = []

    model_ids = ['A', 'B'] if 'stanford' in args.pretrained_model else range(2)
    for run_id in model_ids:
        model = get_model(args, model_path, tokenizer, run_id, results_store)
        saved_models.append(model)

    for i, model in enumerate(saved_models):
        model.to(args.device)
        model.eval()
        test_loss = test_epoch(model, validation_loader, args, get_stats=False)
        results_store['losses'].append(test_loss)
        print(f"Model {i} has loss {test_loss}")
        model.to('cpu')
    
    if args.test_ood:
        code_kl_divergence, code_top1_matching = test_epoch_kl(saved_models, code_loader, args)
        print(f"Code ood kl {code_kl_divergence} and top 1 matching {code_top1_matching}")
        wiki_kl_divergence, wiki_top1_matching = test_epoch_kl(saved_models, validation_loader, args)
        print(f"Wiki ood kl {wiki_kl_divergence} and top 1 matching {wiki_top1_matching}")
        owp_kl_divergence, owp_top1_matching = test_epoch_kl(saved_models, owp_loader, args)
        print(f"Owp ood kl {owp_kl_divergence} and top 1 matching {owp_top1_matching}")
        result_store["code_kl"] = code_kl_divergence
        result_store["wiki_kl"] = wiki_kl_divergence
        result_store["owp_kl"] = owp_kl_divergence
        result_store["code_top1"] = code_top1_matching
        result_store["wiki_top1"] = wiki_top1_matching
        result_store["owp_top1"] = owp_top1_matching

    model2_old = deepcopy(saved_models[1])
    model1_dict = saved_models[0].get_permutation_dict()
    model2_dict = saved_models[1].get_permutation_dict()
    permutation_spec = nanda_transformer_permutation_spec(args.act_type)
    
    state_dict_model2 = model2_old.state_dict()
    state_dict_model1 = saved_models[0].state_dict()
    vanilla_gap, vanilla_losses = linear_mode_connectivity(model, state_dict_model1, 
                                                           state_dict_model2, validation_loader, args)
    print(f'vanilla difference: {vanilla_gap}')
    if args.skip_permutation_testing:
        return 
    for seed in range(5):
        final_permutation = weight_matching(seed, permutation_spec,
                                        model1_dict, model2_dict, max_iter=100)
        model2_dict_new = apply_permutation(permutation_spec, final_permutation, model2_dict)
    
        for i, new_params in enumerate([model1_dict, model2_dict_new]):
            saved_models[i].insert_new_params(new_params)
    
        for i, model in enumerate(saved_models):
            model.to(args.device)
            model.eval()
            test_loss = test_epoch(model, validation_loader, args, get_stats=False)
            print(f"After permutation model {i} has {test_loss}")
            model.to('cpu')
    
        # Linear mode connectivity
            
        model = Transformer(
            num_layers=num_layers,
            d_vocab=d_vocab,
            d_model=d_model,
            d_mlp=d_mlp,
            d_head=d_head,
            num_heads=num_heads,
            n_ctx=n_ctx,
            act_type=act_type,
            use_cache=True,
            use_ln=use_ln,
                                )
            
        state_dict_permuted_model2 = saved_models[1].state_dict()
        gap, permuted_losses = linear_mode_connectivity(model, state_dict_model1, 
                                                        state_dict_permuted_model2, validation_loader, args)
        results_store['permuted_losses'].append(permuted_losses)
        results_store['gap'].append(gap)
        print(f'Permuted difference {gap}')
        
    return 

def main():
    parser = ArgumentParser(
        description="Train language models over a range of learning rates."
    )
    parser.add_argument(
        "--share-embeddings", action="store_true",
        help="Use the same embedding initialization as model 0 for model 1."
    )
    parser.add_argument(
        "--pretrained-model", type=str, default="none",
        help="Use the pre-trained mistral models."
    )
    parser.add_argument(
        "--test-ood", action="store_true",
        help="Test on code data, i.e. out of distribution."
    )
    parser.add_argument(
        "--skip-permutation-testing", action="store_true",
        help="Don't test linear mode connectivity after permuting."
    )
    parser.add_argument(
        "--act-type", type=str,
        help="Activation type for transformer",
    )
    parser.add_argument(
        "--device", type=str,
        help="Device to perform computation on (e.g. 'cuda').",
    )
    parser.add_argument(
        "--model-path", type=str, default="./savedlms",
        help="Path to saved models."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for training dataset.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=0,
        help="Steps to wait before updating params.",
    )
    parser.add_argument(
        "--l1-norm-penalty", type=float, default=0.0,
        help="Weight to give activation l1 norm penalty.",
    )
    parser.add_argument(
        "--fisher-penalty-weight", type=float, default=0.0,
        help="Weight to give activation l1 norm penalty.",
    )
    args = parser.parse_args()
    model_path = Path(args.model_path)
    results = {}

    if 'stanford' in args.pretrained_model:
        tokenizer = AutoTokenizer.from_pretrained('stanford-crfm/alias-gpt2-small-x21')
    else:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    train_loader, validation_loader = get_tokenized_wikitext(args, tokenizer)
    code_loader = get_tokenized_code(args, tokenizer)
    owp_loader = get_tokenized_openwebtext(args, tokenizer)

    if 'stanford' in args.pretrained_model:
        results_path = model_path / 'results' / args.pretrained_model
        load_and_evaluate_model(results, -1, model_path, args, validation_loader, code_loader,
                                owp_loader, tokenizer)
    else:
        learning_rates = [0.0056, 0.003, 0.0018, 0.001, 0.00056, 0.0003, 0.00018, 0.0001]
        results_path = model_path / 'results' / (f'l1pen_{args.l1_norm_penalty}_fpen{args.fisher_penalty_weight}_' +
                                     f'{args.act_type}_seed1234_wd_0.0')
        try:
            os.makedirs(results_path)
        except FileExistsError:
            print(f'{results_path} exists already.')
    
        for learning_rate in learning_rates:
            try:
                results[learning_rate] = defaultdict(list)
                load_and_evaluate_model(results[learning_rate], learning_rate, model_path, args, validation_loader, code_loader,
                                        owp_loader, tokenizer)
            except FileNotFoundError:
                print(f"Skipping lr {learning_rate}")
        
        print(results)
        with open(results_path / 'results.json', 'w') as f:
            json.dump(results, f)
        


if __name__ == "__main__":
    main()
