"""## Permuting trained language models"""
import os
import torch
import pickle
from argparse import ArgumentParser
import numpy as np
from pathlib import Path
from copy import deepcopy
from tabulate import tabulate
from collections import defaultdict

from transparent.transformer import Transformer
from transparent.data_utils import get_tokenized_wikitext
from transparent.logit_lens import top_tokens, convert_to_tokens


class ModelDecoder:
    def __init__(self, learning_rate, model_path, args, validation_loader, tokenizer):
        self.learning_rate = learning_rate
        self.model_path = model_path
        self.args = args
        self.tokenizer = tokenizer
        self.fisher_penalty_weight = 0.0
        self.seed = 1234
        self.weight_decay = 0.0
        self.num_layers = 4
        self.d_vocab = len(self.tokenizer)
        self.d_model = 256
        self.d_mlp = 1024
        self.num_heads = 4
        self.d_head = 64
        self.n_ctx = 128
        act2act = {'solu': 'SoLU', 'gelu': 'GeLU', 'relu': 'ReLU'}
        self.act_type = act2act[self.args.act_type]


    def load_and_evaluate_model(self):
        '''Load models and decode their weights into embedding space'''
        use_ln = False
        saved_models = []
        entropy = []
        sparsity = []
        for run_id in range(2):
            # drive_path = self.model_path / (f'lm_l1pen_{self.args.l1_norm_penalty}_fpen{self.fisher_penalty_weight}'
            drive_path = self.model_path / (f'lm_fpen{self.fisher_penalty_weight}'
                                             f'_{self.args.act_type}_seed{self.seed}_lr_{self.learning_rate}_wd'
                                             f'_{self.weight_decay}_{run_id}')
            print(f'Attempting to load {drive_path}')
    
            if run_id == 1 and self.args.share_embeddings:
                drive_path = drive_path / "model0embeds"
            with open(drive_path / "data.pkl", "rb") as file:
                extra_data = pickle.load(file)
                entropy.append(extra_data[-1][-1])
                sparsity.append(extra_data[-3][-1])
                
            save_dict = torch.load(drive_path / f"final.pth", map_location=torch.device('cpu'))
            model_dict = save_dict["model"]
            model = Transformer(
                            num_layers=self.num_layers,
                            d_vocab=self.d_vocab,
                            d_model=self.d_model,
                            d_mlp=self.d_mlp,
                            d_head=self.d_head,
                            num_heads=self.num_heads,
                            n_ctx=self.n_ctx,
                            act_type=self.act_type,
                            use_cache=True,
                            use_ln=use_ln,
                            )
            model.load_state_dict(model_dict)
            print('Finished loading model weights.')
            # saved_models.append(model)
        
            self.decode_model_weights(model)


    def decode_model_weights(self, model):
        emb = model.get_output_embeddings().data.detach()
            
        K = torch.cat(
            [
                model.get_parameter(f"blocks.{j}.mlp.W_out").T
                for j in range(self.num_layers)
            ]
        ).detach()
        # V = torch.cat(
        #     [
        #         model.get_parameter(f"transformer.h.{j}.mlp.c_proj.weight")
        #         for j in range(num_layers)
        #     ]
        # ).detach()
        #
        # W_Q, W_K, W_V = (
        #     torch.cat(
        #         [
        #             model.get_parameter(f"transformer.h.{j}.attn.c_attn.weight")
        #             for j in range(num_layers)
        #         ]
        #     )
        #     .detach()
        #     .chunk(3, dim=-1)
        # )
        # W_O = torch.cat(
        #     [
        #         model.get_parameter(f"transformer.h.{j}.attn.c_proj.weight")
        # for j in range(num_layers)
        # ]
        # ).detach()
        
        K_heads = K.reshape(self.num_layers, -1, self.d_model)
        # V_heads = V.reshape(num_layers, -1, self.d_model)
        # d_int = K_heads.shape[1]
        #
        # W_V_heads = W_V.reshape(num_layers, self.d_model, num_heads, head_size).permute(
        #     0, 2, 1, 3
        # )
        # W_O_heads = W_O.reshape(num_layers, num_heads, head_size, self.d_model)
        # W_Q_heads = W_Q.reshape(num_layers, self.d_model, num_heads, head_size).permute(
        #     0, 2, 1, 3
        # )
        # W_K_heads = W_K.reshape(num_layers, self.d_model, num_heads, head_size).permute(
        #     0, 2, 1, 3
        # )
        #
        # i1, i2 = 21, 7
        # print(i1, i2)
        print(K_heads.shape)
        self.keys_store = []
        layers, keys, _ = K_heads.shape
        batch_size = 100
        steps = keys // batch_size
        entropy_store = defaultdict(list)
        for layer in range(layers):
            print(f"Analyzing layer {layer}, norm {np.linalg.norm(K_heads[layer, :]):.4f}")
            K_heads [layer, :] = K_heads[layer, :] / np.linalg.norm(K_heads[layer, :], axis=0)
            per_layer_store = []
            for i in range(steps):
                tokens, values, prob_values, entropy = top_tokens(
                    (K_heads[layer, i * batch_size : min((i + 1) * batch_size, keys)]) @ emb,
                    tokenizer=self.tokenizer,
                    k=10,
                    with_extra_info=True,
                )
                # print(tokens.shape)
                num_examples, _ = tokens.shape
                for j in range(num_examples):

                    t, v, p, e = tokens[j, :], values[j, :], prob_values[j, :], entropy[j]
                    entropy_store[layer].append(e)
                    neuron_tokens = convert_to_tokens(t, self.tokenizer)
                    neuron_tokens = [(token, prob) for token, prob in zip(neuron_tokens, p)]
                    per_layer_store.append((neuron_tokens, v, e))

                    if e > 1.5e-4 and e < 3e-4:
                        print(f"Analyzing layer {layer} and neuron {batch_size*i+j} with entropy {e}")
                        print(tabulate([neuron_tokens]))
            self.keys_store.append(per_layer_store)
        per_layer_entropy = [(np.mean(entropy_store[i]), np.std(entropy_store[i])) for i in range(layers)]
        per_layer_entropy = [f"{entropy[0]:.4e} {entropy[1]:.4e}" for entropy in per_layer_entropy]
        print(per_layer_entropy)
            

def main():
    parser = ArgumentParser(
        description="Decode transformer language models over a range of learning rates."
    )
    parser.add_argument(
        "--share-embeddings", action="store_true",
        help="Use the same embedding initialization as model 0 for model 1."
    )
    parser.add_argument(
        "--act-type", type=str,
        help="Activation type for transformer",
    )
    parser.add_argument(
        "--l1-norm-penalty", type=float, default=0.0,
        help="Weight to give activation l1 norm penalty.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for training dataset.",
    )
    args = parser.parse_args()
    model_path = Path('./savedlms')
    learning_rates = [0.0056, 0.003, 0.0018, 0.001, 0.00056, 0.0003, 0.00018, 0.0001]
    
    train_loader, validation_loader, tokenizer = get_tokenized_wikitext(args)
    all_keys = {}
    for learning_rate in learning_rates:
        try:
            decoder = ModelDecoder(learning_rate, model_path, args, validation_loader, tokenizer)
            decoder.load_and_evaluate_model()
            all_keys[model_path] = decoder.keys_store
        except FileNotFoundError:
            print(f"Skipping lr {learning_rate}")
    
    with open("savedlms/{args.act_type}_keys.pkl", "wb") as f:
        pickle.dump(all_keys, f)

    annotated_keys = {}


if __name__ == "__main__":
    main()
