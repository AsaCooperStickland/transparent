# Transparent: Understanding Transformer Internals

# Installation

```
git clone git@github.com:AsaCooperStickland/transparent.git
cd transparent
pip install -e .
```

Training a language model with activation either `relu`,`gelu` or `solu`.
You can add a penalty to the Fisher information matrix with `--fisher-penalty-weight`, or add an L1 norm penalty to the activations with `--l1-norm-penalty`.
```
python transparent/scripts/train_wikitext_transformer.py --act-type $ACTIVATION 
```
Performing a linear mode connectivity check on two models trained with the same settings, for a particular activation and l1/fisher penalty:
```
python transparent/scripts/basin_testing.py --act-type relu --l1-norm-penalty 0.0005 --fisher-penalty-weight 0.1
```
Evaluate custom trained models by decoding their weights into [token space](https://arxiv.org/abs/2209.02535).
Huggingface pretrained model support coming soon!
So far only works for the second feed-forward network weights.
```
python transparent/scripts/decode_value_weights.py --act-type $ACTIVATION 
```
