# Cross-Encoder Rediscovers a Semantic-Variant of BM25

**Authors:** Meng Lu, Catherine Chen, Carsten Eickhoff

This repository contains code and experiments from our paper:  
**"Cross-Encoder Rediscovers a Semantic-Variant of BM25"**

---

## TransformerLens Modifications

This codebase is built on top of a customized version of [`TransformerLens`](https://github.com/neelnanda-io/TransformerLens)。 We modified the following files to support **activation and path patching in a retrieval setting for Cross-Encoder**.

1. `load_from_pretrained.py`: sets up necessary configs  
2. `components.py`: customized model components  
3. `HookedEncoder.py`: supports patching and token-level caching

### Usage Example

```python
pre_trained_model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"
tokenizer, tl_model, pooler_layer, dropout_layer, classifier_layer = load_tokenizer_and_models(pre_trained_model_name, device)

tokenized_query_doc = tokenizer([query], [original_doc], return_tensors="pt", padding=True, truncation=True)

outputs, cache = tl_model.run_with_cache(
    tokenized_query_doc["input_ids"],
    return_type="embeddings",
    one_zero_attention_mask=tokenized_query_doc["attention_mask"],
    token_type_ids=tokenized_query_doc["token_type_ids"]
)
```


## Diagnostic Datasets

We conduct activation patching experiments on several diagnostic datasets:
1. tfc1: tfc1_add_baseline_final_dd_append_corpus.json
2. stmc1: stmc1_add_append_final_dd_corpus.json
3. lnc1: used with experiment_lnc2.py (WIP)
3. tfc2: additional variant of tfc1  (WIP)



## Activation & Path Patching (IR Model)

We adapt activation patching and path patching for use with cross-encoder retrieval models.

### Demo Notebook
```crossencoder_demo_patching.ipynb```: walkthrough of patching procedures on a cross-encoder model.

### Helper Scripts

1. ``` Patching_helpers.py```: implements key patching routines
2. ``` helpers.py```: utilities for processing and visualization


## Controllable IR and Downstream Experiments

1. ```model_editing_SVD_corr.p```: replicates experiments using SVD-based vector editing
2. ```forbidden.py```: analyzes and blocks specific token contributions



## Linear Approximation Experiments

(Note: Some scripts listed here are not yet uploaded)
1. ```paper_graphs2.py```: reproduces results for Section X (WIP)
2. ```BM_rerank_all.py```: performs BM25-style re-ranking with approximated features (WIP)



## Citation