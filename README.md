# Cross-Encoder Rediscovers a Semantic-Variant of BM25

**Authors:** Meng Lu, Catherine Chen, Carsten Eickhoff

This repository contains code and experiments from our paper:  
**"Cross-Encoder Rediscovers a Semantic-Variant of BM25"**

---

## TransformerLens Modifications

This codebase is built on top of a customized version of [`TransformerLens`](https://github.com/neelnanda-io/TransformerLens), with additional changes to support **activation patching in a retrieval setting**.

Modifications were made to support the retrieval model:

### Modified Files:
1. `load_from_pretrained.py`: sets up necessary configs  
2. `components.py`: customized model components  
3. `HookedEncoder.py`: supports patching and token-level caching

### Model Usage Example

```python
tokenized_query_doc = tokenizer([query], [original_doc], return_tensors="pt", padding=True, truncation=True)

outputs, cache = tl_model.run_with_cache(
    tokenized_query_doc["input_ids"],
    return_type="embeddings",
    one_zero_attention_mask=tokenized_query_doc["attention_mask"],
    token_type_ids=tokenized_query_doc["token_type_ids"]
)
```


## Activation & Path Patching (IR Model)

We adapt activation patching and path patching for use with cross-encoder retrieval models.

Demo Notebook (Section 4)
 • crossencoder_demo_patching.ipynb: walkthrough of patching procedures on a cross-encoder model.

Helper Scripts
 • patching_helpers.py: implements key patching routines
 • helpers.py: utilities for processing and visualization
 
Diagnostic Datasets (Table 1)
 • load_diagnostic_datasets.py: functions to load four diagnostic datasets (TFC1, STMC1, LNC2, TFC2) in appropriate format.


## Controllable IR and Downstream Experiments
 • model_editing_SVD_corr.py: replicates experiments using SVD-based vector editing
 • forbidden.py: analyzes and blocks specific token contributions



## Linear Approximation Experiments

(Note: Some scripts listed here are not yet uploaded)
 • paper_graphs2.py: reproduces results for Section X (WIP)
 • BM_rerank_all.py: performs BM25-style re-ranking with approximated features (WIP)



## Citation
