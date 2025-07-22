# Cross-Encoder Rediscovers a Semantic-Variant of BM25

**Authors:** Meng Lu, Catherine Chen, Carsten Eickhoff

This repository contains code and experiments from our paper:  
[**Cross-Encoder Rediscovers a Semantic-Variant of BM25**](https://arxiv.org/abs/2502.04645)

---

## Table of Contents
- [Required Packages](#required-packages)
- [TransformerLens Modifications](#transformerlens-modifications)
  - [Modified Files](#modified-files)
  - [Model Usage Example](#model-usage-example)
- [Activation & Path Patching](#activation--path-patching)
  - [Demo Notebook](#demo-notebook-section-4)
  - [Helper Scripts](#helper-scripts)
  - [Diagnostic Datasets](#diagnostic-datasets-table-1)
- [Controllable IR and Downstream Experiments](#controllable-ir-and-downstream-experiments)
- [Linear Approximation Experiments](#linear-approximation-experiments)
- [How to Cite Us](#how-to-cite-us)

---

---

## Required Packages

Run ```pip install -r requirements.txt``` to install required packages for conducting following experiments.

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


## Activation & Path Patching

We adapt activation patching and path patching for use with cross-encoder retrieval models.

Demo Notebook (Section 4)

 • ```crossencoder_demo_patching.ipynb```: walkthrough of patching procedures on a cross-encoder model.

Helper Scripts

 • ```patching_helpers.py```: implements key patching routines

 • ```helpers.py```: utilities for processing and visualization
 
Diagnostic Datasets (Table 1)

 • ```load_diagnostic_datasets.py```: functions to load four diagnostic datasets (TFC1, STMC1, LNC2, TFC2) in appropriate format.


## Controllable IR and Downstream Experiments

 • ```model_editing_experiment_MAIN.py```: Replicates experiments in Section 4.4 IDF in the Embedding Matrix, Causal Experiment.

 • ```forbidden.py```: Replicates Experiments in Controllable IR and Appendix F.1 Mitigating Adversarial Attacks. We construct a datasets of unsafe words using LDNOOBW and use model-editing method proposed in Section 4.4 erase” the effect of the dangerous token by reducing its importance.

 • ```model_editing_finetune_miniLM.ipynb```: Replicates Experiments in Controllable IR and Appendix F.2 Parameter Efficient Fine-Tuning.



## Linear Approximation Experiments

(Note: Some scripts listed here are not yet uploaded)

 • ```linear_approximation_BM25.ipynb```: reproduces results for Section 5 "Validation of BM25-like Computation", which we formally validate that Relevance Scoring Heads combine soft-TF and IDF in a BM25 manner to compute the final relevance score.

 • ```linear_approximation_beir.py```: Reproduces experiments in Section 5.2 Appendix E for testing the linear model’s generalizability to unseen BeIR datasets and varying query lengths.

Note: Before running the experiment, you need to download the BeIR Datasets:

```
dataset = "scifact"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
out_dir = "beir_datasets"
os.makedirs(out_dir, exist_ok=True)
data_path = util.download_and_unzip(url, out_dir)
```

Additional Details for BeIR: https://github.com/beir-cellar/beir



## How to Cite Us

```
@misc{lu2025crossencoderrediscoverssemanticvariant,
      title={Cross-Encoder Rediscovers a Semantic Variant of BM25}, 
      author={Meng Lu and Catherine Chen and Carsten Eickhoff},
      year={2025},
      eprint={2502.04645},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2502.04645}, 
}
```
