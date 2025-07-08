import torch
import numpy as np
import pandas as pd
import math
import os
from tqdm import tqdm
import random
import json
from functools import partial
import TransformerLens.transformer_lens.utils as utils
from TransformerLens.transformer_lens import patching
from jaxtyping import Float

import plotly.express as px
import plotly.io as pio

from helpers import (
    load_json_file,
    load_tokenizer_and_models,
)

from patching_helpers import (
    get_activations,
    get_act_patch_attn_head_out_all_pos,
    patch_specific_head_attn_z_all_pos,
    patch_specific_multi_heads_attn_z_all_pos
)
import numpy as np
import csv
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from itertools import product
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
import numpy as np
import random

pre_trained_model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"
torch.set_grad_enabled(False)
device = utils.get_device()
tokenizer, tl_model, pooler_layer,dropout_layer,classifier_layer = load_tokenizer_and_models(pre_trained_model_name, device)
selected_query_terms = {"1089763": "miners", "1089401": "tsca", "1088958": "cadi", "1088541": "fletcher,nc", "1088475": "holmes,ny", "1101090": "azadpour", "1088444": "kashan", "1085779": "canopius", "1085510": "carewell", "1085348": "polson", "1085229": "wendelville", "1100499": "trematodiases", "1100403": "arcadis", "1064808": "acantholysis", "1100357": "ardmore", "1062223": "animsition", "1058515": "cladribine", "1051372": "cineplex", "1048917": "misconfiguration", "1045135": "wellesley", "1029552": "tosca", "1028752": "watamote", "1099761": "ari", "1020376": "amplicons", "1002940": "iheartradio", "1000798": "alpha", "992257": "desperation", "197024": "greenhorns", "61277": "brat", "44072": "chatsworth", "195582": "dammam", "234165": "saluki", "196111": "gorm", "329958": "pesto", "100020": "cortana", "193866": "izzam", "448976": "potsherd", "575616": "ankole", "434835": "konig", "488676": "retinue", "389258": "hughes", "443081": "lotte", "511367": "nfcu", "212477": "ouachita", "544060": "dresden", "428773": "wunderlist", "478295": "tigard", "610132": "neodesha", "435412": "lakegirl", "444350": "mageirocophobia", "492988": "saptco", "428819": "swegway", "477286": "antigonish", "478054": "paducah", "1094996": "tacko", "452572": "mems", "20432": "aqsarniit", "559709": "plectrums", "748935": "fraenulum?", "482666": "defdinition", "409071": "ecpi", "1101668": "denora", "537995": "cottafavi", "639084": "hortensia", "82161": "windirstat", "605651": "emmett", "720013": "arzoo", "525047": "trumbull", "978802": "browerville", "787784": "provocative", "780336": "orthorexia", "1093438": "lickspittle", "788851": "qualfon", "61531": "campagnolo", "992652": "setaf", "1092394": "msdcf", "860942": "viastone", "863187": "wintv", "1092159": "northwoods", "990010": "paihia", "840445": "prentice-hall", "775355": "natamycin", "986325": "lapham", "1091654": "parisian", "768411": "mapanything?", "194724": "gesundheit", "985905": "sentral", "1091206": "putrescine", "760930": "islet", "1090945": "ryder", "1090839": "bossov", "1090808": "semispinalis", "774866": "myfortic", "820027": "lithotrophy", "798967": "spredfast", "126821": "scooped", "60339": "stroganoff", "1090374": "strategery", "180887": "enu", "292225": "molasses"}
fbase_path = ""
tfc1_add_queries = pd.read_csv(os.path.join(fbase_path, "tfc1_add_qids_with_text.csv"), header=None, names=["_id", "text"])
tfc1_queries_dict = tfc1_add_queries.set_index('_id').to_dict(orient='index')
tfc1_add_baseline_corpus = load_json_file(os.path.join(fbase_path, "tfc1_add_baseline_final_dd_append_corpus.json"))["corpus"]
target_qids = tfc1_add_queries["_id"].tolist() #[448976] 
tfc1_add_queries = tfc1_add_queries[tfc1_add_queries["_id"].isin(target_qids)] #tfc remains unchanged
tfc1_add_dd_corpus = load_json_file(os.path.join(fbase_path, f"tfc1_add_append_final_dd_corpus.json"))["corpus"]
def BM_score_normal(tokenized_pair_original, doc_token_len, first_SEP_token, L, k=0.5, b=0.9, idf_matrix_row=None):
    """
    Calculate the BM25 score between a query and a document.
    """
    #query_ids = tokenized_pair_original["input_ids"][0][1:6].tolist()
    #doc_ids = tokenized_pair_original["input_ids"][0][7:-1].tolist()
    # Extract query and document tokens
    query_ids = tokenized_pair_original["input_ids"][0][1:first_SEP_token].tolist()
    doc_ids = tokenized_pair_original["input_ids"][0][first_SEP_token+1:-1].tolist() # [6+1:-1] adjusted for simpler indexing
    
    # Calculate BM25 score
    bm25_score = 0.0
    for query_id in query_ids:
        # Directly count occurrences of the query term in the document
        tf = doc_ids.count(query_id)
        #print(f"Term Frequency (TF) for token {query_id}: {tf}")
        
        if tf == 0:
            continue  # Skip terms that don't appear in the document
        
        # Retrieve IDF for the current query term
        idf = (
                    idf_matrix_row[query_id].item()
                    if idf_matrix_row[query_id] != 0
                    else math.log((1 + len(doc_ids)) / (1 + tf)) + 1
                )
        
        # BM25 term score calculation
        term_score = idf * (tf * (k + 1)) / (tf + k * (1 - b + b * (doc_token_len / L)))
        bm25_score += term_score

    return bm25_score
import torch
import numpy as np
from tqdm import tqdm
idf_file_path = 'msmarco_idf.tsv'
# Load the IDF data into a DataFrame
idf_df = pd.read_csv(idf_file_path, sep='\t', header=None, names=['word', 'idf'])
# Convert the DataFrame to a dictionary for quick lookup
idf_dict = pd.Series(idf_df.idf.values, index=idf_df.word).to_dict()
idf_list = []

for i in range(tl_model.cfg.d_vocab):
    token =tokenizer.decode(i)

    #print(token)
    idf = idf_dict.get(token)
    if idf!=None:
        idf_list.append(idf)
        
    else:
        idf_list.append(0)

idf_matrix = torch.tensor(idf_list)

print(idf_matrix.shape)
idf_matrix_row = idf_matrix.squeeze().to(tl_model.W_E.device)

W_E = tl_model.W_E
U, S, Vt = torch.linalg.svd(W_E, full_matrices=False)
U_0 = U[:, 0]


#I want to align U_0 values with the distribution of idf_matrix_row, same mean same standard deviation
# Compute mean and standard deviation of idf_matrix_row
idf_mean = idf_matrix_row.mean()
idf_std = idf_matrix_row.std()

# Compute mean and standard deviation of U_0
U_0_mean = U_0.mean()
U_0_std = U_0.std()

# Apply linear transformation to U_0 to align with idf_matrix_row
U0_idf = (U_0 - U_0_mean) / U_0_std * idf_std + idf_mean

# Check the result
print(f"U_0 aligned mean: {U0_idf.mean().item()}, std: {U0_idf.std().item()}")
print(f"IDF mean: {idf_mean.item()}, IDF std: {idf_std.item()}")


# Constants
#ALL_MATCHING_HEADS = [(1, 7), (5, 9), (7, 9), (8, 5), (8, 3), (8, 0), (3, 1), (2, 1), (7, 8), (4, 9), (7, 2), (7, 4), (7, 3)]
layer_head_strings = ['7.9', '8.1', '4.9', '2.1', '8.8', '6.5', '1.7', '5.7', '5.9', '3.1', '8.0', '6.3', '0.8']
REAL_MATCHING_HEADS = ALL_MATCHING_HEADS = [tuple(map(int, lh.split('.'))) for lh in layer_head_strings]
#OUTPUT_DIR = "MatchingHeads/no_cutoff_BMrerank"
OUTPUT_DIR = "MatchingHeads/BMrerank_saveALL"

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Process datasets
for DATASET in [
    'webis-touche2020', 'arguana', 'climate-fever', 'fever', 'fiqa',
    'hotpotqa', 'nfcorpus', 'nq', 'quora', 'scidocs', 'scifact', 'trec-covid'
]:
    print(f"Processing dataset: {DATASET}")
    output_rows = []
    features_by_query_len = {}
    scores_by_query_len = {}
    BM_scores = {}

    # File paths
    test_tsv_path = f"beir_datasets/BM25_RESULTS/bm25_results_{DATASET}.csv"
    
    corpus_jsonl_path = f"beir_datasets/{DATASET}/corpus.jsonl"
    queries_jsonl_path = f"beir_datasets/{DATASET}/queries.jsonl"

    # Load data
    #query_id,doc_id
    test_data = pd.read_csv(test_tsv_path, 
                        sep=",", 
                        header=None, 
                        names=["query_id", "doc_id", "score"],
                        dtype={"query_id": str, "doc_id": str, "score": str},)
    with open(corpus_jsonl_path, "r") as corpus_file:
        corpus_data = {entry["_id"]: entry for entry in map(json.loads, corpus_file)}
    with open(queries_jsonl_path, "r") as queries_file:
        queries_data = {entry["_id"]: entry for entry in map(json.loads, queries_file)}

    # 4. Group test_data by query_id, then process top 10 for each group
    grouped_test_data = test_data.groupby("query_id", sort=False)

    for orig_rank, (query_id, group_df) in enumerate(grouped_test_data):
        # Take the top 10 rows for this query_id
        top_10 = group_df.head(10)
        
        # Now iterate over the top 10 doc_id rows
        for _, row in top_10.iterrows():
            corpus_id = str(row["doc_id"])
            query = queries_data.get(query_id, {}).get("text", None)
            perturbed_doc = corpus_data.get(corpus_id, {}).get("text", None)
            if query is not None and perturbed_doc is not None:
                tokenized_pair_original = tokenizer([query], [perturbed_doc], return_tensors="pt", padding=True, truncation=True)
                input_list = tokenized_pair_original['input_ids'][0].tolist()
                fir_SEP_position = input_list.index(102)
                doc_token_len = len(input_list[fir_SEP_position + 1:-1])
                query_len = fir_SEP_position - 1

                # Compute features
                feature_list = []
                
                # Prepare activation names based on ALL_MATCHING_HEADS
                names_list = [utils.get_act_name('pattern', layer) for layer, head in ALL_MATCHING_HEADS]
                orig_outputs, act_orig = tl_model.run_with_cache(
                    tokenized_pair_original["input_ids"],
                    return_type="embeddings",
                    one_zero_attention_mask=tokenized_pair_original["attention_mask"],
                    token_type_ids=tokenized_pair_original['token_type_ids'],
                    names_filter=lambda name: name in names_list,
                )
                for tok_index in range(1, query_len + 1):
                    ft1 = idf_tok = -U0_idf[input_list[tok_index]].item()
                    feature_list.append(ft1)
                    for layer, head in ALL_MATCHING_HEADS:
                        pattern_name = utils.get_act_name('pattern', layer)
                        pattern_orig = act_orig[pattern_name][0, head, :, :].cpu().numpy()
                        other_tok_indices = [t for t in range(fir_SEP_position + 1, len(tokenized_pair_original["input_ids"][0])) if t != tok_index]
                        attention_scores_on_other_tokens = pattern_orig[tok_index, other_tok_indices].sum()
                        ft2 = attention_scores_on_other_tokens
                        ft3 = ft1 * ft2
                        feature_list.extend([ft2, ft3])
                

                model_score = classifier_layer(dropout_layer(pooler_layer(orig_outputs))).item()
                # Add features and scores to the respective query_len group
                #features_by_query_len.setdefault(query_len, []).append(feature_list)
                #scores_by_query_len.setdefault(query_len, []).append(model_score)
                BM_score_result = BM_score_normal(tokenized_pair_original, doc_token_len, fir_SEP_position,L = 84.32, idf_matrix_row=idf_matrix_row)
                #print(BM_score_result)
                #BM_scores.setdefault(query_len, []).append(BM_score_result)
                # Append to output list
                output_rows.append({
                    "query_id": query_id,
                    "doc_id": corpus_id,
                    "orig_rank": orig_rank,
                    "query_len": query_len,
                    "model_score": model_score,
                    "BM_score_result": BM_score_result, 
                    "feature_list": str(feature_list),
                })
    output_df = pd.DataFrame(output_rows)
    output_csv_path = f"bm25_results_{DATASET}.csv"
    output_df.to_csv(output_csv_path, index=False)
    print(f"Output for dataset '{DATASET}' saved to {output_csv_path}")
    """
    # Save features and scores grouped by query_len
    for query_len, feature_group in features_by_query_len.items():
        feature_array = np.array(feature_group)
        score_array = np.array(scores_by_query_len[query_len])
        BM_scores_array = np.array(BM_scores[query_len])
        print(np.array(BM_scores))
        print(BM_scores_array)

        # Normalize features
        means = feature_array.mean(axis=0)
        stds = feature_array.std(axis=0)
        stds[stds == 0] = 1  # Avoid division by zero
        normalized_features = (feature_array - means) / stds

        # Save files
        query_len_dir = os.path.join(OUTPUT_DIR, DATASET)
        os.makedirs(query_len_dir, exist_ok=True)
        np.save(os.path.join(query_len_dir, f"features_{query_len}.npy"), normalized_features)
        np.save(os.path.join(query_len_dir, f"model_scores_{query_len}.npy"), score_array)
        np.save(os.path.join(query_len_dir, f"BM_scores_{query_len}.npy"), BM_scores_array)
    
    print(f"Finished processing dataset: {DATASET}")
    """



