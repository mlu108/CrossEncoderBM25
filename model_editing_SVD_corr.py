"""
This code deconstructs the W_E matrix using SVD
and finds the pearson correlation of U matrices to the idf of the training dataset
"""
import pandas as pd
import torch
import numpy as np
import pandas as pd

import os
from tqdm import tqdm
import random

from functools import partial
import TransformerLens.transformer_lens.utils as utils
from TransformerLens.transformer_lens import patching
from jaxtyping import Float

import plotly.express as px
import plotly.io as pio


from CrossEncoderBM25.helpers import (
    load_json_file,
    load_tokenizer_and_models,
)
selected_query_terms = {"1089763": "miners", "1089401": "tsca", "1088958": "cadi", "1088541": "fletcher,nc", "1088475": "holmes,ny", "1101090": "azadpour", "1088444": "kashan", "1085779": "canopius", "1085510": "carewell", "1085348": "polson", "1085229": "wendelville", "1100499": "trematodiases", "1100403": "arcadis", "1064808": "acantholysis", "1100357": "ardmore", "1062223": "animsition", "1058515": "cladribine", "1051372": "cineplex", "1048917": "misconfiguration", "1045135": "wellesley", "1029552": "tosca", "1028752": "watamote", "1099761": "ari", "1020376": "amplicons", "1002940": "iheartradio", "1000798": "alpha", "992257": "desperation", "197024": "greenhorns", "61277": "brat", "44072": "chatsworth", "195582": "dammam", "234165": "saluki", "196111": "gorm", "329958": "pesto", "100020": "cortana", "193866": "izzam", "448976": "potsherd", "575616": "ankole", "434835": "konig", "488676": "retinue", "389258": "hughes", "443081": "lotte", "511367": "nfcu", "212477": "ouachita", "544060": "dresden", "428773": "wunderlist", "478295": "tigard", "610132": "neodesha", "435412": "lakegirl", "444350": "mageirocophobia", "492988": "saptco", "428819": "swegway", "477286": "antigonish", "478054": "paducah", "1094996": "tacko", "452572": "mems", "20432": "aqsarniit", "559709": "plectrums", "748935": "fraenulum?", "482666": "defdinition", "409071": "ecpi", "1101668": "denora", "537995": "cottafavi", "639084": "hortensia", "82161": "windirstat", "605651": "emmett", "720013": "arzoo", "525047": "trumbull", "978802": "browerville", "787784": "provocative", "780336": "orthorexia", "1093438": "lickspittle", "788851": "qualfon", "61531": "campagnolo", "992652": "setaf", "1092394": "msdcf", "860942": "viastone", "863187": "wintv", "1092159": "northwoods", "990010": "paihia", "840445": "prentice-hall", "775355": "natamycin", "986325": "lapham", "1091654": "parisian", "768411": "mapanything?", "194724": "gesundheit", "985905": "sentral", "1091206": "putrescine", "760930": "islet", "1090945": "ryder", "1090839": "bossov", "1090808": "semispinalis", "774866": "myfortic", "820027": "lithotrophy", "798967": "spredfast", "126821": "scooped", "60339": "stroganoff", "1090374": "strategery", "180887": "enu", "292225": "molasses"}

fbase_path = ""
tfc1_add_queries = pd.read_csv(os.path.join(fbase_path, "tfc1_add_qids_with_text.csv"), header=None, names=["_id", "text"])
tfc1_queries_dict = tfc1_add_queries.set_index('_id').to_dict(orient='index')
tfc1_add_baseline_corpus = load_json_file(os.path.join(fbase_path, "tfc1_add_baseline_final_dd_append_corpus.json"))["corpus"]
target_qids = tfc1_add_queries["_id"].tolist() #[448976] 
tfc1_add_queries = tfc1_add_queries[tfc1_add_queries["_id"].isin(target_qids)] #tfc remains unchanged
tfc1_add_dd_corpus = load_json_file(os.path.join(fbase_path, f"tfc1_add_append_final_dd_corpus.json"))["corpus"]
pre_trained_model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"
torch.set_grad_enabled(False)
device = utils.get_device()
tokenizer, tl_model, pooler_layer,dropout_layer,classifier_layer = load_tokenizer_and_models(pre_trained_model_name, device)

import torch

W_E = tl_model.W_E
# Perform SVD
U, S, Vt = torch.linalg.svd(W_E, full_matrices=False)
print(U.shape)

print(S.shape)
print(Vt.shape)

# Check reconstruction
reconstructed_W_E = U @ torch.diag(S) @ Vt
print("Difference between original and reconstructed:", torch.norm(W_E - reconstructed_W_E))

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
idf_matrix_row = idf_matrix.squeeze().to(W_E.device)


import torch
idf_matrix_row = idf_matrix.squeeze().to(W_E.device)
for k in range(50):
    U_k = U[:20000, k].unsqueeze(1)  # First 25000 rows of U's first column
    S_k = S[k]  # First singular value
    Vt_k = Vt[k, :20000].unsqueeze(0)  # First 25000 columns of Vt's first row

    # Calculate uv and sum it across the second dimension
    uv = U_k @ Vt_k
    uv_sum = uv.sum(1)  # Sum along the second dimension (axis=1)

    # Create a mask to filter out the entries where idf_matrix_row is zero
    idf_matrix_row_subset = idf_matrix_row[:20000]  # Subset to match dimensions
    non_zero_mask = idf_matrix_row_subset != 0

    # Apply the mask to both uv_sum and idf_matrix_row
    uv_sum_non_zero = uv_sum[non_zero_mask]
    idf_matrix_row_non_zero = idf_matrix_row_subset[non_zero_mask]

    # Calculate Pearson correlation for non-zero entries
    correlation_matrix = torch.corrcoef(torch.stack((uv_sum_non_zero, idf_matrix_row_non_zero)))
    pearson_correlation = correlation_matrix[0, 1]  # Extract correlation coefficient

    print(f"Pearson Correlation between uv_sum and non-zero IDF Matrix Row: {pearson_correlation.item()}")