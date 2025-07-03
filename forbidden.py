"""
This code contains our preliminary experiment on adversarial attack mitigation using IDF editing
"""

import pandas as pd
import numpy as np
import torch
import csv

import pandas as pd
import numpy as np
#obtain the activations for layer_head_list
#get their attention pattern for each posn, for each
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


from helpers import (
    load_json_file,
    load_tokenizer_and_models,
)

pre_trained_model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"
torch.set_grad_enabled(False)
device = utils.get_device()
tokenizer, tl_model, pooler_layer,dropout_layer,classifier_layer = load_tokenizer_and_models(pre_trained_model_name, device)

# Replace 'idf.tsv' with the path to your IDF TSV file
idf_file_path = 'msmarco_idf.tsv'
# Load the IDF data into a DataFrame
idf_df = pd.read_csv(idf_file_path, sep='\t', header=None, names=['word', 'idf'])
# Convert the DataFrame to a dictionary for quick lookup
idf_dict = pd.Series(idf_df.idf.values, index=idf_df.word).to_dict()

def compute_NDCGf(MODE=None,SCALE=None): 
    # Define relevance scores for the labels
    label_scores = {
        'F': -10,  # Forbidden
        'S': 1,    # Somewhat Relevant
        'P': 2     # Perfectly Relevant
    }

    # Load the CSV data into a pandas DataFrame
    csv_file_path = 'model_editing/forbidden/ltrf-cqa-dataset.csv'
    df = pd.read_csv(csv_file_path)

    # Function to calculate DCG@k
    def dcg_at_k(relevance_scores, k):
        relevance_scores = np.array(relevance_scores)[:k]
        discounts = np.log2(np.arange(2, len(relevance_scores) + 2))
        return np.sum(relevance_scores / discounts)

    # Function to calculate Ideal DCG@k
    def ideal_dcg_at_k(relevance_scores, k):
        ideal_relevance_scores = sorted(relevance_scores, reverse=True)
        return dcg_at_k(ideal_relevance_scores, k)

    # Function to calculate Worst DCG@k
    def worst_dcg_at_k(relevance_scores, k):
        worst_relevance_scores = sorted(relevance_scores)
        return dcg_at_k(worst_relevance_scores, k)

    # Function to calculate nDCG_f based on the given formula
    def ndcg_f(relevance_scores, k):
        if k == 1:
            # Single-document query handling
            if relevance_scores[0] == label_scores['F']:
                return 0  # Penalize if the document is forbidden
            else:
                return 1  # Perfect score if the document is relevant
        else:
            # For multi-document queries, proceed with normal nDCG_f calculation
            wk = dcg_at_k(relevance_scores, k)
            ik_f = ideal_dcg_at_k(relevance_scores, k)
            wk_f = worst_dcg_at_k(relevance_scores, k)
            print(f"wk:{wk},ik_f:{ik_f},wk_f:{wk_f}")
            
            if ik_f > wk_f:
                result = (wk - wk_f) / (ik_f - wk_f)
            else:
                result = 0  # Return 0 if Ik,f is not greater than Wk,f
            print(f'NDCGf result: {result}')
            return result

    # Adjust to handle each question ID separately
    results = {}

    # Group the dataframe by question_id
    grouped = df.groupby('question_id')

    # Store individual nDCG_f values
    ndcg_f_values = []

    # Process each group (set of documents per question_id)
    for question_id, group in grouped:
        if len(group) == 1:
            # Handle single-document queries
            #print(f"Single document query for question_id: {question_id}")
            relevance_scores = [label_scores[label] for label in group['label']]
            ndcg_f_value = ndcg_f(relevance_scores, 1)  # Single document k=1
        else:
            #print(f"Multiple document query for question_id: {question_id}")
            relevance_scores = [label_scores[label] for label in group['label']]
            # Sort the group by crossencoder score (descending)
            if MODE==None or SCALE==None:
                by_col = 'crossencoder_score'
            else:
                by_col = f'crossencoder_score_{MODE}_{SCALE}'
            group_sorted = group.sort_values(by=by_col, ascending=False)
            k = len(group_sorted)
            ndcg_f_value = ndcg_f(relevance_scores, k)
        
        # Store the result for this question
        results[question_id] = ndcg_f_value
        ndcg_f_values.append(ndcg_f_value)


    # Calculate the overall nDCG_f (average nDCG_f across all queries)
    overall_ndcg_f = np.mean(ndcg_f_values)
    print(f"Overall nDCG_f: {overall_ndcg_f:.4f}")
    with open('model_editing/forbidden/log.txt', 'a') as f:
        f.write(MODE,SCALE)
        f.write(f"Overall nDCG_f: {overall_ndcg_f:.4f}")


# Step 1: Read words from obsenewords.txt
with open('model_editing/forbidden/obsenewords.txt', 'r') as f:
    obsenewords = [line.strip() for line in f.readlines()]

# Step 2: Read the "ngram" column from refined_ngram_dict.csv
ngram_df = pd.read_csv('model_editing/forbidden/refined_ngram_dict.csv')
ngrams = ngram_df['ngram'].tolist()

# Step 3: Combine both lists
unsafe_words = obsenewords + ngrams

# Tokenize unsafe words
tokenized_unsafe_words = tokenizer([' '.join(unsafe_words)], ['a'], return_tensors="pt", padding=True, truncation=True)
index102 = tokenized_unsafe_words['input_ids'][0].tolist().index(102)
tokenized_unsafe_words = tokenized_unsafe_words['input_ids'][0][1:index102].tolist()

# Load the original dataset
csv_file_path = 'model_editing/forbidden/ltrf-cqa-dataset.csv'
df = pd.read_csv(csv_file_path)

# Loop through modes and scales
for MODE, SCALE in [('decrease', 15), ('increase', 15), ('decrease', 30), ('increase', 30), ('decrease', 5), ('increase', 5), ('decrease', 13), ('increase', 13), ('decrease', 40), ('increase', 40),('decrease', 60), ('increase', 60),('decrease', 100), ('increase', 100)]:
    W_E = tl_model.W_E
    # Perform SVD
    U, S, Vt = torch.linalg.svd(W_E, full_matrices=False)
    orig_U_0 = U[:, 0].clone()

    # Modify U[:, 0] for the unsafe words
    for id in tokenized_unsafe_words:
        if U[:, 0][id] > 0:
            if MODE == 'increase':
                orig_U_0[id] = U[:, 0][id] * -SCALE
            else:
                orig_U_0[id] = U[:, 0][id] * SCALE
        else:
            if MODE == 'increase':
                orig_U_0[id] = U[:, 0][id] * SCALE
            else:
                orig_U_0[id] = U[:, 0][id] * -SCALE

    U[:, 0] = orig_U_0
    reconstructed_W_E = U @ torch.diag(S) @ Vt
    np.save('syn_reconstructed_U_sec_high_30.npy', reconstructed_W_E.cpu().numpy())
    print("Difference between original and reconstructed:", torch.norm(W_E - reconstructed_W_E))

    # Step 1: Add new columns to the dataframe
    new_column_name = f'crossencoder_score_{MODE}_{SCALE}'
    df[new_column_name] = None  # Create a new column in the DataFrame

    # Step 2: Process each row and calculate scores
    for index, row in df.iterrows():
        question_id = row['question_id']
        question_text = row['question_text']
        answer_text = row['answer_text']
        try:
            # Tokenize the question-answer pair
            tokenized_pair_baseline = tokenizer([question_text], [answer_text], return_tensors="pt", padding=True, truncation=True)
            #print(question_text,answer_text)
            # Run model and get embeddings
            baseline_outputs, _ = tl_model.run_with_cache(
                tokenized_pair_baseline["input_ids"],
                return_type="embeddings",
                one_zero_attention_mask=tokenized_pair_baseline["attention_mask"],
                token_type_ids=tokenized_pair_baseline['token_type_ids']
            )
            
            # Compute crossencoder score
            score = classifier_layer(dropout_layer(pooler_layer(baseline_outputs)))

            # Assign the computed score to the new column
            df.at[index, new_column_name] = score.item()  # Make sure to use .item() to convert torch tensor to a number

        # Debug print for tracking progress
        except ValueError as e:
            pass

    # Step 3: Save the updated DataFrame back to the CSV
    df.to_csv(csv_file_path, index=False)  # Overwrite the CSV with new columns added

    # Step 4: Compute NDCGf
    compute_NDCGf(MODE=MODE, SCALE=SCALE)
    print("Computed NDCGf.")