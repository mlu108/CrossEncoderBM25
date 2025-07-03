"""
This code checks 
(1) whether updating the IDF of a token increases the document that 
contains this token to be higher, and the reverse trend.
        
(2) It also examines gets the attention pattern for each Semantic Scoring Heads
and see if the idf change induces the expected change in these heads' attention value on the token
--> the result shows expected trend 

(3) model_editing_full (commented out) is establishing the baseline that when increasing/decreasing token id's value
in the W_E doesn't get you the expected result

NOTE: how exactly is this done is through (janky right now for just experiment's sake):
we save the reconstructed W_E each time after editing some tokens' idf into a local .npy file
np.save('syn_reconstructed_U_sec_high_30.npy', reconstructed_W_E.cpu().numpy())

And we edit the transformer_lens components.py Embed class to let tl_model also embeds with the
reconstructed W_E instead of the original W_E
class Embed(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.W_E: Float[torch.Tensor, "d_vocab d_model"] = nn.Parameter(
            torch.empty(self.cfg.d_vocab, self.cfg.d_model, dtype=cfg.dtype)
        )
        # Some models (e.g. Bloom) need post embedding layer norm
        if cfg.post_embedding_ln:
            self.ln = LayerNorm(cfg)

    def forward(
        self, tokens: Int[torch.Tensor, "batch pos"]
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        # If A has shape [a, b] and B has shape [c, d], then A[:, B] has shape [a, c, d]
        # B acts as a tensor of indices into the second dimension (so >=0 and <b)
        U, S, Vt = torch.linalg.svd(self.W_E, full_matrices=False)
        #print("Singular values:", S)
        #for i in range(1):
        #    S[i]=0
        U_0_min = U[:, 0].min()
        U_0_max = U[:, 0].max()
        U[:, 0]= torch.ones(self.cfg.d_vocab).to(self.cfg.device)*U_0_min#0#0.08

        reconstructed_W_E = U @ torch.diag(S) @ Vt

        loaded_W_E = np.load('syn_reconstructed_U_sec_high_30.npy')
        reconstructed_W_E = torch.from_numpy(loaded_W_E).to(self.cfg.device)
            
    
"""
import pandas as pd
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import os
from tqdm import tqdm
import random
import csv
from functools import partial
import TransformerLens.transformer_lens.utils as utils
from TransformerLens.transformer_lens import patching
from jaxtyping import Float

import plotly.express as px
import plotly.io as pio
random.seed(36)

from helpers import (
    load_json_file,
    load_tokenizer_and_models,
    get_type_pos_dict
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

idf_file_path = 'msmarco_idf.tsv'
# Load the IDF data into a DataFrame
idf_df = pd.read_csv(idf_file_path, sep='\t', header=None, names=['word', 'idf'])
# Convert the DataFrame to a dictionary for quick lookup
idf_dict = pd.Series(idf_df.idf.values, index=idf_df.word).to_dict()
idf_list = []
use_reduced_dataset = True #NOTE: for prepend we will use reduced dataset to look at whether the general pattern accords 
perturb_type ="append"#"append"
n_queries = 50
n_docs = 30

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
selected_query_terms = {"1089763": "miners", "1089401": "tsca", "1088958": "cadi", "1088541": "fletcher,nc", "1088475": "holmes,ny", "1101090": "azadpour", "1088444": "kashan", "1085779": "canopius", "1085510": "carewell", "1085348": "polson", "1085229": "wendelville", "1100499": "trematodiases", "1100403": "arcadis", "1064808": "acantholysis", "1100357": "ardmore", "1062223": "animsition", "1058515": "cladribine", "1051372": "cineplex", "1048917": "misconfiguration", "1045135": "wellesley", "1029552": "tosca", "1028752": "watamote", "1099761": "ari", "1020376": "amplicons", "1002940": "iheartradio", "1000798": "alpha", "992257": "desperation", "197024": "greenhorns", "61277": "brat", "44072": "chatsworth", "195582": "dammam", "234165": "saluki", "196111": "gorm", "329958": "pesto", "100020": "cortana", "193866": "izzam", "448976": "potsherd", "575616": "ankole", "434835": "konig", "488676": "retinue", "389258": "hughes", "443081": "lotte", "511367": "nfcu", "212477": "ouachita", "544060": "dresden", "428773": "wunderlist", "478295": "tigard", "610132": "neodesha", "435412": "lakegirl", "444350": "mageirocophobia", "492988": "saptco", "428819": "swegway", "477286": "antigonish", "478054": "paducah", "1094996": "tacko", "452572": "mems", "20432": "aqsarniit", "559709": "plectrums", "748935": "fraenulum?", "482666": "defdinition", "409071": "ecpi", "1101668": "denora", "537995": "cottafavi", "639084": "hortensia", "82161": "windirstat", "605651": "emmett", "720013": "arzoo", "525047": "trumbull", "978802": "browerville", "787784": "provocative", "780336": "orthorexia", "1093438": "lickspittle", "788851": "qualfon", "61531": "campagnolo", "992652": "setaf", "1092394": "msdcf", "860942": "viastone", "863187": "wintv", "1092159": "northwoods", "990010": "paihia", "840445": "prentice-hall", "775355": "natamycin", "986325": "lapham", "1091654": "parisian", "768411": "mapanything?", "194724": "gesundheit", "985905": "sentral", "1091206": "putrescine", "760930": "islet", "1090945": "ryder", "1090839": "bossov", "1090808": "semispinalis", "774866": "myfortic", "820027": "lithotrophy", "798967": "spredfast", "126821": "scooped", "60339": "stroganoff", "1090374": "strategery", "180887": "enu", "292225": "molasses"}

#updating the SVD dictionary
import itertools
for FIRST_TOKEN_POS, SEC_TOKEN_POS in [(0,0)]:#just for visualization code to stay the same so keeping the same data folder structure
#FIRST_TOKEN_POS, SEC_TOKEN_POS = 1,4
    modes = ['increase','decrease']#['increase','decrease']
    scales = [300,400,500,180,250]#[1,3,5,8,10,15,25]#[1,5,10,15,20,30]#[x for x in range(1, 50) if x not in {1, 2, 3, 5, 8, 10, 20, 30, 50}]
    #scales = [80,90,100,120,140,160,200]

    #MODEL_EDITING_SVD
    for MODE, SCALE in itertools.product(modes, scales):
        print((MODE, SCALE))

        W_E = tl_model.W_E
        # Perform SVD
        U, S, Vt = torch.linalg.svd(W_E, full_matrices=False)
        orig_U_0 = U[:,0].clone().detach() 

        for i, qid in enumerate(tqdm(target_qids)):
            query = tfc1_queries_dict[qid]['text']
            target_docs = tfc1_add_dd_corpus[str(qid)]

            for j, doc_id in enumerate(target_docs):
               
                perturbed_doc = tfc1_add_dd_corpus[str(qid)][doc_id]["text"]
                tokenized_pair_perturbed = tokenizer([query],[perturbed_doc], return_tensors="pt",padding=True, truncation=True)
                decoded_tokens = [tokenizer.decode(tok) for tok in tokenized_pair_perturbed["input_ids"][0]]
                new_text_array = [d+' '+str(i) for i,d in enumerate(decoded_tokens)]
                type_pos_dict = get_type_pos_dict(new_text_array,qid,mode='append')
                Q_plus_pos_tuple = type_pos_dict['Qplus']
                selected_token_vocab_ids = []
                try:
                    for pos in range(Q_plus_pos_tuple[0][0],Q_plus_pos_tuple[0][1]+1):
                            selected_token_vocab_ids.append(tokenized_pair_perturbed["input_ids"][0][pos].item())
                except IndexError as e:
                    pass
                for token_id in selected_token_vocab_ids:
                    if U[:,0][token_id]>=0 and SCALE!=1:
                        if MODE=='increase':
                            orig_U_0[token_id] =U[:,0][token_id]*-SCALE
                        elif MODE =='decrease':
                            orig_U_0[token_id] =U[:,0][token_id]*SCALE
                    elif U[:,0][token_id]<0 and SCALE!=1:#if <0, want to increase
                        if MODE=='increase':
                            orig_U_0[token_id] =U[:,0][token_id]*SCALE
                        elif MODE=='decrease':
                            orig_U_0[token_id] =U[:,0][token_id]*-SCALE
                   
        U[:,0] = orig_U_0 
       

        # Check reconstruction
        reconstructed_W_E = U @ torch.diag(S) @ Vt
        np.save('syn_reconstructed_U_sec_high_30.npy', reconstructed_W_E.cpu().numpy())
        print("Difference between original and reconstructed:", torch.norm(W_E - reconstructed_W_E))


    
        from patching_helpers import get_activations
        from statistics import stdev
        torch.set_grad_enabled(False)
        #if we do the idf switch # specifically, we will find U_0(A) and then replace that as U_0(B), then we should flip the sore

        # Load queries and docs from files
        fbase_path = ""
        # Load files
        #tfc1_precomputed_scores = pd.read_csv(os.path.join(fbase_path, f"tfc1_add_{perturb_type}_target_qids_scores.csv"))
        tfc1_add_queries = pd.read_csv(os.path.join(fbase_path, "tfc1_add_qids_with_text.csv"), header=None, names=["_id", "text"])
        tfc1_queries_dict = tfc1_add_queries.set_index('_id').to_dict(orient='index')
        #print(tfc1_queries_dict)
        tfc1_add_baseline_corpus = load_json_file(os.path.join(fbase_path, "tfc1_add_baseline_final_dd_append_corpus.json"))["corpus"]
        #tfc1_add_dd_corpus = load_json_file(os.path.join(fbase_path, f"tfc1_add_{perturb_type}_final_dd_corpus.json"))["corpus"]

        torch.set_grad_enabled(False)
        device = utils.get_device()

        target_qids = tfc1_add_queries["_id"].tolist() #[448976] 
        tfc1_add_queries = tfc1_add_queries[tfc1_add_queries["_id"].isin(target_qids)] #tfc remains unchanged

        pre_trained_model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"
        tokenizer, tl_model, pooler_layer,dropout_layer,classifier_layer = load_tokenizer_and_models(pre_trained_model_name, device)


        finished_qids = []
        # Loop through each query and run activation patching
        for i, qid in enumerate(tqdm(target_qids[:1])):
            query = tfc1_queries_dict[qid]['text']
            target_docs = tfc1_add_dd_corpus[str(qid)]
            if use_reduced_dataset:
                random.seed(36)
                target_doc_ids = random.sample(target_docs.keys(), n_docs)
                target_docs = {doc_id: target_docs[doc_id] for doc_id in target_doc_ids}
            for j, doc_id in enumerate(target_docs):
                try:
                    if True:
                        original_doc = tfc1_add_baseline_corpus[str(qid)][doc_id]["text"]
                        perturbed_doc = tfc1_add_dd_corpus[str(qid)][doc_id]["text"]
                        tokenized_pair_baseline = tokenizer([query],[original_doc], return_tensors="pt",padding=True,truncation=True)
                        tokenized_pair_perturbed = tokenizer([query],[perturbed_doc], return_tensors="pt",padding=True, truncation=True)
                        cls_tok = tokenized_pair_baseline["input_ids"][0][0]
                        sep_tok = tokenized_pair_baseline["input_ids"][0][-1]
                        filler_token = tokenizer.encode("a", add_special_tokens=False)[0]
                        b_len = torch.sum(tokenized_pair_baseline["attention_mask"]).item()
                        p_len = torch.sum(tokenized_pair_perturbed["attention_mask"]).item()
                        if b_len != p_len: 
                            adj_n = p_len - b_len
                            filler_tokens = torch.full((adj_n,), filler_token)
                            filler_attn_mask = torch.full((adj_n,), tokenized_pair_baseline["attention_mask"][0][1]) 
                            filler_token_id_mask = torch.full((adj_n,), tokenized_pair_baseline["token_type_ids"][0][-1]) #just filling in the document
                            adj_doc = torch.cat((tokenized_pair_baseline["input_ids"][0][1:-1], filler_tokens))
                            tokenized_pair_baseline["input_ids"] = torch.cat((cls_tok.view(1), adj_doc, sep_tok.view(1)), dim=0).view(1,-1)
                            tokenized_pair_baseline["attention_mask"] = torch.cat((tokenized_pair_baseline["attention_mask"][0], filler_attn_mask), dim=0).view(1,-1)
                            tokenized_pair_baseline["token_type_ids"] = torch.cat((tokenized_pair_baseline["token_type_ids"][0], filler_token_id_mask), dim=0).view(1,-1)
        
                        decoded_tokens = [tokenizer.decode(tok) for tok in tokenized_pair_perturbed["input_ids"][0]]
                        #print(decoded_tokens)
                        new_text_array = [d+' '+str(i) for i,d in enumerate(decoded_tokens)]
                        type_pos_dict = get_type_pos_dict(new_text_array,qid,mode='append')
                        Q_plus_pos_tuple = type_pos_dict['Qplus']
                        #print(Q_plus_pos_tuple)
                        layer_head_list=[(10,1)]
                        names_list=[]
                        for layer in list(set([l[0] for l in layer_head_list])):# the different layer number
                            names_list.append(utils.get_act_name('pattern',layer))
                        act = get_activations(tl_model, tokenized_pair_perturbed,names_list)#[:,head,:,:] 
                                
                        #TODO EDIT: what we are interested is attention on the selected query token!!!
                        for layer_head_index,(layer, head) in enumerate(layer_head_list):
                            name_in_namelist = utils.get_act_name('pattern',layer)
                            pattern = act[name_in_namelist][0,head,:,:].cpu().numpy()#batch=0, head -> [seqQ, seqK]#
                            CLS_attention_on_selected_query=0
                            for position_looking_at in range(Q_plus_pos_tuple[0][0],Q_plus_pos_tuple[0][1]+1):
                                CLS_attention_on_selected_query+=pattern[0,position_looking_at]#.sum() #CHANGE: attention from first class token to this token
                            #print(CLS_attention_on_selected_query)
                            CLS_attention_on_selected_query/=len(range(Q_plus_pos_tuple[0][0],Q_plus_pos_tuple[0][1]+1))
                            #print(CLS_attention_on_selected_query)

                        baseline_outputs,_ = tl_model.run_with_cache(
                            tokenized_pair_baseline["input_ids"],
                            return_type="embeddings", #"embedding"
                            one_zero_attention_mask=tokenized_pair_baseline["attention_mask"],
                            token_type_ids = tokenized_pair_baseline['token_type_ids'] #added this for cross-encoders, need token_type_ids
                        )
                        perturbed_outputs,perturbed_cache = tl_model.run_with_cache(
                            tokenized_pair_perturbed["input_ids"],
                            return_type="embeddings", #embedding
                            one_zero_attention_mask=tokenized_pair_perturbed["attention_mask"],
                            token_type_ids = tokenized_pair_perturbed['token_type_ids']
                        )
                        baseline_score =classifier_layer(dropout_layer(pooler_layer(baseline_outputs)))
                        perturbed_score = classifier_layer(dropout_layer(pooler_layer(perturbed_outputs)))
                        
                        
                        results_dir = f'model_editing/SVD_experiments_perturb/U/{FIRST_TOKEN_POS}_{SEC_TOKEN_POS}/{MODE}/effects'
                        Path(results_dir).mkdir(parents=True, exist_ok=True)

                        csv_filename = os.path.join(results_dir,f'{MODE}_{SCALE}_effects_full.csv')
                        with open(csv_filename, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([baseline_score.item(),perturbed_score.item()])  # Write the scores
                        results_dir = f'model_editing/SVD_experiments_perturb/U/{FIRST_TOKEN_POS}_{SEC_TOKEN_POS}/{MODE}/attention'
                        Path(results_dir).mkdir(parents=True, exist_ok=True)
                        csv_filename = os.path.join(results_dir,f'{MODE}_{SCALE}_attention_full.csv')
                        with open(csv_filename, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([CLS_attention_on_selected_query])  # Write the scores


                except ValueError as e:
                    pass
 
   
    #MODEL_EDITING_FULL
    for MODE, SCALE in itertools.product(modes, scales):
        print((MODE, SCALE))


        reconstructed_W_E = tl_model.W_E.clone()


        

        for i, qid in enumerate(tqdm(target_qids)):
            query = tfc1_queries_dict[qid]['text']
            target_docs = tfc1_add_dd_corpus[str(qid)]

            for j, doc_id in enumerate(target_docs):
               
                perturbed_doc = tfc1_add_dd_corpus[str(qid)][doc_id]["text"]
                tokenized_pair_perturbed = tokenizer([query],[perturbed_doc], return_tensors="pt",padding=True, truncation=True)
                decoded_tokens = [tokenizer.decode(tok) for tok in tokenized_pair_perturbed["input_ids"][0]]
                new_text_array = [d+' '+str(i) for i,d in enumerate(decoded_tokens)]
                type_pos_dict = get_type_pos_dict(new_text_array,qid,mode='append')
                Q_plus_pos_tuple = type_pos_dict['Qplus']
                selected_token_vocab_ids = []
                try:
                    for pos in range(Q_plus_pos_tuple[0][0],Q_plus_pos_tuple[0][1]+1):
                            selected_token_vocab_ids.append(tokenized_pair_perturbed["input_ids"][0][pos].item())
                except IndexError as e:
                    pass
                for token_id in selected_token_vocab_ids:
                    if SCALE!=1 and MODE=='increase':
                        reconstructed_W_E[token_id] =tl_model.W_E[token_id]*(-SCALE)#-5
                    else:
                        reconstructed_W_E[token_id] =tl_model.W_E[token_id]*SCALE#-5
        np.save('syn_reconstructed_U_sec_high_30.npy', reconstructed_W_E.cpu().numpy())



        import csv
        from patching_helpers import get_activations
        from statistics import stdev
        torch.set_grad_enabled(False)
        #if we do the idf switch # specifically, we will find U_0(A) and then replace that as U_0(B), then we should flip the sore

        # Load queries and docs from files
        fbase_path = ""
        # Load files
        #tfc1_precomputed_scores = pd.read_csv(os.path.join(fbase_path, f"tfc1_add_{perturb_type}_target_qids_scores.csv"))
        tfc1_add_queries = pd.read_csv(os.path.join(fbase_path, "tfc1_add_qids_with_text.csv"), header=None, names=["_id", "text"])
        tfc1_queries_dict = tfc1_add_queries.set_index('_id').to_dict(orient='index')
        #print(tfc1_queries_dict)
        tfc1_add_baseline_corpus = load_json_file(os.path.join(fbase_path, "tfc1_add_baseline_final_dd_append_corpus.json"))["corpus"]
        #tfc1_add_dd_corpus = load_json_file(os.path.join(fbase_path, f"tfc1_add_{perturb_type}_final_dd_corpus.json"))["corpus"]

        torch.set_grad_enabled(False)
        device = utils.get_device()

        target_qids = tfc1_add_queries["_id"].tolist() #[448976] 
        tfc1_add_queries = tfc1_add_queries[tfc1_add_queries["_id"].isin(target_qids)] #tfc remains unchanged

        pre_trained_model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"
        tokenizer, tl_model, pooler_layer,dropout_layer,classifier_layer = load_tokenizer_and_models(pre_trained_model_name, device)


        finished_qids = []
        # Loop through each query and run activation patching
        for i, qid in enumerate(tqdm(target_qids[:1])):
            query = tfc1_queries_dict[qid]['text']
            target_docs = tfc1_add_dd_corpus[str(qid)]
            if use_reduced_dataset:
                random.seed(36)
                target_doc_ids = random.sample(target_docs.keys(), n_docs)
                target_docs = {doc_id: target_docs[doc_id] for doc_id in target_doc_ids}
            for j, doc_id in enumerate(target_docs):
                try:
                    if True:
                        original_doc = tfc1_add_baseline_corpus[str(qid)][doc_id]["text"]
                        perturbed_doc = tfc1_add_dd_corpus[str(qid)][doc_id]["text"]
                        tokenized_pair_baseline = tokenizer([query],[original_doc], return_tensors="pt",padding=True,truncation=True)
                        tokenized_pair_perturbed = tokenizer([query],[perturbed_doc], return_tensors="pt",padding=True, truncation=True)
                        cls_tok = tokenized_pair_baseline["input_ids"][0][0]
                        sep_tok = tokenized_pair_baseline["input_ids"][0][-1]
                        filler_token = tokenizer.encode("a", add_special_tokens=False)[0]
                        b_len = torch.sum(tokenized_pair_baseline["attention_mask"]).item()
                        p_len = torch.sum(tokenized_pair_perturbed["attention_mask"]).item()
                        if b_len != p_len: 
                            adj_n = p_len - b_len
                            filler_tokens = torch.full((adj_n,), filler_token)
                            filler_attn_mask = torch.full((adj_n,), tokenized_pair_baseline["attention_mask"][0][1]) 
                            filler_token_id_mask = torch.full((adj_n,), tokenized_pair_baseline["token_type_ids"][0][-1]) #just filling in the document
                            adj_doc = torch.cat((tokenized_pair_baseline["input_ids"][0][1:-1], filler_tokens))
                            tokenized_pair_baseline["input_ids"] = torch.cat((cls_tok.view(1), adj_doc, sep_tok.view(1)), dim=0).view(1,-1)
                            tokenized_pair_baseline["attention_mask"] = torch.cat((tokenized_pair_baseline["attention_mask"][0], filler_attn_mask), dim=0).view(1,-1)
                            tokenized_pair_baseline["token_type_ids"] = torch.cat((tokenized_pair_baseline["token_type_ids"][0], filler_token_id_mask), dim=0).view(1,-1)
        
                        decoded_tokens = [tokenizer.decode(tok) for tok in tokenized_pair_perturbed["input_ids"][0]]
                        #print(decoded_tokens)
                        new_text_array = [d+' '+str(i) for i,d in enumerate(decoded_tokens)]
                        type_pos_dict = get_type_pos_dict(new_text_array,qid,mode='append')
                        Q_plus_pos_tuple = type_pos_dict['Qplus']
                        #print(Q_plus_pos_tuple)
                        layer_head_list=[(10,1)]
                        names_list=[]
                        for layer in list(set([l[0] for l in layer_head_list])):# the different layer number
                            names_list.append(utils.get_act_name('pattern',layer))
                        act = get_activations(tl_model, tokenized_pair_perturbed,names_list)#[:,head,:,:] 
                                
                        #TODO EDIT: what we are interested is attention on the selected query token!!!
                        for layer_head_index,(layer, head) in enumerate(layer_head_list):
                            name_in_namelist = utils.get_act_name('pattern',layer)
                            pattern = act[name_in_namelist][0,head,:,:].cpu().numpy()#batch=0, head -> [seqQ, seqK]#
                            CLS_attention_on_selected_query=0
                            for position_looking_at in range(Q_plus_pos_tuple[0][0],Q_plus_pos_tuple[0][1]+1):
                                CLS_attention_on_selected_query+=pattern[0,position_looking_at]#.sum() #CHANGE: attention from first class token to this token
                            #print(CLS_attention_on_selected_query)
                            CLS_attention_on_selected_query/=len(range(Q_plus_pos_tuple[0][0],Q_plus_pos_tuple[0][1]+1))
                            #print(CLS_attention_on_selected_query)

                        baseline_outputs,_ = tl_model.run_with_cache(
                            tokenized_pair_baseline["input_ids"],
                            return_type="embeddings", #"embedding"
                            one_zero_attention_mask=tokenized_pair_baseline["attention_mask"],
                            token_type_ids = tokenized_pair_baseline['token_type_ids'] #added this for cross-encoders, need token_type_ids
                        )
                        perturbed_outputs,perturbed_cache = tl_model.run_with_cache(
                            tokenized_pair_perturbed["input_ids"],
                            return_type="embeddings", #embedding
                            one_zero_attention_mask=tokenized_pair_perturbed["attention_mask"],
                            token_type_ids = tokenized_pair_perturbed['token_type_ids']
                        )
                        baseline_score =classifier_layer(dropout_layer(pooler_layer(baseline_outputs)))
                        perturbed_score = classifier_layer(dropout_layer(pooler_layer(perturbed_outputs)))
                        
                        
                        results_dir = f'model_editing/SVD_experiments_perturb/full/{FIRST_TOKEN_POS}_{SEC_TOKEN_POS}/{MODE}/effects'
                        Path(results_dir).mkdir(parents=True, exist_ok=True)

                        csv_filename = os.path.join(results_dir,f'{MODE}_{SCALE}_effects_full.csv')
                        with open(csv_filename, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([baseline_score.item(),perturbed_score.item()])  # Write the scores
                        results_dir = f'model_editing/SVD_experiments_perturb/full/{FIRST_TOKEN_POS}_{SEC_TOKEN_POS}/{MODE}/attention'
                        Path(results_dir).mkdir(parents=True, exist_ok=True)
                        csv_filename = os.path.join(results_dir,f'{MODE}_{SCALE}_attention_full.csv')
                        with open(csv_filename, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([CLS_attention_on_selected_query])  # Write the scores

                except ValueError as e:
                    pass
