import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import json

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import CrossEncoder
from datasets import Value, Dataset
from torch.utils.data import DataLoader

from TransformerLens.transformer_lens import HookedEncoder


# Function to load JSON file into a Python dictionary
def load_json_file(file_path):
    with open(file_path, 'r') as file:
        # Load JSON data into a dictionary
        data = json.load(file)
    return data

def load_tokenizer_and_models(hf_model_name, device):
    cross_encoder_model = CrossEncoder(hf_model_name).model
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    hf_model = AutoModel.from_pretrained(hf_model_name)
    hf_model.to(device)
    pooler_layer = hf_model.pooler.to(device)
    dropout_layer = cross_encoder_model.dropout.to(device)
    classifier_layer = cross_encoder_model.classifier.to(device)
    tl_model = HookedEncoder.from_pretrained(hf_model_name, device=device, hf_model=hf_model)
    return tokenizer, tl_model, pooler_layer,dropout_layer,classifier_layer

# def preprocess(dataset, tokenize_fn):
def preprocess(dataset, tokenizer, remove_cols=["title", "text"]):
    def tokenize_fn(inputs):
    #    return tokenizer(inputs["text"], padding="max_length", truncation=True)
        return tokenizer(inputs["text"], truncation=True)

    tokenized_dataset = dataset.cast_column("_id", Value(dtype="int32"))
    tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=remove_cols)

    # Set _id as label
    # tokenized_dataset = tokenized_dataset.map(lambda x: x["_id"] == torch.tensor(int(x["_id"])))

    # Convert to torch tensors
    tokenized_dataset = tokenized_dataset.with_format("torch", columns=["input_ids", "attention_mask", "_id"]) #, output_all_columns=True

    return tokenized_dataset


def preprocess_queries(queries_df, tokenizer):
    dataset = Dataset.from_pandas(queries_df) #converts the pandas DataFrame queries_df into a Hugging Face Dataset object.
    tokenized_dataset = preprocess(dataset, tokenizer, remove_cols=["text"])
    dataloader = DataLoader(tokenized_dataset, batch_size=1)#batch_size = 1, 1 sample is 1 batch

    return dataloader


def preprocess_corpus(corpus_dict, tokenizer):
    corpus_df = pd.DataFrame.from_dict(corpus_dict, orient="index")
    corpus_df.index.name = "_id"
    dataset = Dataset.from_pandas(corpus_df)
    #for thing in dataset:
        #print(thing['text'])
        #print(thing['query_term_orignal_ct'])
    tokenzied_dataset = preprocess(dataset, tokenizer, remove_cols=["text"])
    dataloader = DataLoader(tokenzied_dataset, batch_size=1)

    return dataloader


def generate_query_doc_pair(q_tokenized,d_tokenized):#batch, posn
    batch_num = q_tokenized['input_ids'].size(0)
    tokenized_pair={}
    tokenized_pair['input_ids']=torch.cat((q_tokenized['input_ids'],d_tokenized['input_ids'][:,1:]), dim=1) #1,82
    tokenized_pair['attention_mask']=torch.cat((q_tokenized['attention_mask'],d_tokenized['attention_mask'][:,1:]), dim=1) #1,82
    tokenized_pair['token_type_ids']= torch.vstack([
                                                    torch.cat(
                                                        (torch.zeros(q_tokenized['input_ids'][batch].shape[0], dtype=torch.long), 
                                                        torch.ones(d_tokenized['input_ids'][batch].shape[0] - 1, dtype=torch.long)),
                                                        dim=0
                                                    ) for batch in range(batch_num)
                                                    ])
    print(tokenized_pair['token_type_ids'].shape)
    return tokenized_pair

'''
Encoding loop for Huggingface models
'''
def encode_hf(model, dataloader, device):
  result = np.empty((0,768))
  all_labels = np.empty(0)

  # send batch to device
  for i, batch in enumerate(tqdm(dataloader)):
    # print(batch.items())
    labels = batch.pop("_id")
    batch = {k: v.to(device) for k, v in batch.items()}

    # set model to eval and disable gradient calculations
    model.eval()
    with torch.no_grad():
      # [0] selects the hidden states
      # [:,0,:] selects the CLS token for each vector
      outputs = model(**batch)[0][:,0,:].squeeze(0)

      # LATER: collect hidden states here?

    embeddings = outputs.detach().cpu().numpy()
    # labels = batch["_ids"].detach().cpu().numpy()

    # print(embeddings.shape)
    result = np.concatenate((result, embeddings), axis=0)
    # all_labels = all_labels + labels
    all_labels = np.concatenate((all_labels, labels))

  return result, np.asarray(all_labels)


'''
Encoding loop for TransformerLens models
'''
def encode_tl(model, dataloader):
    result = np.empty((0,768))
    all_labels = np.empty(0)
   
    for _, batch in enumerate(tqdm(dataloader)):
        labels = batch.pop("_id")

        # get input ids and attention masks
        input_ids = batch.get("input_ids")
        attn_mask = batch.get("attention_mask")

        outputs = model(input_ids, return_type="embeddings", one_zero_attention_mask=attn_mask)
        embeddings = outputs[:,0,:].squeeze(0).detach().cpu().numpy()

        result = np.concatenate((result, embeddings), axis=0)
        all_labels = np.concatenate((all_labels, labels))
        # all_labels = all_labels + labels


    return result, np.asarray(all_labels)


# Compute ranking scores using dot product
def compute_ranking_scores(query_embedding, doc_embeddings, doc_ids):
#    return torch.matmul(query_embedding, doc_embeddings.t())
    scores = np.matmul(query_embedding, doc_embeddings.T)
    
    # sort scores and labels
    sorted_idx = np.argsort(scores)[::-1]

    return scores[sorted_idx], doc_ids[sorted_idx].astype(int)


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap
def plot_heatmap_block(data, labels,diagram_path,qid,did):
    #diagram_path = diagram/block/
    print(data.shape) #(3, 12, 22) #12 layers, 22 labels
    blocks = ["resid_pre"]#["resid_pre", "attn_out", "mlp_out"]
    for i in range(len(blocks)):
        title = blocks[i]
        plot_data = data[i,:,:].squeeze() #(12,22)
        plot_data = np.flipud(plot_data)
        plt.figure(figsize=(20, 10))
        sns.heatmap(plot_data, xticklabels=labels, yticklabels=range(11,-1,-1), cmap='viridis', cbar=True, vmax=1.0)
        #!!!FILP yticklabels and FLIP the data on y axis
        plt.title(title)
        plt.xlabel('posn')
        plt.ylabel('Layers')
        #plt.show()
        title = title.replace(" ", "_")
        save_path = os.path.join(diagram_path, title,f"_token_type")
        os.makedirs(save_path, exist_ok=True)  # Create directory if it doesn't exist
        file_name = f"{qid}_{did}.png"
        
        plt.savefig(os.path.join(save_path, file_name))
        print(f"saved to {save_path}")
        plt.close()  # Close the figure to free memory


def plot_heatmap_attention(data):
    print(data.shape)  # (12,12) layer head
    data = np.flipud(data)
    colors = [
        (0.0, 'red'),    # minimum value (negative): red
        ((0.3 / 1.3), 'white'),  # middle value (zero): white
        (1.0, 'blue')    # maximum value (positive): blue
    ]
    cmap = LinearSegmentedColormap.from_list('red_white_blue', colors, N=256)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(data, xticklabels=range(0,12), yticklabels=range(11,-1,-1), cmap=cmap, cbar=True, 
                vmax=1, vmin=-0.3,
                annot=True,  # This adds the actual numbers to the heatmap
                fmt=".2f",   # Format for the numbers displayed
                annot_kws={"size": 7, "color": "black"}  # Font size of the annotations
                )
    
    # Flip yticklabels and flip the data on the y-axis
    plt.title("Attention")
    plt.xlabel('head')
    plt.ylabel('layer')
    plt.show()


def normalize_patching_results(block_results):
    # Calculate min and max of block_results
    # Calculate the maximum of the absolute values
    max_abs = np.max(np.abs(block_results))
    # Normalize to the range [-1, 1], preserving signs
    block_results_norm = block_results / max_abs
    return block_results_norm

def get_type_pos_dict(text_array,qid,mode='append'):
    
    selected_query_terms = {"1089763": "miners", "1089401": "tsca", "1088958": "cadi", "1088541": "fletcher,nc", "1088475": "holmes,ny", "1101090": "azadpour", "1088444": "kashan", "1085779": "canopius", "1085510": "carewell", "1085348": "polson", "1085229": "wendelville", "1100499": "trematodiases", "1100403": "arcadis", "1064808": "acantholysis", "1100357": "ardmore", "1062223": "animsition", "1058515": "cladribine", "1051372": "cineplex", "1048917": "misconfiguration", "1045135": "wellesley", "1029552": "tosca", "1028752": "watamote", "1099761": "ari", "1020376": "amplicons", "1002940": "iheartradio", "1000798": "alpha", "992257": "desperation", "197024": "greenhorns", "61277": "brat", "44072": "chatsworth", "195582": "dammam", "234165": "saluki", "196111": "gorm", "329958": "pesto", "100020": "cortana", "193866": "izzam", "448976": "potsherd", "575616": "ankole", "434835": "konig", "488676": "retinue", "389258": "hughes", "443081": "lotte", "511367": "nfcu", "212477": "ouachita", "544060": "dresden", "428773": "wunderlist", "478295": "tigard", "610132": "neodesha", "435412": "lakegirl", "444350": "mageirocophobia", "492988": "saptco", "428819": "swegway", "477286": "antigonish", "478054": "paducah", "1094996": "tacko", "452572": "mems", "20432": "aqsarniit", "559709": "plectrums", "748935": "fraenulum?", "482666": "defdinition", "409071": "ecpi", "1101668": "denora", "537995": "cottafavi", "639084": "hortensia", "82161": "windirstat", "605651": "emmett", "720013": "arzoo", "525047": "trumbull", "978802": "browerville", "787784": "provocative", "780336": "orthorexia", "1093438": "lickspittle", "788851": "qualfon", "61531": "campagnolo", "992652": "setaf", "1092394": "msdcf", "860942": "viastone", "863187": "wintv", "1092159": "northwoods", "990010": "paihia", "840445": "prentice-hall", "775355": "natamycin", "986325": "lapham", "1091654": "parisian", "768411": "mapanything?", "194724": "gesundheit", "985905": "sentral", "1091206": "putrescine", "760930": "islet", "1090945": "ryder", "1090839": "bossov", "1090808": "semispinalis", "774866": "myfortic", "820027": "lithotrophy", "798967": "spredfast", "126821": "scooped", "60339": "stroganoff", "1090374": "strategery", "180887": "enu", "292225": "molasses"}
    selected_query_token = selected_query_terms[str(qid)]
    first_SEP = True
    type_pos_dict = {'CLS':[],'SEP1':[],'SEP2':[],"Qplus":[],"Qsub":[],"Dinj":[],"Dplus":[],"Dsub":[],"Dother":[]}

    info_dict_arr=[]#[((posstart,posend),(token, None))]
    for token_pos in text_array:
        try:
            token,pos = token_pos.split(' ')[0],int(token_pos.split(' ')[1])
            if "##" in token:
                previous_tuple = info_dict_arr[-1]#((posstart,posend),(token, None))
                prev_pos_tuple,prev_token = previous_tuple[0],previous_tuple[1]
                pos_tuple = (prev_pos_tuple[0],pos)
                new_token = prev_token+token.split("##")[-1] #exclude ## part
                info_dict_arr[-1] = (pos_tuple,new_token)
            else:
                tuple = ((pos,pos),token)
                if '[SEP]' in token and first_SEP:
                    tuple = ((pos,pos),token)
                    info_dict_arr.append(tuple)
                    type_pos_dict['SEP1'].append((pos,pos))
                    query_info_dict_arr = info_dict_arr
                    info_dict_arr = []
                    first_SEP=False
                info_dict_arr.append(tuple)
            #print(info_dict_arr)
        except IndexError as e:
            pass
    doc_info_dict_arr = info_dict_arr
    type_pos_dict['CLS'].append(query_info_dict_arr[0][0]) #pos
    type_pos_dict['SEP2'].append(doc_info_dict_arr[-1][0]) #pos
    if mode =='append':
        type_pos_dict['Dinj'].append(doc_info_dict_arr[-2][0]) #pos
    elif mode =='prepend':
        type_pos_dict['Dinj'].append(doc_info_dict_arr[1][0]) #pos
    unselected_query_tokens = []
    for item in query_info_dict_arr:
        token,pos = item[1],item[0]
        if pos not in type_pos_dict['CLS'] and pos not in type_pos_dict['Dinj'] and pos not in type_pos_dict['SEP1'] and pos not in type_pos_dict['SEP2']:
            if token ==selected_query_token:
                type_pos_dict['Qplus'].append(pos) #pos
            else:
                unselected_query_tokens.append(token)
                type_pos_dict['Qsub'].append(pos) #pos
    for item in doc_info_dict_arr:
        token,pos = item[1],item[0]
        if pos not in type_pos_dict['CLS'] and pos not in type_pos_dict['Dinj'] and pos not in type_pos_dict['SEP1'] and pos not in type_pos_dict['SEP2']:
            if token ==selected_query_token:
                type_pos_dict['Dplus'].append(pos)#it is not Ding but it is the selected_q_token
            elif token in unselected_query_tokens:
                type_pos_dict['Dsub'].append(pos) 
            else:
                type_pos_dict['Dother'].append(pos) 

    return type_pos_dict



def inject_tokens(doc, tokens):
    import random
    """
    Injects the given tokens into the document at random positions,
    but not after the second-to-last word.
    """
    words = doc.split()
    max_position = max(len(words) - 2, 0)  # Ensure we don't go out of bounds
    pos_arr =[]
    for token in tokens:
        position = random.randint(0, max_position)
        pos_arr.append(position)
        words.insert(position, token)
    return ' '.join(words),pos_arr

def inject_tokens_at_positions(doc, tokens, positions):
    """
    Injects the given tokens into the document at the specified positions.
    """
    words = doc.split()
    # Adjust positions for negative indices
    adjusted_positions = [pos if pos >= 0 else len(words) + pos for pos in positions]
    for token, position in zip(tokens, adjusted_positions):
        words.insert(position, token)
    return ' '.join(words)

def fill_multi_unequal_inputs_len_to_max(tokenized_pairs_arr,max_tokenized_pair_arr,tokenizer):
    cls_tok = max_tokenized_pair_arr["input_ids"][0][0]
    sep_tok = max_tokenized_pair_arr["input_ids"][0][-1]
    filler_token = tokenizer.encode("a", add_special_tokens=False)[0]
    max_len = torch.sum(max_tokenized_pair_arr["attention_mask"]).item()
    #result_arr_all = [max_tokenized_pair_arr]
    result_arr_all = []
    for tokenized_pair_baseline in tokenized_pairs_arr:
        b_len = torch.sum(tokenized_pair_baseline["attention_mask"]).item()
        if b_len != max_len: 
            #print("b_len != p_len, adjusting length")
            adj_n = max_len - b_len
            filler_tokens = torch.full((adj_n,), filler_token)
            filler_attn_mask = torch.full((adj_n,), tokenized_pair_baseline["attention_mask"][0][1]) 
            filler_token_id_mask = torch.full((adj_n,), tokenized_pair_baseline["token_type_ids"][0][-1]) #just filling in the document
            adj_doc = torch.cat((tokenized_pair_baseline["input_ids"][0][1:-1], filler_tokens))
            tokenized_pair_baseline["input_ids"] = torch.cat((cls_tok.view(1), adj_doc, sep_tok.view(1)), dim=0).view(1,-1)
            tokenized_pair_baseline["attention_mask"] = torch.cat((tokenized_pair_baseline["attention_mask"][0], filler_attn_mask), dim=0).view(1,-1)
            tokenized_pair_baseline["token_type_ids"] = torch.cat((tokenized_pair_baseline["token_type_ids"][0], filler_token_id_mask), dim=0).view(1,-1)
            result_arr_all.append(tokenized_pair_baseline)
    result_arr_all.append(max_tokenized_pair_arr)
    return result_arr_all


def get_tokenized_orig_perturbed(tokenized_pair_baseline,tokenized_pair_perturbed,filler_token):
    #original_doc = tfc1_add_baseline_corpus[str(qid)][doc_id]["text"]
    #perturbed_doc = tfc1_add_dd_corpus[str(qid)][doc_id]["text"]
    #tokenized_pair_baseline = tokenizer([query],[original_doc], return_tensors="pt",padding=True,truncation=True)
    #tokenized_pair_perturbed = tokenizer([query],[perturbed_doc], return_tensors="pt",padding=True, truncation=True)
    cls_tok = tokenized_pair_baseline["input_ids"][0][0]
    sep_tok = tokenized_pair_baseline["input_ids"][0][-1]
    #filler_token = tokenizer.encode("a", add_special_tokens=False)[0]
    b_len = torch.sum(tokenized_pair_baseline["attention_mask"]).item()
    p_len = torch.sum(tokenized_pair_perturbed["attention_mask"]).item()

    if b_len != p_len: 
        #print("b_len != p_len, adjusting length")
        adj_n = p_len - b_len
        filler_tokens = torch.full((adj_n,), filler_token)
        filler_attn_mask = torch.full((adj_n,), tokenized_pair_baseline["attention_mask"][0][1]) 
        filler_token_id_mask = torch.full((adj_n,), tokenized_pair_baseline["token_type_ids"][0][-1]) #just filling in the document
        adj_doc = torch.cat((tokenized_pair_baseline["input_ids"][0][1:-1], filler_tokens))
        tokenized_pair_baseline["input_ids"] = torch.cat((cls_tok.view(1), adj_doc, sep_tok.view(1)), dim=0).view(1,-1)
        tokenized_pair_baseline["attention_mask"] = torch.cat((tokenized_pair_baseline["attention_mask"][0], filler_attn_mask), dim=0).view(1,-1)
        tokenized_pair_baseline["token_type_ids"] = torch.cat((tokenized_pair_baseline["token_type_ids"][0], filler_token_id_mask), dim=0).view(1,-1)
    return tokenized_pair_baseline, tokenized_pair_perturbed


def plot_heatmap_attention_general(data,name='Attention'):
    # Flip the data vertically
    data = np.flipud(data)
    
    # Define the custom colormap
    colors = ['red', 'white', 'blue']  # Red for negative, White for zero, Blue for positive
    cmap_name = 'custom_cmap'
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)
    
    plt.figure(figsize=(8, 6))
    
    y_len, x_len = data.shape[0], data.shape[1]
    
    # Create the heatmap with the custom colormap and annotations
    sns.heatmap(
        data,
        xticklabels=range(0, x_len),
        yticklabels=range(y_len - 1, -1, -1),
        cmap=cmap,
        center=0,
        cbar=True,
        annot=True,  # Annotate the heatmap with the actual values
        fmt=".2f",  # Format the annotations to 2 decimal places
        annot_kws={"size": 6, "color": "black"}, # Customize annotation appearance
        #vmax=-0.045,
        #vmin=-0.25

    )
    
    plt.title(name)
    plt.xlabel('Head')
    plt.ylabel('Layer')
    plt.show()


def plot_heatmap_attention_general_heads(data, name='Attention', ax=None):
    # Flip the data vertically
    data = np.flipud(data)
    
    # Define the custom colormap
    colors = ['red', 'white', 'blue']  # Red for negative, White for zero, Blue for positive
    cmap_name = 'custom_cmap'
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)
    
    y_len, x_len = data.shape[0], data.shape[1]
    
    # Create the heatmap with the custom colormap and annotations
    sns.heatmap(
        data,
        xticklabels=range(0, x_len),
        yticklabels=range(y_len - 1, -1, -1),
        cmap=cmap,
        center=0,
        cbar=True,
        annot=True,  # Annotate the heatmap with the actual values
        fmt=".2f",  # Format the annotations to 2 decimal places
        annot_kws={"size": 6, "color": "black"},  # Customize annotation appearance
        ax=ax  # Plot on the provided axis
    )
    
    ax.set_title(name)
    ax.set_xlabel('Head')
    ax.set_ylabel('Layer')