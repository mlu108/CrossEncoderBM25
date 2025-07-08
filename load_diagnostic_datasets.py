#tfc1: tfc1_add_baseline_final_dd_append_corpus.json
#stmc1: stmc1_add_append_final_dd_corpus.json
from helpers import load_json_file
import pandas as pd
import os
import csv
import random

selected_query_terms = {"1089763": "miners", "1089401": "tsca", "1088958": "cadi", "1088541": "fletcher,nc", "1088475": "holmes,ny", "1101090": "azadpour", "1088444": "kashan", "1085779": "canopius", "1085510": "carewell", "1085348": "polson", "1085229": "wendelville", "1100499": "trematodiases", "1100403": "arcadis", "1064808": "acantholysis", "1100357": "ardmore", "1062223": "animsition", "1058515": "cladribine", "1051372": "cineplex", "1048917": "misconfiguration", "1045135": "wellesley", "1029552": "tosca", "1028752": "watamote", "1099761": "ari", "1020376": "amplicons", "1002940": "iheartradio", "1000798": "alpha", "992257": "desperation", "197024": "greenhorns", "61277": "brat", "44072": "chatsworth", "195582": "dammam", "234165": "saluki", "196111": "gorm", "329958": "pesto", "100020": "cortana", "193866": "izzam", "448976": "potsherd", "575616": "ankole", "434835": "konig", "488676": "retinue", "389258": "hughes", "443081": "lotte", "511367": "nfcu", "212477": "ouachita", "544060": "dresden", "428773": "wunderlist", "478295": "tigard", "610132": "neodesha", "435412": "lakegirl", "444350": "mageirocophobia", "492988": "saptco", "428819": "swegway", "477286": "antigonish", "478054": "paducah", "1094996": "tacko", "452572": "mems", "20432": "aqsarniit", "559709": "plectrums", "748935": "fraenulum?", "482666": "defdinition", "409071": "ecpi", "1101668": "denora", "537995": "cottafavi", "639084": "hortensia", "82161": "windirstat", "605651": "emmett", "720013": "arzoo", "525047": "trumbull", "978802": "browerville", "787784": "provocative", "780336": "orthorexia", "1093438": "lickspittle", "788851": "qualfon", "61531": "campagnolo", "992652": "setaf", "1092394": "msdcf", "860942": "viastone", "863187": "wintv", "1092159": "northwoods", "990010": "paihia", "840445": "prentice-hall", "775355": "natamycin", "986325": "lapham", "1091654": "parisian", "768411": "mapanything?", "194724": "gesundheit", "985905": "sentral", "1091206": "putrescine", "760930": "islet", "1090945": "ryder", "1090839": "bossov", "1090808": "semispinalis", "774866": "myfortic", "820027": "lithotrophy", "798967": "spredfast", "126821": "scooped", "60339": "stroganoff", "1090374": "strategery", "180887": "enu", "292225": "molasses"}
selected_query_replaced_pronoun = {
    "1089763": "they",         # miners
    "1089401": "it",           # tsca
    "1088958": "it",           # cadi
    "1088541": "it",           # fletcher, nc
    "1088475": "it",           # holmes, ny
    "1101090": "he",           # azadpour
    "1088444": "it",           # kashan
    "1085779": "they",         # canopius (assuming organization)
    "1085510": "it",           # carewell
    "1085348": "it",           # polson
    "1085229": "it",           # wendelville
    "1100499": "they",         # trematodiases
    "1100403": "they",         # arcadis (assuming organization)
    "1064808": "it",           # acantholysis
    "1100357": "it",           # ardmore
    "1062223": "it",           # animsition
    "1058515": "it",           # cladribine
    "1051372": "it",           # cineplex
    "1048917": "it",           # misconfiguration
    "1045135": "it",           # wellesley
    "1029552": "it",           # tosca
    "1028752": "it",           # watamote
    "1099761": "he",           # ari
    "1020376": "they",         # amplicons
    "1002940": "it",           # iheartradio
    "1000798": "it",           # alpha
    "992257": "it",            # desperation
    "197024": "they",          # greenhorns
    "61277": "he",             # brat (assuming person)
    "44072": "it",             # chatsworth
    "195582": "it",            # dammam
    "234165": "it",            # saluki
    "196111": "he",            # gorm
    "329958": "it",            # pesto
    "100020": "it",            # cortana
    "193866": "he",            # izzam
    "448976": "it",            # potsherd
    "575616": "it",            # ankole
    "434835": "he",            # konig
    "488676": "it",            # retinue
    "389258": "he",            # hughes
    "443081": "it",            # lotte
    "511367": "it",            # nfcu
    "212477": "it",            # ouachita
    "544060": "it",            # dresden
    "428773": "it",            # wunderlist
    "478295": "it",            # tigard
    "610132": "it",            # neodesha
    "435412": "she",           # lakegirl
    "444350": "it",            # mageirocophobia
    "492988": "it",            # saptco
    "428819": "it",            # swegway
    "477286": "it",            # antigonish
    "478054": "it",            # paducah
    "1094996": "he",           # tacko
    "452572": "they",          # mems
    "20432": "they",           # aqsarniit
    "559709": "they",          # plectrums
    "748935": "it",            # fraenulum
    "482666": "it",            # defdinition
    "409071": "it",            # ecpi
    "1101668": "she",          # denora
    "537995": "he",            # cottafavi
    "639084": "she",           # hortensia
    "82161": "it",             # windirstat
    "605651": "he",            # emmett
    "720013": "she",           # arzoo
    "525047": "it",            # trumbull
    "978802": "it",            # browerville
    "787784": "it",            # provocative
    "780336": "it",            # orthorexia
    "1093438": "he",           # lickspittle
    "788851": "they",          # qualfon (assuming organization)
    "61531": "it",             # campagnolo
    "992652": "they",          # setaf
    "1092394": "it",           # msdcf
    "860942": "it",            # viastone
    "863187": "it",            # wintv
    "1092159": "they",         # northwoods
    "990010": "it",            # paihia
    "840445": "it",            # prentice-hall
    "775355": "it",            # natamycin
    "986325": "he",            # lapham
    "1091654": "they",         # parisian
    "768411": "it",            # mapanything
    "194724": "it",            # gesundheit
    "985905": "it",            # sentral
    "1091206": "it",           # putrescine
    "760930": "it",            # islet
    "1090945": "he",           # ryder
    "1090839": "he",           # bossov
    "1090808": "it",           # semispinalis
    "774866": "it",            # myfortic
    "820027": "it",            # lithotrophy
    "798967": "it",            # spredfast
    "126821": "it",            # scooped
    "60339": "it",             # stroganoff
    "1090374": "it",           # strategery
    "180887": "it",            # enu
    "292225": "it",            # molasses
    }

def load_tfc1_dataset():
    """ 
    Load the TFC1 dataset.
    Given a baseline document, we perturb it by appending one more selected query term.
    """
    tfc1_add_queries = pd.read_csv(os.path.join("tfc1_add_qids_with_text.csv"), header=None, names=["_id", "text"])
    tfc1_queries_dict = tfc1_add_queries.set_index('_id').to_dict(orient='index')
    target_qids = tfc1_add_queries["_id"].tolist() 
    tfc1_add_baseline_corpus = load_json_file("tfc1_add_baseline_final_dd_append_corpus.json")["corpus"]
    tfc1_add_dd_corpus = load_json_file(f"tfc1_add_append_final_dd_corpus.json")["corpus"]
    return target_qids, tfc1_queries_dict, tfc1_add_baseline_corpus, tfc1_add_dd_corpus

""" Usage example:
for i, qid in enumerate(target_qids):
    query = tfc1_queries_dict[qid]['text']
    target_docs = stmc1_add_dd_corpus[str(qid)]
    for j, doc_id in enumerate(target_docs):
        original_doc = tfc1_add_baseline_corpus[str(qid)][doc_id]["text"]
        perturbed_doc = stmc1_add_dd_corpus[str(qid)][doc_id]["text"]   
"""

def load_stmc1_dataset():
    """
    Load the STMC1 dataset.
    Given a baseline document, we perturb it by appending one more semantically similar term to the selected query term. 
    """
    tfc1_add_queries = pd.read_csv(os.path.join("tfc1_add_qids_with_text.csv"), header=None, names=["_id", "text"])
    tfc1_queries_dict = tfc1_add_queries.set_index('_id').to_dict(orient='index')
    target_qids = tfc1_add_queries["_id"].tolist() 
    tfc1_add_baseline_corpus = load_json_file("tfc1_add_baseline_final_dd_append_corpus.json")["corpus"]
    stmc1_add_dd_corpus = load_json_file(f"stmc1_add_append_final_dd_corpus.json")["corpus"]
    return target_qids, tfc1_queries_dict, tfc1_add_baseline_corpus, stmc1_add_dd_corpus


def load_lnc2():
    """
    Load the LNC2 dataset.
    For each baseline doc, create five perturbations by sequentially appending an increasing number of random sentences
    from a non-relevant query in the base dataset.
    
    Returns:
        qid_list: list of unique qids (strings)
        lnc2_sentences_varying_lengths: dict {qid: list of 5 modified sentences strings}
    """
    input_file_path='tfc2_lnc1_query_answer_injected.csv'
    rows = []
    with open(input_file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        headers = next(reader)  # Skip headers
        for row in reader:
            rows.append(row)

    # Extract unique qids as strings, strip whitespace
    qid_list = list({row[0].strip() for row in rows})

    # Build doc_dict with string keys stripped
    doc_dict = {row[0].strip(): row[2].lower().replace('"', '') for row in rows}

    # Build correspondence dict with consistent string keys
    shuffled_qids = qid_list.copy()
    random.shuffle(shuffled_qids)
    correspondence_dict = dict(zip(qid_list, shuffled_qids))

    lnc2_sentences_varying_lengths = {}

    for row in rows:
        qid = row[0].strip()
        query = row[1]
        doc = row[2].lower().replace('"', '')

        # Defensive: skip if qid not in selected_query_terms or pronouns
        if qid not in selected_query_terms or qid not in selected_query_replaced_pronoun:
            continue

        term = selected_query_terms[qid]
        pronoun = selected_query_replaced_pronoun[qid]

        another_qid = correspondence_dict[qid]
        if another_qid not in doc_dict or another_qid not in selected_query_terms or another_qid not in selected_query_replaced_pronoun:
            continue

        another_doc = doc_dict[another_qid]
        another_term = selected_query_terms[another_qid]
        another_pronoun = selected_query_replaced_pronoun[another_qid]

        # Split baseline doc sentences
        sentences = doc.split(". ")
        # Create baseline repeated sentences (first sentence repeated 5 times)
        modified_sentences = [sentences[0] + '.' for _ in range(5)]

        # Prepare another_doc sentences with pronoun replacement
        another_doc_replaced = another_doc.replace(another_term, another_pronoun)
        another_sentences = another_doc_replaced.split(". ")

        # Build incremental appended answers
        another_answers = []
        for i in range(5):
            # Join first i sentences (i=0 means empty string)
            answer = ". ".join(another_sentences[:i])
            another_answers.append(answer)

        # Append incremental answers to baseline repeated sentences
        for idx in range(min(len(modified_sentences), len(another_answers))):
            if another_answers[idx]:
                modified_sentences[idx] += " " + another_answers[idx]

        lnc2_sentences_varying_lengths[qid] = modified_sentences

    return qid_list, lnc2_sentences_varying_lengths

def load_tfc2_dataset():
    """ 
    Load the TFC2 dataset.
    We use the baseline document to create five relevant sentences starting with selected query term’s pronoun and perturb by incrementally restoring the term.
    Returns:
    1. target_qids: list of query IDs,
    2. tfc1_queries_dict: dictionary of queries keyed by _id,
    3. tfc2_data: dictionary keyed by query ID, where each value contains:
    original query,
    original doc,
    list of answers with replacements for counts 5 down to 0,
    baseline_answer corresponding to replacement count=5 (the first in your list).
    """

    tfc1_add_queries = pd.read_csv(os.path.join("tfc1_add_qids_with_text.csv"),
                                  header=None, names=["_id", "text"])
    tfc1_queries_dict = tfc1_add_queries.set_index('_id').to_dict(orient='index')
    target_qids = tfc1_add_queries["_id"].tolist()
    tfc2_data = {}
    input_file_path = 'tfc2_lnc1_query_answer_injected.csv'
    with open(input_file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            qid, query, doc = row
            qid_int = qid
            term = selected_query_terms.get(str(qid), None)
            pronoun = selected_query_replaced_pronoun.get(qid_int, None)
            if term is None or pronoun is None:
                continue
            answers = [doc.lower().replace(term.lower(), pronoun, count)
                for count in [5, 4, 3, 2, 1, 0]]

            # Store in dict keyed by qid
            tfc2_data[qid_int] = {
                "query": query,
                "docs_varying_number_of_matched_word": answers,
                "baseline_doc": answers[0]  # count=5 is baseline (first in list)
            }

    return target_qids,tfc1_queries_dict, tfc2_data
