import os
import re
import sys
import copy
import json
import torch
import asyncio
import numpy as np
from tqdm import tqdm
from async_llm import run_async

MEM_TYPES = ["EM", "PM", "GM"]
ELEMENTS = ["T", "P", "L"]

def load_data(load_path):
    if os.path.splitext(os.path.basename(load_path))[1] == ".jsonl":
        data = []
        with open(load_path, "r") as f:
            for line in f.readlines():
                data.append(json.loads(line.strip()))
    elif os.path.splitext(os.path.basename(load_path))[1] == ".json":
        data = json.load(open(load_path))
        if isinstance(data, dict):
            data = list(data.values())
    else:
        raise NotImplementedError()
    return data

def setup_memory():
    class_res = {mem_type: [] for mem_type in MEM_TYPES}
    class_res["EM_Info"] = {element:[] for element in ELEMENTS}
    class_res['EM_Index'] = []
    return class_res

def load_session(load_path, save_path):
    data = load_data(load_path)
    print(f"number of data: {len(data)}")

    session_dicts, session_ids = {}, []
    for entry in tqdm(data):
        for id, session in entry['sessions'].items():
            if id not in session_dicts:
                session_dicts[id] = session
                session_ids.append(id)

    if os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            for line in f:
                classified_dict = json.loads(line.strip())
                for key in classified_dict.keys():
                    session_ids.remove(key)
    return session_dicts, session_ids

def load_route_res(data_path, route_res_path):
    data = load_data(data_path)
    print(f"number of data: {len(data)}")

    mem_route_dict = {}
    with open(route_res_path, "r", encoding="utf-8") as f:
        for line in f:
            route_dict = json.loads(line.strip())
            mem_route_dict.update(route_dict)
    return data, mem_route_dict


def load_corpus(load_path, route_res_path):
    data = load_data(load_path)
    print(f"number of data: {len(data)}")

    session_dicts, session_ids = {}, []
    for entry in tqdm(data):
        for id, session in entry['sessions'].items():
            if id not in session_dicts:
                session_dicts[id] = session
                session_ids.append(id)

    em_dicts = {}
    with open(route_res_path, "r", encoding="utf-8") as f:
        for line in f:
            route_dict = json.loads(line.strip())
            em_dicts.update(route_dict)

    return session_dicts, em_dicts

def memory_route(texts, model, tokenizer, device, threshold=0.5):
    results = []
    inputs = tokenizer(
        texts,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits).cpu().numpy() 
    for prob in probs:
        if np.all(prob < threshold):
            predictions = np.zeros_like(prob).astype(int).tolist()
            predictions[np.argmax(prob)] = 1
        else:
            predictions = (prob >= threshold).astype(int).tolist()
        results.append(predictions)
    return results

def extract_event(text_list):
    Event_Element = ["T", "P", "L"]
    event_pattern = re.compile(
        r"Time: ([^\n]*)\s*(?=Person\(s\)|\Z)"
        r"Person\(s\): ([^\n]*)\s*(?=Location|\Z)"
        r"Location: ([^\n]*)\s*(?=\[Event\]|\Z)",
        re.MULTILINE
    )
    event_list = []
    event_index_list = []
    for text in text_list:
        events = {ele:[] for ele in Event_Element}
        if text:
            event_blocks = re.split(r'\[Event \d+\]', text)[1:]
            for block in event_blocks:
                ev_match = event_pattern.match(block.strip())
                if not ev_match: continue
                event_data = [ev_match[1].strip(), ev_match[2].strip(), ev_match[3].strip()]
                if all("N/A" in text for text in event_data): continue
                for i, element in enumerate(Event_Element):
                    if 'N/A' in event_data[i]:
                        events[element].append('N/A')
                    else:
                        events[element].append(re.sub(r"[,\[\]]", "", event_data[i]))
        event_list.append(events)
        event_index_list.append(len(events['T']))
    return event_list, event_index_list

def extract_event_query(text_list):
    Event_Element = ["T", "P", "L", "E"]
    event_pattern = re.compile(
        r"Time: ([^\n]*)\s*(?=Person\(s\)|\Z)"
        r"Person\(s\): ([^\n]*)\s*(?=Location|\Z)"
        r"Location: ([^\n]*)\s*(?=Event|\Z)"
        r"Event: ([^\n]*)\s*(?=Summary|\Z)",
        re.MULTILINE
    )
    events_list = []
    for text in text_list:
        events = {'key':[], 'val': []}
        if text:
            match = event_pattern.match(text.strip())
            event_data = {
                "T": match[1].strip(),
                "P": match[2].strip(),
                "L": match[3].strip(),
                "E": match[4].strip()
            }
            for element in Event_Element:
                if 'N/A' not in event_data[element]:
                    if element != "E": events['key'].append(element)
                    events['val'].append(re.sub(r"[,\[\]]", "", event_data[element]))
        events_list.append(events)
    return events_list


def store_route(sample, sess_route):
    for tag, turn_ids in sess_route.items():
        if tag in MEM_TYPES:  
            sample[tag].extend(turn_ids) 
        elif tag == 'EM_Info':
            for element in ELEMENTS:
                for temp_ele in turn_ids[element]:
                    if 'N/A' in temp_ele: sample[tag][element].append('N/A')
                    else: sample[tag][element].append(temp_ele.replace(",", ""))
    for i, turn_id in enumerate(sess_route['EM']):
        sample['EM_Index'].extend([turn_id] * sess_route['EM_Index'][i])

def process_em(data, types):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    em_prompt_path = os.path.join(base_dir, 'model/instructions/event_extract.md')
    with open(em_prompt_path, 'r', encoding='utf-8') as f:
        em_prompt = f.read()
    
    em_memory = []
    em_memory.extend(data[i] for i in range(len(types)) if types[i][0]==1)
    prompt_list = [em_prompt.format(text_to_be_processed=text) for text in em_memory]
    em_list = asyncio.run(run_async(prompt_list))
    event_list, event_index_list = extract_event(em_list)
    return event_list, event_index_list


def hybrid_query_expansion(data):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    keyword_prompt_path = os.path.join(base_dir, 'model/instructions/keyword_extract.md')
    em_q_prompt_path = os.path.join(base_dir, 'model/instructions/query_event_extract.md')
    with open(keyword_prompt_path, 'r', encoding='utf-8') as f:
        key_prompt = f.read()
    with open(em_q_prompt_path, 'r', encoding='utf-8') as f:
        em_query_prompt = f.read()

    prompt_list, em_prompt_list, key_list = [], [], []
    for sample in data:
        qa_list = sample['qa']
        for qa in qa_list:
            if qa['retrieval_type'] == 'PM': prompt_list.append(key_prompt.format(query=qa['question']))
            if qa['retrieval_type'] == 'EM': em_prompt_list.append(em_query_prompt.format(text_to_be_processed=qa['question']))
    pm_index = len(prompt_list)
    prompt_list.extend(em_prompt_list)
    output_list = asyncio.run(run_async(prompt_list))
    pm_list, em_list = output_list[:pm_index], output_list[pm_index:]
    for key in pm_list:
        keyword = key.split(",")[:3]
        keyword = ",".join(keyword)
        key_list.append(keyword)
    event_list = extract_event_query(em_list)
    return key_list, event_list

def retrieval_res_setup():
    retrieval_res = {}
    for k in [1, 3, 5, 10]:
        retrieval_res[f'recall@{k}'] = []
        retrieval_res[f'ndcg@{k}'] = []
    return retrieval_res

def corpus_process(sample, emb_dict):
    corpus, corpus_ids, corpus_embs = [], [], {}
    for cur_sess_id, sess in sample['sessions'].items():
        sess_emb = emb_dict[cur_sess_id]
        for key, value in sess_emb.items():
            if key not in corpus_embs: corpus_embs[key] = []
            corpus_embs[key].append(value)
        for turn_id, turn in enumerate(sess):
            corpus.append(turn)
            corpus_ids.append(f"{cur_sess_id}-turn_{turn_id}")

    for key, value in corpus_embs.items():
        corpus_embs[key] = torch.cat(corpus_embs[key], dim=0)
    return corpus, corpus_ids, corpus_embs

def routed_corpus_process(qa, sample, corpus_ids, corpus_embs, corpus):
    routed_ids, routed_embs = [], copy.deepcopy(corpus_embs)
    for corpus_id in corpus_ids:
        for key in qa['retrieval_scope']:
            if corpus_id in sample[key]:
                routed_ids.append(corpus_id)
                break
    indices = [corpus_ids.index(turn_id) for turn_id in routed_ids]
    routed_corpus = [corpus[i] for i in indices]
    routed_embs['sess'] = corpus_embs['sess'][indices]
    return routed_ids, routed_embs, routed_corpus

def full_rank_restore(routed_ids, corpus_ids, routed_ranks):
    full_ranks = [None] * len(routed_ids)
    unsorted_indices = [i for i in range(len(corpus_ids))]
    for rk, idx in enumerate(routed_ranks):
        full_index = corpus_ids.index(routed_ids[idx])
        full_ranks[rk] = full_index
        unsorted_indices.remove(full_index)
    if unsorted_indices: full_ranks.extend(unsorted_indices)
    return full_ranks        

def get_answer(ans):
    strip_word_list = [
        "\nDialogs:",
        "\n[bot]:",
        "\nAssistant:",
        "\nReview:",
        "\n",
        "[bot]:",
    ]
    cut_word_list = ["\n[human]:", "\nQuestion:", "\nQ:"]

    for strip_word in strip_word_list:
        ans = ans.strip(strip_word)
    for cut_word in cut_word_list:
        if cut_word in ans:
            ans = ans.split(cut_word)[0]
    return ans

def load_prompt(save_path):
    if os.path.exists(save_path):
        with open(save_path, 'w') as file:
            pass  
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    llm_judge_path = os.path.join(base_dir, f"model/instructions/llm_judge.md")
    memory_prun_path = os.path.join(base_dir, f"model/instructions/memory_prun.md")
    response_path = os.path.join(base_dir, f"model/instructions/response_generation.md")
    with open(llm_judge_path, "r", encoding="utf-8") as f:
        PROMPT_J = f.read()
    with open(memory_prun_path, "r", encoding="utf-8") as f:
        PROMPT_PRUN = f.read()
    with open(response_path, "r", encoding="utf-8") as f:
        PROMPT_G = f.read()

    return PROMPT_PRUN, PROMPT_G, PROMPT_J

def load_gen_corpus(sample):
    corpus = []
    for sess in sample['sessions'].values():
        for i in range(0, len(sess), 2):
            if i + 1 < len(sess): 
                combined = sess[i] + '\n' + sess[i + 1]
                corpus.append(combined)
                corpus.append(combined)
            else:
                corpus.append(sess[i])
    return corpus