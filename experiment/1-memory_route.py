# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]
"""
File: memory_route.py
Description: route the memory to its corresponding type.
"""

import os
import sys
import json
import torch
import argparse
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment.data_utils import load_session, memory_route, setup_memory, process_em

MEM_TYPES = ["EM", "PM", "GM"]
ELEMENTS = ["T", "P", "L"]

def main(data_name, batch):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_path = os.path.join(base_dir, f'data/process_data/{data_name}/{data_name}_with_fm.json')
    router_path = os.path.join(base_dir, f'model/router/')
    save_path = os.path.join(base_dir, f"data/process_data/{data_name}/{data_name}_memory_routed.jsonl")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    session_dicts, session_ids = load_session(load_path, save_path)
            
    tokenizer = BertTokenizer.from_pretrained(router_path)
    model = BertForSequenceClassification.from_pretrained(router_path)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()

    for i in tqdm(range(0, len(session_ids), batch)):
        sess_res = {}
        batch_ids = session_ids[i:i+batch]
        batch_data = [sess for batch_id in batch_ids for sess in session_dicts[batch_id]]
        batch_types = memory_route(batch_data, model, tokenizer, device)
        ev_list, ev_index_list = process_em(batch_data, batch_types)
        batch_index, em_index = 0, 0
        for sess_id in batch_ids:
            sess_len = len(session_dicts[sess_id])
            sess_res[sess_id] = setup_memory()
            res = sess_res[sess_id]
            sess_class = batch_types[batch_index: batch_index + sess_len]
            for turn_id, turn_class in enumerate(sess_class):
                for k, tag in enumerate(turn_class):
                    if tag == 1: 
                        res[MEM_TYPES[k]].append(f"{sess_id}-turn_{turn_id}")
                    if tag == 1 and k == 0:
                        for element in ELEMENTS: res["EM_Info"][element].extend(ev_list[em_index][element])
                        res['EM_Index'].append(ev_index_list[em_index])
                        em_index += 1
            batch_index += sess_len 
        with open(save_path, "a", encoding="utf-8") as f:
            json_line = json.dumps(sess_res, ensure_ascii=False)
            f.write(json_line + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="route the memory to its corresponding type.")
    parser.add_argument('--data', default='longmemeval_s')
    parser.add_argument('--batch', default=200, 
                        help='The batches number for parallel invocation of the LLM')
    args = parser.parse_args()
    print(args)

    main(args.data, args.batch)