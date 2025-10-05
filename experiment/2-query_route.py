# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]
"""
File: query_route.py
Description: route the query to its corresponding type and store the route result of memory .
"""

import os
import sys
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment.data_utils import load_route_res, hybrid_query_expansion, setup_memory, memory_route, store_route


MEM_TYPES = ["EM", "PM", "GM"]
ELEMENTS = ["T", "P", "L"]

def main(data_name, batch):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, f'data/process_data/{data_name}/{data_name}_with_fm.json')
    route_res_path = os.path.join(base_dir, f"data/process_data/{data_name}/{data_name}_memory_routed.jsonl")
    router_path = os.path.join(base_dir, f'model/router/')
    save_path = os.path.join(base_dir, f"data/process_data/{data_name}/{data_name}_routed.jsonl")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    data, mem_route_dict = load_route_res(data_path, route_res_path)
    if not data_name.startswith("longmemeval"): batch = 1

    tokenizer = BertTokenizer.from_pretrained(router_path)
    model = BertForSequenceClassification.from_pretrained(router_path)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()

    # route the query type and store the memory route result.
    for sample in tqdm(data):
        his_samples = sample["sessions"]
        sample.update(setup_memory())
        for his_sess_id in his_samples.keys():
            sess_route = mem_route_dict[his_sess_id]
            store_route(sample, sess_route)
        qa_list = sample['qa']
        for qa in qa_list:
            type_list = memory_route(qa['fake_memory'], model, tokenizer, device)
            type_count = np.sum(np.stack(type_list, axis=0), axis=0)
            qa["retrieval_scope"] = [MEM_TYPES[i] for i in range(len(type_count)) if type_count[i] > 0]
            qa["retrieval_type"] = MEM_TYPES[np.argmax(type_count)]

    # hybrid query expansion strategy.
    for i in tqdm(range(0, len(data), batch)):  
        batch_data = data[i:i+batch]
        key_list, ev_list = hybrid_query_expansion(batch_data)
        pm_index, em_index = 0, 0
        for sample in batch_data:
            for qa in sample['qa']:
                if qa['retrieval_type'] == 'PM':
                    qa['query_keywords'] = key_list[pm_index]
                    pm_index += 1
                if qa['retrieval_type'] == 'EM':
                    qa['query_event'] = ev_list[em_index]
                    em_index += 1
            with open(save_path, "a", encoding="utf-8") as f:  # 使用 "a" 模式追加
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="route the query to its corresponding type and store the route result of memory.")
    parser.add_argument('--data', default='locomo')
    parser.add_argument('--batch', default=200, 
                        help='The batches number for parallel invocation of the LLM')
    args = parser.parse_args()
    print(args)

    main(args.data, args.batch)