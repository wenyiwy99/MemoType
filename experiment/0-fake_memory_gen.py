# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]
"""
File: memory_generate.py
Description: This file is designed to generate fake memory and store it in the file.
"""
import os
import sys
import json
import argparse
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.memotype import MemoType
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(data_name, batch):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_path = os.path.join(base_dir, f'data/process_data/{data_name}.json')
    save_path = os.path.join(base_dir, f'data/process_data/{data_name}/{data_name}_with_fm.json')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(load_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    memo = MemoType()

    if not data_name.startswith("longmemeval"):
        # The locomo dataset performs parallel calls to the LLM for each conversation.
        for sample in tqdm(data):
            qa_list = sample['qa']
            query_list = []
            for qa in qa_list:
                query_list.append(qa['question'])
            memory_list = memo.memory_gen(query_list, data_name)
            for j, qa in enumerate(qa_list):
                qa['fake_memory'] = memory_list[j]
    else:
        # The longmemeval dataset concatenates questions from all conversations and then invokes the LLM in parallel.
        for i in tqdm(range(0, len(data), batch)):
            batch_data = data[i:i+batch]
            query_list, ans_index = [], 0
            for sample in batch_data:
                qa_list = sample['qa'] 
                for qa in qa_list:
                    query_list.append(qa['question'])
            memory_list = memo.memory_gen(query_list, data_name)
            for sample in batch_data:
                qa_list = sample['qa'] 
                for qa in qa_list:
                    qa['fake_memory'] = memory_list[ans_index]
                    ans_index += 1

    with open(save_path, "w") as f: 
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="fake memory generation.")
    parser.add_argument('--data', default='longmemeval_s', 
                        choices=['longmemeval_s', 'locomo', 'longmemeval_m'])
    parser.add_argument('--batch', default=100, type=int, 
                        help='The batches number for parallel invocation of the LLM')
    args = parser.parse_args()
    print(args)
    main(args.data, args.batch)
