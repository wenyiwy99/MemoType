# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""
File:retrieve.py
Description: retrieve memory with different retrievers
"""
import os
import csv
import sys
import json
import torch
import argparse
from tqdm import tqdm
from ret_eval_utils import evaluate_retrieval, retrieval_csv_save
from data_utils import load_data, retrieval_res_setup, corpus_process, routed_corpus_process, full_rank_restore

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.memotype import MemoType


def main(data_name, retriever):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, f'data/process_data/{data_name}/{data_name}_routed.jsonl')
    emb_path = os.path.join(base_dir, f'data/data_emb/{data_name}/{data_name}_{retriever}.pt')
    save_path = os.path.join(base_dir, f"result/retrieval/{data_name}/{data_name}_{retriever}.jsonl")
    csv_path = os.path.join(base_dir, f"result/retrieval/{data_name}/rrf_{data_name}_{retriever}.csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    data = load_data(data_path)
    emb_dict = torch.load(emb_path, weights_only=True)

    memo = MemoType(retriever)

    retrieval_res = retrieval_res_setup()

    for sample in tqdm(data):
        corpus, corpus_ids, corpus_embs = corpus_process(sample, emb_dict)
        for qa in sample['qa']:
            routed_ids, routed_embs, routed_corpus = routed_corpus_process(qa, sample, corpus_ids, corpus_embs, corpus)
            routed_ranks = memo.retrieve_rrf(qa, sample, routed_ids, routed_embs)
            full_ranks = full_rank_restore(routed_ids, corpus_ids, routed_ranks)
            qa['retrieval_res'] = full_ranks
            if not data_name.startswith("longmemeval") or not qa["question_id"].endswith("_abs"):
                for k in [1, 3, 5, 10]:
                    recall_any, ndcg_any = evaluate_retrieval(full_ranks, qa['answer_ids'], corpus_ids, k)
                    retrieval_res[f'recall@{k}'].append(recall_any)
                    retrieval_res[f'ndcg@{k}'].append(ndcg_any)
        with open(save_path, "a", encoding="utf-8") as f:  # 使用 "a" 模式追加
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    retrieval_csv_save(csv_path, retrieval_res)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="retrieve memory with different retrievers.")
    parser.add_argument('--data', default='locomo')
    parser.add_argument("--retriever", default='contriever', 
                        choices=["contriever", "mpnet", "minilm","qaminilm"])
    args = parser.parse_args()
    print(args)

    main(args.data, args.retriever)
