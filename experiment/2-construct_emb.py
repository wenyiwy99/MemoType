# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]
"""
File: construct_emb.py
Description: construct the embedding of the memory.
"""

import os
import sys
import json
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from data_utils import load_data, load_session
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


ELEMENTS = ["T", "P", "L"]

class EmbeddingModel():
    def __init__(self, retriever):
        self.retriever = retriever
        if retriever == 'contriever':
            self.model = AutoModel.from_pretrained('facebook/contriever').to(torch.device('cuda', 0))
            self.tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
        elif retriever == 'mpnet':
            self.model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-cos-v1')
        elif retriever == 'minilm':
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        elif retriever == 'qaminilm':
            self.model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')

    def get_emb(self, expansion):
        def mean_pooling(token_embeddings, mask):
            token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
            sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
            return sentence_embeddings
        
        if self.retriever == 'contriever':
            with torch.no_grad():
                all_docs_vectors = []
                dataloader = DataLoader(expansion, batch_size=64, shuffle=False)
                for batch in dataloader:
                    inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                    outputs = self.model(**inputs)
                    cur_docs_vectors = mean_pooling(outputs[0], inputs['attention_mask']).detach().cpu()
                    all_docs_vectors.append(cur_docs_vectors)
                all_docs_vectors = torch.concat(all_docs_vectors, axis=0)
        else:
            all_docs_vectors = self.model.encode(expansion)

        for i, text in enumerate(expansion):
            if text == 'N/A': all_docs_vectors[i] = torch.zeros(all_docs_vectors.size(1))
        
        return all_docs_vectors


def main(data_name, retriever):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, f'data/process_data/{data_name}.json')
    route_res_path = os.path.join(base_dir, f"data/process_data/{data_name}/{data_name}_memory_routed.json")
    save_path = os.path.join(base_dir, f"data/data_emb/{data_name}/{data_name}_{retriever}.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    emb_dicts = {}
    mem_route_dicts = load_data(route_res_path)
    session_dicts, _ = load_session(data_path)
    model = EmbeddingModel(retriever)

    for sess_id in tqdm(session_dicts.keys()):
        emb_dicts[sess_id] = {}
        sess = session_dicts[sess_id]
        em_info = mem_route_dicts[sess_id]['EM_Info']
        if sess: emb_dicts[sess_id]['sess'] = model.get_emb(sess)
        for ele in ELEMENTS:
            if em_info[ele]: emb_dicts[sess_id][ele] = model.get_emb(em_info[ele])

    torch.save(emb_dicts, save_path)
    
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="construct the embedding of memory.")
    parser.add_argument('--data', default='locomo')
    parser.add_argument('--retriever', default='contriever', 
                        choices=["contriever", "mpnet", "minilm","qaminilm"])
    args = parser.parse_args()
    print(args)

    main(args.data, args.retriever)