# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import os
import sys
import torch
import asyncio
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment.async_llm import run_async


categories = ["EM", "PM", "GM"]

class MemoType:
    def __init__(
        self,
        retriever='contriever'
    ):
        
        self.retriever = retriever
        self.device = 'cuda'
        if retriever == 'contriever':
            self.model = AutoModel.from_pretrained('facebook/contriever').to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
            
        elif retriever == 'mpnet':
            self.model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-cos-v1')
        elif retriever == 'minilm':
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        elif retriever == 'qaminilm':
            self.model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')

        base_dir = os.path.dirname(os.path.abspath(__file__))
        ua_memory_prompt_path = os.path.join(base_dir, f'instructions/user_assistant_memory_gen.md')
        uu_memory_prompt_path = os.path.join(base_dir, f'instructions/user_user_memory_gen.md')
        with open(ua_memory_prompt_path, "r", encoding="utf-8") as f:
            self.ua_memory_gen_prompt = f.read()
        with open(uu_memory_prompt_path, "r", encoding="utf-8") as f:
            self.uu_memory_gen_prompt = f.read()

    def get_score(self, query, corpus_emb):
        if self.retriever == 'contriever':
            tokenizer, model = self.tokenizer, self.model
            def mean_pooling(token_embeddings, mask):
                token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
                sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
                return sentence_embeddings

            with torch.no_grad():
                inputs = tokenizer([query], padding=True, truncation=True, return_tensors='pt')
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                outputs = model(**inputs)
                query_vectors = mean_pooling(outputs[0], inputs['attention_mask']).detach().cpu()
                scores = (query_vectors @ corpus_emb.T).squeeze()

        elif self.retriever in ['mpnet', 'minilm', 'qaminilm']:
            query_emb = self.model.encode(query, convert_to_tensor=True)
            scores = (query_emb @ corpus_emb.T).squeeze()
        
        return scores


    def retrieve_rrf(self, qa, sample, routed_ids, routed_embs):
        rank_list = []
        scores = self.get_score(qa['fake_memory'][0], routed_embs['sess'])
        rank_list.append(scores.argsort(descending=True).tolist() )

        if qa['retrieval_type'] == 'EM':
            scores1 = self.get_score(qa['question'], routed_embs['sess'])
            em_index_list = [i for i, turn_id in enumerate(routed_ids) if turn_id in sample['EM']]
            if len(qa['query_event']['key']) > 0:
                for i, ele in enumerate(qa['query_event']['key']):
                    if ele in ['L', 'P', 'T'] :
                        scores_e = self.get_score(qa['query_event']['val'][i], routed_embs[ele])
                        full_scores_e = torch.zeros_like(scores1)
                        index = 0
                        for em_index in em_index_list:
                            em_len = sample['EM_Index'].count(routed_ids[em_index])
                            if em_len > 0:
                                score = max(scores_e[index:index+em_len])
                                full_scores_e[em_index] = score
                                index += em_len
                        scores1 += 0.5 * full_scores_e
                score2 = self.get_score(qa['query_event']['val'][-1], routed_embs['sess'])  
                scores1 += 0.5 * score2
            rank_list.append(scores1.argsort(descending=True).tolist())
        elif qa['retrieval_type'] == 'GM':
            scores2 = self.get_score(qa['question'], routed_embs['sess'])
            rank_list.append(scores2.argsort(descending=True).tolist())
        else:
            scores2 = self.get_score(qa['question'] + qa['query_keywords'], routed_embs['sess']) 
            rank_list.append(scores2.argsort(descending=True).tolist())

        fused_rank = self.reciprocal_rank_fusion(rank_list)
        return fused_rank


    def reciprocal_rank_fusion(self, rank_lists, k=60):
        """
        Implements the Reciprocal Rank Fusion (RRF) algorithm.

        Parameters:
            rank_lists (list of list of str): A list of ranked lists. Each inner list contains document IDs ranked in descending order of relevance.
            k (int): The constant parameter for RRF to adjust the impact of rank. Default is 60.

        Returns:
            dict: A dictionary mapping document IDs to their RRF scores, sorted in descending order of scores.
        """

        # Dictionary to store the accumulated RRF scores for each document
        rrf_scores = {}

        for rank_list in rank_lists:
            for rank, doc_id in enumerate(rank_list):
                # Calculate RRF score contribution for this document
                score = 1 / (k + rank + 1)  # rank is 0-based, so add 1
                if doc_id in rrf_scores:
                    rrf_scores[doc_id] += score
                else:
                    rrf_scores[doc_id] = score

        # Sort documents by RRF score in descending order
        sorted_docs = [doc_id for doc_id, _ in sorted(rrf_scores.items(), key=lambda item: (item[1], doc_id), reverse=True)]
        return sorted_docs
    
    def memory_gen(self, query_list, data_name='longmemeval_s', num=5):
        if data_name.startswith("longmemeval"):
            memory_gen_prompt = self.ua_memory_gen_prompt
        else:
            memory_gen_prompt = self.uu_memory_gen_prompt
        prompt_list = []
        for query in query_list:
            prompt = memory_gen_prompt.format(query_to_be_answered=query)
            prompt_list.extend([prompt]*num)
        answer_list = asyncio.run(run_async(prompt_list, 1))
        final_answer_list = []
        for i in range(0, len(answer_list), num):
            temp_answer_list = answer_list[i:i+num]
            final_answer_list.append(temp_answer_list)
        return final_answer_list
