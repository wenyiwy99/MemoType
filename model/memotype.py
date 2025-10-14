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
from experiment.data_utils import reciprocal_rank_fusion, extract_event_query


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

        self.init_prompt()

    def init_prompt(self):
        def load_prompt(file_name):
            file_path = os.path.join(base_dir, 'instructions', file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.ua_memory_gen_prompt = load_prompt('user_assistant_memory_gen.md')
        self.uu_memory_gen_prompt = load_prompt('user_user_memory_gen.md')
        self.key_prompt = load_prompt('keyword_extract.md')
        self.em_query_prompt = load_prompt('query_event_extract.md')

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


    def retrieve(self, qa, sample, routed_ids, routed_embs):
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

        fused_rank = reciprocal_rank_fusion(rank_list)
        return fused_rank
    
    def memory_gen(self, query_list, data_name='longmemeval_s', num=5):
        if data_name.startswith("longmemeval"):
            memory_gen_prompt = self.ua_memory_gen_prompt
        else:
            memory_gen_prompt = self.uu_memory_gen_prompt
        prompt_list = []
        for query in query_list:
            prompt = memory_gen_prompt.format(query_to_be_answered=query)
            prompt_list.extend([prompt]*num)
        answer_list = asyncio.run(run_async(prompt_list, model='gpt-4o-mini', temp=2, topp=0.9))
        final_answer_list = []
        for i in range(0, len(answer_list), num):
            temp_answer_list = answer_list[i:i+num]
            final_answer_list.append(temp_answer_list)
        return final_answer_list
    
    def hybrid_query_expansion(self, data):
        prompt_list, em_prompt_list, key_list = [], [], []
        for sample in data:
            qa_list = sample['qa']
            for qa in qa_list:
                if qa['retrieval_type'] == 'PM': prompt_list.append(self.key_prompt.format(query=qa['question']))
                if qa['retrieval_type'] == 'EM': em_prompt_list.append(self.em_query_prompt.format(text_to_be_processed=qa['question']))
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
