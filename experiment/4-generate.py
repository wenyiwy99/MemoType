# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]
import os
import json
import asyncio
import argparse
from async_llm import run_async
from gen_eval_utils import gen_res_save
from data_utils import load_data, get_answer, load_prompt, load_gen_corpus

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main(data_name, retriever, topk):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_path = os.path.join(base_dir, f"result/retrieval/{data_name}/{data_name}_{retriever}.json")
    save_path = os.path.join(base_dir, f"result/generation/{data_name}/{data_name}_{retriever}_top{topk}.json")
    csv_path = os.path.join(base_dir, f"result/generation/{data_name}/{data_name}_{retriever}_top{topk}.csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    data = load_data(load_path)
    PROMPT_PRUN, PROMPT_G, PROMPT_J = load_prompt()
    prun_idx, res_idx, llm_idx = 0, 0, 0
    prompt_list, llm_prompt_list, pred_all, answer_all, llm_results = [], [], [], [], []
    for sample in data:
        corpus = load_gen_corpus(sample)
        qa_list = sample['qa']
        for qa in qa_list:
            text_list = []
            for i in range(args.topk):
                text_list.append(corpus[qa['retrieval_res'][i]])
            retrieved_texts = "\n\n".join(text_list)
            prompt_list.append(PROMPT_PRUN.format(context=retrieved_texts, question=qa['question']))
    prun_retrieval_list = asyncio.run(run_async(prompt_list)) 

    prompt_list = []
    for sample in data:
        qa_list = sample['qa']
        for qa in qa_list:
            prompt_list.append(PROMPT_G.format(context=prun_retrieval_list[prun_idx], question=qa['question']))
            prun_idx += 1
    response_list = asyncio.run(run_async(prompt_list))   
    
    for sample in data:
        qa_list = sample['qa']
        for qa in qa_list:        
            qa["response"] = response_list[res_idx]
            res_idx += 1
            ans = get_answer(qa["response"])
            pred_all.append(ans)
            answer_all.append(str(qa["answer"]))
            llm_prompt_list.append(PROMPT_J.format(question=qa['question'], answer=str(qa["answer"]), response=ans))

    response_list = asyncio.run(run_async(llm_prompt_list, temp=0, model='gpt-4o'))
    for sample in data:
        qa_list = sample['qa']
        for qa in qa_list:
            if 'yes' in response_list[llm_idx]: 
                llm_results.append(1)
                qa['g4j'] = 1
            else: 
                llm_results.append(0)
                qa['g4j'] = 0
            llm_idx += 1
    with open(save_path, "w", encoding="utf-8") as f:  
        json.dump(data, f, ensure_ascii=False, indent=4)                
        
    gen_res_save(pred_all, answer_all, llm_results, csv_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="query answer generation.")
    parser.add_argument('--data', default='locomo')
    parser.add_argument("--retriever", default='contriever', 
                        choices=["contriever", "mpnet", "minilm","qaminilm"])
    parser.add_argument('--topk', default=3, type=int)
    args = parser.parse_args()
    if not args.data.startswith("longmemeval"): args.topk = 10
    print(args)

    main(args.data, args.retriever, args.topk)

