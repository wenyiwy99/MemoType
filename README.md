# MemType

An official source code for paper Memory Type Matters: Enhancing Long-Term Memory in Large Language Models with Hybrid Strategies.

### TriMEM Benchmark
  TriMEM is a memory multi-classification dataset we constructed, which categorizes memory into: Episodic Memory, Personal Semantic Memory, and General Semantic Memory, with binary labels (0 or 1) indicating whether the memory belongs to each type.

  For more detailed information about TriMEM, please refer to `TriMEM/TriMEM.csv`.

### Environment
  conda env create -f environment.yml


### Data Process
The LongMemEval-S and LoCoMo datasets have been downloaded and processed.LongMemEval-M can be downloaded from https://github.com/xiaowu0162/LongMemEval

Put the LongMemEval-M dataset in `data/origin_data/`, and  run:
python data_init_process.py

The processed data will be available in `data/process_data/`


### MemoType

**Enter your API key and URL** in `experiment/async_llm.py`.

Run the following commands to reproduce our results:


python3 /MemoType/experiment/0-router_training.py 
python3 /MemoType/experiment/1-memory_route.py --data locomo
python3 /MemoType/experiment/2-construct_emb.py --data locomo
python3 /MemoType/experiment/2-query_route --data locomo
python3 /MemoType/experiment/3-retrieve --data locomo
python3 /MemoType/experiment/4-generate --data locomo
# The numbers in front of the filenames indicate the order of running.


