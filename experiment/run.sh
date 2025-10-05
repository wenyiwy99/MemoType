#!/bin/bash
#SBATCH --job-name=wy_job             # 作业名称-请修改为自己任务名字 
#SBATCH --output=/home/yiwen23/0-script/New_Memory/output_%j.txt        # 标准输出文件名 (%j 表示作业ID)-请修改为自己路径
#SBATCH --error=/home/yiwen23/0-script/New_Memory/error_%j.txt          # 标准错误文件名-请修改为自己路径
#SBATCH --cpus-per-task=12             # 每个任务使用的CPU核心数
#SBATCH --mem=200G                      # 申请100GB内存
#SBATCH --time=48:00:00               # 运行时间限制，格式为hh:mm:ss
conda inits
conda activate wy_memory #-请修改为自己conda环境 1e-2 5e-02 1e-3 --data locomo 
CUDA_VISIBLE_DEVICES="0" 

# python3 /home/yiwen23/My_Work/TriMeM/5-class_memory.py --data longmemeval_s
python3 /home/yiwen23/My_Work/3-MemoType/experiment/1-memory_route.py