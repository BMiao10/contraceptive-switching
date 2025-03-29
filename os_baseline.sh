#!/bin/bash

#$ -q gpu.q
#$ -S /bin/bash
#$ -cwd
#$ -m ea
#$ -l gpu_mem=26G
#$ -l h_rt=01:55:59

module load cuda
LD_LIBRARY_PATH="/usr/local/cuda-11.0.3/lib64:/wynton/protected/home/ichs/bmiao/lib64"
export LD_LIBRARY_PATH

source ~/.bashrc
conda activate llama

export CUDA_VISIBLE_DEVICES=$SGE_GPU

gemma_path="gemma-7b-it"
llama3_chat_path="Meta-Llama-3.1-8B-Instruct"
gemma2_path="gemma2-9b-it"
biomistral_path="BioMistral-7B"
starling_beta_path='starling-7b-beta'
starling_alpha_path='starling-7b-alpha'
llama3_orig_chat_path="llama-3-8b-chat-hf"
jsl_path="JSL-MedMNX-7B-SFT"

models=($gemma_path
        $llama3_chat_path
        $gemma2_path
        $biomistral_path
        $starling_beta_path
        $starling_alpha_path
        $llama3_orig_chat_path
        $jsl_path)
          
for model in "${models[@]}"; do
  echo "Current model: ${model}"
  python -u os_baseline.py \
    --model_config_fpath="./utils/configs/${model}.json"\
    --data_fpath="./data/contraceptives/gpt4/validation.parquet.gzip"\
    --out_dir="./data/contraceptives/open_source" \
    --out_file_name="2024-08-31_${model}_prompt-dev_responses.csv"\
    --task='Task: Contraceptives can be classified into "Injectable", "Intravaginal", "Transdermal", "Intrauterine", "Oral", or "Implant". Answer the following questions - 1. What new contraceptive was suggested, proposed, or prescribed? If the patient is not starting a new contraceptive, write "NA" 2. What was the previous contraceptive the patient used? If none, write "NA" 3. In 20 words or less, why was the last contraceptive stopped or planned to be stopped? If the contraceptive was not stopped, write "NA". \nUse the following format: {"new_contraceptive":str,"last_contraceptive":str,"reason_last_contraceptive_stopped":str}'\
    --batch_size=1 \
    --truncate_note_to=6000 > out.txt
done

