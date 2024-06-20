export CUDA_VISIBLE_DEVICES=6
export HF_ENDPOINT=https://hf-mirror.com

model_name=Llama-2-7b
block_num=12
dataset=wikitext2
ppl_search_file=ppls/${model_name}_mix_alpaca_ns_64_del_order_list.json


python eval.py \
        --do-eval \
        --model-path /data/lgzhong/tiny/models/${model_name}\
        --del-block-num ${block_num} \
        --cal-dataset ${dataset} \
        --ppl-search-file ${ppl_search_file}\
        --ppl-eval-batch-size 1 \
        --device cuda \
        --compute-dtype bf16 