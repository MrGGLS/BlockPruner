export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=6

model_name=Llama-2-7b
nsamples=32
dataset=alpaca
block_num=20
block_type=mix

python block_search.py \
        --model-path /data/lgzhong/tiny/models/${model_name}\
        --block-type ${block_type} \
        --cal-nsamples ${nsamples} \
        --del-block-num ${block_num} \
        --cal-dataset ${dataset} \
        --ppl-search-path ppls \
        --ppl-eval-batch-size 2 \
        --device cuda 