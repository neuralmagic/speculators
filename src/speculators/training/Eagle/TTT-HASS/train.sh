CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch -m --multi_gpu --mixed_precision=bf16 --main_process_port 29501 train.main_train \
    --basepath train/llama3-1 \
    --tmpdir /tmpdata/megan/3_1_8B \
    --cpdir layerNormOnBigDataKL00_v00 \
    --configpath train/llama3_8_B.json \
    --epoch 3 \
    --bs 2 \
    --topk_w 0 \
    --topk 1 \
    --forward_num_total 3 \


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch -m --multi_gpu --mixed_precision=bf16 --main_process_port 29501 train.main_hass \
#     --basepath train/llama3-1 \
#     --tmpdir /tmpdata/megan/3_1_8B \
#     --cpdir hass \
#     --configpath train/EAGLE-LLaMA3-Instruct-8B  \
#     --epoch 5 \
#     --bs 1 \
#     --topk_w 0 \
#     --topk 1 \
#     --forward_num_total 5


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch -m --mixed_precision=bf16 train.main_hass \
#     --basepath train/mistral-small \
#     --tmpdir /tmpdata/megan/70B \
#     --cpdir llama3_1_8B_spec_ft \
#     --configpath train/EAGLE-LLaMA3-Instruct-8B \
#     --epoch 10 \
#     --lr 0.00001 \
#     --bs 2 \
#     --topk 10 \
#     --topk_w 1 \
#     --forward_num_total 3 \
#     --ckpt_path llama3_1_8B_spec_baseline/state_9
