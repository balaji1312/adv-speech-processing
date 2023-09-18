
this_dir=`pwd`


stage=2
end_stage=4


train_log=sfw_train_lm.log

if [ $stage -le 1 ] && [ $end_stage -ge 1 ]; then
    CUDA_VISIBLE_DEVICES=0 python huggingface_wav2vec2_finetune.py
fi



file_path=${this_dir}/${train_log}

if [ ! -f $file_path ]; then
    touch $file_path
fi

datasets="test dev"
models="large_3e5_sfw/checkpoint-15000"

if [ $stage -le 2 ] && [ $end_stage -ge 2 ]; then
    for x in ${models}; do
        for y in ${datasets}; do
            python huggingface_wav2vec2_evaluate.py \
            --model=$x  \
            --dataset=$y >> $file_path
        done
    done
fi

# models="large_1e5/checkpoint-10000 large_3e5/checkpoint-10000 large_5e5/checkpoint-10000"
# lms="libri_full_4gram.arpa.gz libri10h_4gram.arpa MyST10h_4gram.arpa MyST_full_4gram.arpa"
lms="MyST_full_4gram.arpa"

# train_log=train_large_llm.log
# file_path=${this_dir}/${train_log}

# if [ ! -f $file_path ]; then
#     touch $file_path
# fi

# models="large_5e5/checkpoint-10000"

if [ $stage -le 3 ] && [ $end_stage -ge 3 ]; then
    for x in ${models}; do
        for y in ${datasets}; do
            for z in ${lms}; do
                python huggingface_wav2vec2_evaluate_withlm.py \
                --model=$x  \
                --dataset=$y  \
                --lm=$z >> $file_path
            done
        done
    done
fi

# train_log=train_large_llm_vtlp.log
# file_path=${this_dir}/${train_log}

# if [ ! -f $file_path ]; then
#     touch $file_path
# fi

llms="myst_full"
# datasets="test"

if [ $stage -le 4 ] && [ $end_stage -ge 4 ]; then
    for y in ${datasets}; do
        for x in ${llms}; do
            CUDA_VISIBLE_DEVICES=0 python huggingface_wav2vec2_evaluate_llm_cuda.py \
            --model=large_3e5_sfw/checkpoint-15000 \
            --dataset=$y \
            --lm=MyST_full_4gram.arpa  \
            --llm_model_name=$x >> $file_path
        done
    done
fi

