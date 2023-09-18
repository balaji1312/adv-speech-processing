
this_dir=`pwd`


stage=3
end_stage=3


train_log=train_large_mystlm.log

if [ $stage -le 1 ] && [ $end_stage -ge 1 ]; then
    CUDA_VISIBLE_DEVICES=0 python huggingface_wav2vec2_finetune.py
fi



file_path=${this_dir}/${train_log}

if [ ! -f $file_path ]; then
    touch $file_path
fi

datasets="test_clean test_other dev_clean dev_other"
models="large_3e5/checkpoint-10000 large_5e5/checkpoint-10000 large_7e5/checkpoint-10000 large_3e5/checkpoint-1400 large_5e5/checkpoint-1200 large_7e5/checkpoint-1000"

if [ $stage -le 2 ] && [ $end_stage -ge 2 ]; then
    for x in ${models}; do
        for y in ${datasets}; do
            python huggingface_wav2vec2_evaluate.py \
            --model=$x  \
            --dataset=$y >> $file_path
        done
    done
fi

models="large_3e5/checkpoint-10000 large_5e5/checkpoint-10000 large_3e5/checkpoint-1400 base_5e5/checkpoint-800 base_5e5/checkpoint-10000 base_7e5/checkpoint-10000"
lms="MyST10h_4gram.arpa MyST_full_4gram.arpa"

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

llms="bert ft_10h ft_full gpt2"

if [ $stage -le 4 ] && [ $end_stage -ge 4 ]; then
    for y in ${datasets}; do
        for x in ${llms}; do
            CUDA_VISIBLE_DEVICES=3 python huggingface_wav2vec2_evaluate_llm.py \
            --model=large_3e5/checkpoint-10000 \
            --dataset=$y \
            --lm=libri_full_4gram.arpa.gz  \
            --llm_model_name=$x >> $file_path
        done
    done
fi

mlms="bert roberta"

if [ $stage -le 5 ] && [ $end_stage -ge 5 ]; then
    for y in ${datasets}; do
        for x in ${mlms}; do
            CUDA_VISIBLE_DEVICES=3 python huggingface_wav2vec2_evaluate_llm_cuda.py \
            --model=large_3e5/checkpoint-10000 \
            --dataset=$y \
            --lm=libri_full_4gram.arpa.gz  \
            --llm_model_name=$x >> $file_path
        done
    done
fi