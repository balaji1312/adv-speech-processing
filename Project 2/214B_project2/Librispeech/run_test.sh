
this_dir=`pwd`


#Stage: 1 - no lm, 2 - ngram, 3- llm

stage=1
end_stage=3



train_log=project_eval.log


file_path=${this_dir}/${train_log}

if [ ! -f $file_path ]; then
    touch $file_path
fi

#Add test dataset in the line below
datasets=""
models="large_3e5/checkpoint-10000"

if [ $stage -le 1 ] && [ $end_stage -ge 1 ]; then
    for x in ${models}; do
        for y in ${datasets}; do
            python huggingface_wav2vec2_evaluate.py \
            --model=$x  \
            --dataset=$y >> $file_path
        done
    done
fi

lms="libri_full_4gram.arpa.gz"

if [ $stage -le 2 ] && [ $end_stage -ge 2 ]; then
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

llms="ft_full"

if [ $stage -le 3 ] && [ $end_stage -ge 3 ]; then
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
