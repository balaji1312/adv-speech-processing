## ReadMe

Group 1

Members: Natarajan Balaji Shankar, Khushbu Pahwa, Eray Eren, Aurosweta Mahapatra

Location of all files: /data/balaji/Project2_Group1/214B_project2

### General Instructions

The following instructions are for both running experiments from scratch as well as for examining models we have finetuned in the given folder

To run experiments again it is necessary to create a new virtual env with the following packages: transformers, accelerate, datasets, evaluate, pyctcdecode, librosa, jiwer, torchaudio, kenlm

### Task 1

To run task 1, all we need to is navigate inside the Librispeech folder and edit the config of training args in huggingface_wav2vec2_finetune.py. To evaluate we run huggingface_wav2vec2_evaluate.py. 

Note: to make running inference easier, we have added command line arguments to huggingface_wav2vec2_evaluate.py to run the file from a bash script.

Our finetuned models are stored in this format: <model_size>_<lr>/checkpoint-<chk#> Our best performing model on librispeech is the large_3e5/checkpoint-10000.

### Task 2

Similar to task 1 we store all our models in this format: <model_size>_<lr>/checkpoint-<chk#>. To finetune with a different config we now edit huggingface_wav2vec2_finetune.py in the MyST folder and we evaluate using huggingface_wav2vec2_evaluate.py. Our best performing model without augmentation on MyST is the large_5e5/checkpoint-10000.

Augmentations: Our code for augmentations is present at the following location:

SFW - ./MyST/gen_sfw.py

Rest of the augmentations - ./augmentations

Note: It is not necessary to perform the augmentations again as they have been store in the ./MyST/data and ./MyST/data_vtlp_sp_1.1_0.9 folders. The combined paths of all augmentations are stored in train_vtlp_sfw_sp_pp.csv.

The models trained on augmented data are stored in the following format: large_3e5_<aug_type>/checkpoint-<chk#>.

Our best performing model with augmentation on MyST is the large_3e5_vtlp/checkpoint-15000.

### Task 3

We store the various n-gram models used for our experiments at ./langauge_model. We store the various finetuned GPT2 models at ./langauge_model/gpt2 and finetuned Bert models at ./langauge_model/libri_full.

To perform n-gram decoding we can utilise the script huggingface_wav2vec2_evaluate_withlm.py, which has been updated with command line arguments for easier chaining from a bash script. For LLM decoding we can utilise either huggingface_wav2vec2_evaluate_llm.py or huggingface_wav2vec2_evaluate_llm_cuda.py based on whether the requested model can fit on the GPU. These scripts accept command line arguments as well.

The results from LLM decoding (the hyp files) are present at ./Librispeech/large_3e5/checkpoint-10000 and ./MyST/large_3e5_vtlp/checkpoint-15000, but they may have been overwritten from multiple runs. Training logs are also present at ./Librispeech/train_large_llm.log and ./MyST/train_large_llm_vtlp.log.

It is also possible to run Task 1, 2 and 3 sequentially using the scripts ./Librispeech/run.sh ./MyST/run.sh, where the only prior edits necessary would be to set the config in huggingface_wav2vec2_finetune.py and edit the path to the log file.

### Task 4

Files location: ./layerwise-analysis

For layerwise CCA analysis:

LibriSpeech

Step 1: Extracting the transformer hidden layer presentations for the well performing fine-tuned models

python3 layerwise-analysis/huggingface_wav2vec2_evaluate_libri.py. You can skip this step since we already saved the representations. So, this is optional.

Step 2: Extracting CCA scores for each layer

python3 layerwise-analysis/codes/tools/get_scores_libri.py

Scores are printed to the terminal, please save those to a file if you want to keep those.

python3 layerwise-analysis/codes/tools/get_scores_libri.py > libri_cca.txt

MyST

Step 1: Extracting the transformer hidden layer presentations for the well performing fine-tuned models

python3 layerwise-analysis/huggingface_wav2vec2_evaluate_myst.py

You can skip this step since we already saved the representations. So, this is optional.

Step 2: Extracting CCA scores for each layer

python3 layerwise-analysis/codes/tools/get_scores_myst.py

Scores are printed to the terminal, please save those to a file if you want to keep those.

python3 layerwise-analysis/codes/tools/get_scores_libri.py > myst_cca.txt