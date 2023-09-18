import json
import shutil
import torch
import numpy as np

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from datasets import load_dataset, Dataset, Audio, Value, Features, load_metric
from evaluate import load
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ProcessorWithLM
from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments, Trainer
from transformers import Wav2Vec2CTCTokenizer

from transformers import AutoModelForCTC, AutoProcessor, AutoTokenizer

from pyctcdecode import build_ctcdecoder
from ctcdecode import CTCBeamDecoder

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import BertTokenizer,BertForMaskedLM

# from mlm.scorers import MLMScorer, MLMScorerPT, LMScorer
# from mlm.models import get_pretrained

# import mxnet as mx

# ctxs = [mx.gpu(0)] 


######NOTE######
#Before you start evaluating, make sure your model/checkpoint folder has vocab.json
#If not, copy the vocab.json file to the model/checkpoint folder.
#Also, make sure that you use the same vocab.json file while training and testing.
#While creating the baseline scripts, we realized that a different vocab.json file is created everytime we run the train script.
#So, for a model, once the vocab.json is created, make sure you save a copy of that file.

#When you run this script the second time, you will get an error saying "File exists <checkpoint_model>/language_model".
#To solve this error, everytime you rerun this script, delete the "language_model" folder created in your "checkpoint_model" path.
################
import argparse

parser = argparse.ArgumentParser(description="options for dataset, model and arpa")
parser.add_argument("--model", required=True, type=str) #model location
parser.add_argument("--dataset", required=True, type=str) #dataset name
parser.add_argument("--lm", required=True, type=str) #lm location
parser.add_argument("--llm_model_name", required=True, type=str) #llm type

args = parser.parse_args()
#Path where your finetuned model is saved.
checkpoint_model=args.model

#Path where your language model is saved.
language_model="/data/balaji/Project2_Group1/214B_project2/langauge_model/" + args.lm #libri10h_3gram.arpa

REF="_base_1h_llm_ref.txt" #Your choice of suffix
HYP="_base_1h_llm_hyp.txt" #Your choice of suffix

dataset  = args.dataset

ref_file = open(checkpoint_model+ "/"+dataset+REF, "w")   #For ease of identification, use the name of the test set being decoded here
hyp_file = open(checkpoint_model + "/"+dataset+HYP, "w")   #For ease of identification, use the name of the test set being decoded here
#Based on the above variables, the reference and decoded texts will be saved in files:
#"test_other_base_1h_withlm_ref.txt" and "test_other_base_1h_withlm_hyp.txt"

#Data Preparation
print("\n####DATA PREPERATION####\n")

features = Features(
    {
        "text": Value("string"),
        'path': Value('string'),
        "audio": Audio(sampling_rate=16000)
    }
)

#PASS THE TEST SET THAT YOU WANT TO DECODE
#Change the csv accordingly, when running your experiment.
libri_data = load_dataset(
    'csv', data_files={
        'test': dataset+'.csv',
    }
)

libri_data = libri_data.cast(features)


def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]

    with processor.as_target_processor():
        #batch["labels"] = processor(batch["text"]).input_ids
        batch["labels"] = batch["text"]
    return batch

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}



#EVALUATION

print("\n####EVALUATION OF THE FINETUNED MODEL STARTED####\n")

model = Wav2Vec2ForCTC.from_pretrained(checkpoint_model)
processor = Wav2Vec2Processor.from_pretrained(checkpoint_model)

vocab_dict = processor.tokenizer.get_vocab()
sorted_vocab_dict = {k: v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}

print (list(sorted_vocab_dict.keys()))

decoder = build_ctcdecoder(
    labels=list(sorted_vocab_dict.keys()),
    kenlm_model_path=language_model,
)

# num_classes =  len(sorted_vocab_dict) # Number of classes (including the blank symbol)
# beam_width = 25   # Beam width for beam search
# scorer = Scorer()  # You can use the default scorer or create a custom one

model_name = args.llm_model_name

if model_name=='gpt2':
    llm_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    llm_model = GPT2LMHeadModel.from_pretrained(model_name)
    llm_model.eval()
elif model_name=='bert':
    llm_model = BertForMaskedLM.from_pretrained('bert-large-uncased')
    llm_model.eval()
    # Load pre-trained model tokenizer (vocabulary)
    llm_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
else:
    base_dir = '/data/balaji/Project2_Group1/214B_project2/langauge_model/gpt2/'
    llm_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    llm_model = GPT2LMHeadModel.from_pretrained(base_dir + model_name)
    llm_model.eval()


# labels = ''.join([k for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])])
# decoder = CTCBeamDecoder(
#     labels,
#     model_path=None,
#     alpha=0,
#     beta=0,
#     cutoff_top_n=20,
#     cutoff_prob=1.0,
#     beam_width=100,
#     num_processes=4,
#     blank_id=0,
#     log_probs_input=False
# )

# scorer = MLMScorerPT(llm_model, llm_vocab, llm_tokenizer, ctxs)

def score_sentences(sentence):

    if model_name=='bert':
        tensor_input = llm_tokenizer.encode(sentence, return_tensors='pt')
        repeat_input = tensor_input.repeat(tensor_input.size(-1)-2, 1)
        mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
        masked_input = repeat_input.masked_fill(mask == 1, llm_tokenizer.mask_token_id)
        labels = repeat_input.masked_fill( masked_input != llm_tokenizer.mask_token_id, -100)
        with torch.inference_mode():
            loss = -1*model(masked_input, labels=labels).loss

        # inputs= llm_tokenizer.encode_plus(sentence,return_tensors="pt", add_special_tokens = True,pad_to_max_length = True, return_attention_mask = True)

        # input_ids = inputs["input_ids"]

        # labels = inputs["input_ids"].clone()

        # with torch.no_grad():
        #     ot = llm_model(
        #         input_ids=input_ids,
        #         labels=labels,
        #         attention_mask=inputs["attention_mask"],
        #         token_type_ids=inputs["token_type_ids"]
        #     )
        #     # print(type(ot))
        #     # print(ot)
        #     loss = -1*ot['loss']
        # tokenize_input = llm_tokenizer.tokenize(sentence)
        # tokenize_input = ["[CLS]"]+tokenize_input+["[SEP]"]
        # tensor_input = torch.tensor([llm_tokenizer.convert_tokens_to_ids(tokenize_input)])
        # with torch.no_grad():
        #     loss=llm_model(tensor_input)[0]
    else:
        tokenize_input = llm_tokenizer.encode(sentence)
        tensor_input = torch.tensor([tokenize_input])
        
        loss=-1*llm_model(tensor_input, labels=tensor_input)[0]
    
    # print(loss)
    return loss.detach().numpy()


def map_to_result(batch):
    with torch.no_grad():
        input_values = torch.tensor(batch["input_values"], device="cpu").unsqueeze(0)
        logits = model(input_values).logits
        #print (logits.size())

    # pred_ids = torch.argmax(logits, dim=-1)

    # probabilities = torch.stack(probabilities)  # Convert the list to a tensor
    # input_lengths = torch.tensor([probabilities.shape[0]])  # Length of input sequence

    # Perform beam search decoding
    # print(logits.shape)
    logits = logits.squeeze().detach().numpy()
    beams = decoder.decode_beams(logits, beam_prune_logp=-100, token_min_logp=-100)

    sentences = [b[0] for b in beams]

    # beam_list = [beam_results[0][i][:out_lens[0][i]] for i in range(beam_results.shape[1])]

    # sentences = processor.batch_decode(beam_list)[0]

    # print(sentences)

    if(len(sentences)>16):
        sentences = sentences[:16]

    scores = [score_sentences(s) for s in sentences]
    
    # scores = scorer.score_sentences(sentences)
    # print(scores)

    

    best_candidate = sentences[int(np.argmax(scores))]

    # sentences = processor.batch_decode(logits.numpy()).text
    batch["pred_str"] = best_candidate
    # print ("Hypothesis: ", batch["pred_str"])
    #batch["text"] = processor_with_lm.decode(batch["labels"], group_tokens=False)
    batch["text"] = batch["labels"]
    # print ("Reference: ", batch["text"])

    return batch

# decoder.save_to_dir(checkpoint_model) #line commented out for bash script to run

# from transformers import Wav2Vec2ProcessorWithLM

# processor_with_lm = Wav2Vec2ProcessorWithLM(
#     feature_extractor=processor.feature_extractor,
#     tokenizer=processor.tokenizer,
#     decoder=decoder
# )

#So far "libri_data" had the columns: ['text', 'path', 'audio'].
#You can check that by: 
print ("Initial column names: \n", libri_data.column_names) 

#Now, we will have to process the audio files so that they can be fed into the wav2vec. This is done by the function "prepare_dataset()" defined above.
#After, prepare_dataset() operation, two extra columns ['input_values', 'labels'] are added to our data.
#The updated libri_data will have columns: ['text', 'path', 'audio', 'input_values', 'labels']
#In order to finetune the model, we will require only columns ['input_values', 'labels'].
#Hence, we drop columns ['text', 'path', 'audio'] from libri_data as follows:

libri_data = libri_data.map(prepare_dataset, remove_columns=libri_data.column_names["test"], num_proc=8)

#We can check if we have the desired columns in the updated "libri_data" as follows:
print ("Column names after batch preperation: \n",libri_data.column_names)

#Now, let us test the performance on our test set.
#We use map_to_result() to calculate the test set outputs.
#map_to_result() add two extra columns [pred_str, text], i.e., predicted text and reference text respectively.
#The output is saved in the variable "results".
#To calculate Word Error Rate (WER), we will require only columns [pred_str, text].
#Hence we drop the other columns i.e., ['input_values', 'labels'] as follows:

results = libri_data["test"].map(map_to_result, remove_columns=libri_data["test"].column_names)

#We can check if we have the desired columns in the updated "results" as follows:
print ("results column names:: \n",results.column_names)

wer_metric = load("wer")

#for pred in results["pred_str"]:
#   print ("Before: ", pred)
#   str(pred).replace("[PAD]", "")
#   print ("After: ", pred)

print("Test WER: {:.3f}".format(wer_metric.compute(predictions=results["pred_str"], references=results["text"])))
print(checkpoint_model, dataset, args.lm, args.llm_model_name)

for i in range(len(results["text"])):
    ref_file.write("Line_"+str(i)+"\t")
    ref_file.write(results["text"][i]+"\n")
    hyp_file.write("Line_"+str(i)+"\t")
    hyp_file.write(results["pred_str"][i]+"\n")

ref_file.close()
hyp_file.close()
