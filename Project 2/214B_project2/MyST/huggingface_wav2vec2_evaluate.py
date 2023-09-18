import json
import shutil
import torch
import numpy as np

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from datasets import load_dataset, Dataset, Audio, Value, Features, load_metric
from evaluate import load
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments, Trainer
from transformers import Wav2Vec2CTCTokenizer

######NOTE######
#Before you start evaluating, make sure your model/checkpoint folder has vocab.json
#If not, copy the vocab.json file to the model/checkpoint folder.
#Also, make sure that you use the same vocab.json file while training and testing.
#While creating the baseline scripts, we realized that a different vocab.json file is created everytime we run the train script.
#So, for a model, once the vocab.json is created, make sure you save a copy of that file.
################

import argparse

parser = argparse.ArgumentParser(description="options for dataset, model and arpa")
parser.add_argument("--model", required=True, type=str) #model location
parser.add_argument("--dataset", required=True, type=str) #dataset name

args = parser.parse_args()
#Path where your finetuned model is saved.
checkpoint_model=args.model

REF="_1h_ref.txt"  #Your choice of suffix
HYP="_1h_hyp.txt"  #Your choice of suffix

dataset = args.dataset

ref_file = open(checkpoint_model+ "/"+dataset+REF, "w") #For ease of identification, use the name of the test set being decoded here 
hyp_file = open(checkpoint_model+ "/"+dataset+HYP, "w") #For ease of identification, use the name of the test set being decoded here

#Based on the above variables, the reference and decoded texts will be saved in files:
#"test_1h_ref.txt" and "test_1h_hyp.txt"

#Data Preparation
print("\n####DATA PREPERATION####\n")

features = Features(
    {
        "text": Value("string"),
        'path': Value('string'),
        "audio": Audio(sampling_rate=16000)
    }
)

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
        batch["labels"] = processor(batch["text"]).input_ids

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

def map_to_result(batch):
    with torch.no_grad():
        input_values = torch.tensor(batch["input_values"], device="cpu").unsqueeze(0)
        logits = model(input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_str"] = processor.batch_decode(pred_ids)[0]
    batch["text"] = processor.decode(batch["labels"], group_tokens=False)

    return batch

#EVALUATION

print("\n####EVALUATION OF THE FINETUNED MODEL STARTED####\n")

#Load the finetuned model

processor = Wav2Vec2Processor.from_pretrained(checkpoint_model)
model = Wav2Vec2ForCTC.from_pretrained(checkpoint_model)

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
print(checkpoint_model, dataset)



for i in range(len(results["text"])):
    ref_file.write("Line_"+str(i)+"\t")
    ref_file.write(results["text"][i]+"\n")
    hyp_file.write("Line_"+str(i)+"\t")
    hyp_file.write(results["pred_str"][i]+"\n")

ref_file.close()
hyp_file.close()
