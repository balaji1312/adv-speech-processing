import json
import shutil
import torch
import os
import numpy as np

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from datasets import load_dataset, Dataset, Audio, Value, Features, load_metric
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments, Trainer
from transformers import Wav2Vec2CTCTokenizer

#Path where you want to save your finetuned model.
demo_finetuned_model = "large_3e5_sfw"

#Make sure that while finetuning and evaluating, you use the same vocab.json file.
#Hence, everytime we start a new model, we delete the old vocab.json (if any).
#The way this script is written, a new vocab.json is created every time we run this script.
#This can create a problem (i.e., vocab.json mismatch). Hence, make sure to keep this in mind!

#If there is no vocab.json file already, the following code block will give an error. 
#So, to avoide the error, you can comment the "else" condition.
if not(os.path.exists(demo_finetuned_model)):
    os.mkdir(demo_finetuned_model)
else:
    os.remove(demo_finetuned_model+"/vocab.json")

#Data Preparation

print("\n####DATA PREPERATION####\n")

features = Features(
    {
        "text": Value("string"),
        'path': Value('string'),
        "audio": Audio(sampling_rate=16000)
    }
)


#Just for the sake of testing, I have passed the same file for train and test.
#Change it accordingly, when running the actual experiment.

libri_data = load_dataset(
    'csv', data_files={
        'train': 'sfw_new.csv',
        'dev': 'dev.csv',
    }
)

libri_data = libri_data.cast(features)

def extract_all_chars(batch):
  all_text = " ".join(batch["text"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}

vocabs = libri_data.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=libri_data.column_names["train"])
vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["dev"]["vocab"][0]))
vocab_dict = {v: k for k, v in enumerate(vocab_list)}
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]
vocab_dict["<s>"] = len(vocab_dict)
vocab_dict["</s>"] = len(vocab_dict)
vocab_dict["<unk>"] = len(vocab_dict)
vocab_dict["<pad>"] = len(vocab_dict)

print("Length of the vocabulary: ",len(vocab_dict))
print ("Vocabulary: ",vocab_dict)

with open(demo_finetuned_model+'/vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)

tokenizer = Wav2Vec2CTCTokenizer(demo_finetuned_model+"/vocab.json", unk_token="<unk>", pad_token="<pad>", word_delimiter_token="|")

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]

    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch


libri_data = libri_data.map(prepare_dataset, remove_columns=libri_data.column_names["train"], num_proc=4)

#The follwoing class helps to pad the inputs in a given batch so that all of them have the same length.

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
wer_metric = load_metric("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


#We add a linear layer on top of the pretrained wav2vec model.
#The output dimension of the added layer will be equal to the number of token in our vocabulary (vocab.json)
#CTC loss is used to train this added layer
#If using any other pretrained model (say Hubert) change the CTC layer accordingly (For example "HubertForCTC" for Hubert)
#Also update the model path (facebook/wav2vec2-base) accoringly.

print("\n####DOWNLOADING THE FOUNDATION MODEL####\n")
 
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-large",
    ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
)

# You can print the model to see the model architecture.

print ("Model architecture: ", model)

#We can freeze any desired portion of the foundation model.
#Making use of the Model architecture printed above, we can choose what part of the model has to be frozen.

#Example: Freeze the feature extractor of the wav2vec2

##model.freeze_feature_encoder()

#Example: Freeze the entire wav2vec, except the final linear layer added on top.
#for name, param in model.named_parameters():
#    if name.startswith("lm_head"): # choose whatever you like here
#        param.requires_grad = True
#    else:
#        param.requires_grad = False

#Printing what paramters will get finetuned. 
#Make sure that it matches the layers that you want to train.

#print ("The paramters that will get finetuned are: \n")
#for name, param in model.named_parameters():
#    if (param.requires_grad):
#        print (name)


#TRAINING

print("\n####DEFINING TRAIN ARGUMENTS####\n")

#Define Train arguments

training_args = TrainingArguments(
  output_dir=demo_finetuned_model,
  push_to_hub=False,
  group_by_length=True,
  #auto_find_batch_size=True,
  per_device_train_batch_size=4,
  gradient_accumulation_steps=2,
  evaluation_strategy="steps",
  #num_train_epochs=30,
  fp16=True,
  gradient_checkpointing=True,
  max_steps=15000,
  save_steps=500,
  eval_steps=500,
  logging_steps=500,
  learning_rate=3e-5,
  weight_decay=0.005,
  warmup_steps=1000,
  save_total_limit=5,
  load_best_model_at_end=True,
  seed=42,
  data_seed=42,
)

#Train/finetune the model

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=libri_data["train"],
    eval_dataset=libri_data["dev"],
    tokenizer=processor.feature_extractor,
)

print("\n####FINETUNING STARTED####\n")

trainer.train()

# trainer.train(resume_from_checkpoint = True)

print("\n####SAVING FINETUNED MODEL####\n")
trainer.save_model(demo_finetuned_model)


#EVALUATION

#print("\n####EVALUATION OF THE FINETUNED MODEL STARTED####\n")

#Copy the vocab file to the folder where the finetuned model is saved.

#shutil.copyfile("vocab.json", "wav2vec2-base-libri-finetune_demo/vocab.json")
 
#Load the finetuned model

#processor = Wav2Vec2Processor.from_pretrained(demo_finetuned_model)
#model = Wav2Vec2ForCTC.from_pretrained(demo_finetuned_model)

#def map_to_res]ult(batch):
#    with torch.no_grad():
#        input_values = torch.tensor(batch["input_values"], device="cpu").unsqueeze(0)
#        logits = model(input_values).logits
#
#    pred_ids = torch.argmax(logits, dim=-1)
#    batch["pred_str"] = processor.batch_decode(pred_ids)[0]
#    batch["text"] = processor.decode(batch["labels"], group_tokens=False)
#  
#    return batch

#results = libri_data["dev"].map(map_to_result, remove_columns=libri_data["dev"].column_names)
#print("Test WER: {:.3f}".format(wer_metric.compute(predictions=results["pred_str"], references=results["text"])))
