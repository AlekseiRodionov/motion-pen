import os

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import TrOCRProcessor
from transformers import VisionEncoderDecoderModel
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import default_data_collator
import evaluate


class CHDataset(Dataset):
   def __init__(self, root_dir, df, processor):
       self.root_dir = root_dir
       self.df = df
       self.processor = processor

   def __getitem__(self, idx):
       file_name = self.df["file"][idx]
       text = self.df["label"][idx]
       image = Image.open(os.path.join(self.root_dir, file_name)).convert("RGB")
       pixel_values = self.processor(image, return_tensors="pt").pixel_values
       labels = self.processor.tokenizer(
           text, padding="max_length", max_length=4).input_ids
       labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
       encoding = {
           "pixel_values": pixel_values.squeeze(),
           "labels": torch.tensor(labels),
       }
       return encoding

   def __len__(self):
       return len(self.df)


train_df = pd.read_csv(
    'Data/train/labels.csv',
)

test_df = pd.read_csv(
    'Data/test/labels.csv',
)

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")

train_dataset = CHDataset(
   root_dir=os.path.join('Data', 'train'),
   df=train_df,
   processor=processor,
)

test_dataset = CHDataset(
   root_dir=os.path.join('Data', 'test'),
   df=test_df,
   processor=processor,
)

model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 4
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4

training_args = Seq2SeqTrainingArguments(
   num_train_epochs=1,
   predict_with_generate=True,
   eval_strategy="epoch",
   save_strategy="epoch",
   per_device_train_batch_size=4,
   per_device_eval_batch_size=4,
   fp16=True,
   output_dir="trained_model",
)

cer_metric = evaluate.load("cer")

def compute_metrics(pred):
   labels_ids = pred.label_ids
   pred_ids = pred.predictions

   pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
   labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
   label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

   cer = cer_metric.compute(predictions=pred_str, references=label_str)

   return {"cer": cer}

trainer = Seq2SeqTrainer(
   model=model,
   processing_class=processor.tokenizer,
   args=training_args,
   compute_metrics=compute_metrics,
   train_dataset=train_dataset,
   eval_dataset=test_dataset,
   data_collator=default_data_collator,
)
trainer.train()

processor.save_pretrained("trained_model/weights")
model.save_pretrained("trained_model/weights")