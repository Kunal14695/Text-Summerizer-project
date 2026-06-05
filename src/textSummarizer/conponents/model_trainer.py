from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
import torch
import os
from textSummarizer.entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)

        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_ckpt
        ).to(device)

        seq2seq_data_collector = DataCollatorForSeq2Seq(
            tokenizer, model=model_pegasus
        )

        dataset_samsum_pt = load_from_disk(self.config.data_path)

        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4,
            logging_steps=10,
            save_steps=1000,
            eval_strategy="no",
            report_to="none"
        )

        trainer = Trainer(
            model=model_pegasus,
            args=trainer_args,
            tokenizer=tokenizer,
            data_collator=seq2seq_data_collector,
            train_dataset=dataset_samsum_pt["train"],
            eval_dataset=dataset_samsum_pt["validation"]
        )

        trainer.train()

        os.makedirs(self.config.root_dir, exist_ok=True)

        model_pegasus.save_pretrained(
            os.path.join(self.config.root_dir, "flan-t5-small-samsum-model")
        )

        tokenizer.save_pretrained(
            os.path.join(self.config.root_dir, "tokenizer")
        )