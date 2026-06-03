import os
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from textSummarizer.config.configuration import ConfigurationManager, ModelTrainerConfig
from textSummarizer.conponents.model_trainer import ModelTrainer


class ModelTrainer:

    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)

        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

        dataset = load_from_disk(str(self.config.data_path))

        trainer_args = TrainingArguments(
            output_dir=str(self.config.root_dir),
            num_train_epochs=self.config.num_train_epochs,
            warmup_steps=self.config.warmup_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_train_batch_size,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            evaluation_strategy=self.config.evaluation_strategy,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            predict_with_generate=False,
            fp16=torch.cuda.is_available(),
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=model,
            args=trainer_args,
            tokenizer=tokenizer,
            data_collator=data_collator,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("validation"),
        )

        trainer.train()

        os.makedirs(self.config.root_dir, exist_ok=True)
        model.save_pretrained(os.path.join(self.config.root_dir, "pegasus-samsum-model"))
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))


class ModelTrainerTrainingPipeline:
    def main(self):
        cfg = ConfigurationManager().get_model_trainer_config()

        config = ModelTrainerConfig(
            root_dir=Path(cfg.root_dir),
            data_path=Path(cfg.data_path),
            model_ckpt=str(cfg.model_ckpt),
            num_train_epochs=int(cfg.num_train_epochs),
            warmup_steps=int(cfg.warmup_steps),
            per_device_train_batch_size=int(cfg.per_device_train_batch_size),
            weight_decay=float(cfg.weight_decay),
            logging_steps=int(cfg.logging_steps),
            evaluation_strategy=str(cfg.evaluation_strategy),
            eval_steps=int(cfg.eval_steps),
            save_steps=float(cfg.save_steps),
            gradient_accumulation_steps=int(cfg.gradient_accumulation_steps),
        )

        ModelTrainer(config).train()

