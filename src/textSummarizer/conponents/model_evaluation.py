from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk, load_metric
import torch
import pandas as pd
from tqdm import tqdm
from textSummarizer.entity import ModelEvaluationConfig

import gc
import torch
from tqdm import tqdm
import evaluate
import pandas as pd
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class ModelEvaluation:
    def __init__(self, config):
        self.config = config

    def generate_batch_sized_chunks(self, list_of_elements, batch_size):
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i:i + batch_size]

    def calculate_metric_on_test_ds(
        self,
        dataset,
        metric,
        model,
        tokenizer,
        batch_size=1,
        device="cpu",
        column_text="dialogue",
        column_summary="summary"
    ):

        model.eval()

        article_batches = list(
            self.generate_batch_sized_chunks(dataset[column_text], batch_size)
        )

        target_batches = list(
            self.generate_batch_sized_chunks(dataset[column_summary], batch_size)
        )

        for article_batch, target_batch in tqdm(
            zip(article_batches, target_batches),
            total=len(article_batches)
        ):

            inputs = tokenizer(
                article_batch,
                max_length=128,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )

            # ✅ FIX: ensure SAME device as model
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                summaries = model.generate(
                    **inputs,
                    max_length=32,
                    num_beams=1,
                    early_stopping=True
                )

            decoded_summaries = tokenizer.batch_decode(
                summaries,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            metric.add_batch(
                predictions=decoded_summaries,
                references=target_batch
            )

            del inputs
            del summaries
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return metric.compute()

    def evaluate(self):

        device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)

        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_path
        ).to(device)   # ✅ FIX: model moved to GPU

        dataset_samsum = load_from_disk(self.config.data_path)

        rouge_metric = evaluate.load("rouge")

        score = self.calculate_metric_on_test_ds(
            dataset=dataset_samsum["test"].select(range(10)),  # small test for speed
            metric=rouge_metric,
            model=model,
            tokenizer=tokenizer,
            batch_size=2,
            device=device,
            column_text="dialogue",
            column_summary="summary"
        )

        print(score)

        pd.DataFrame([score]).to_csv(
            self.config.metric_file_name,
            index=False
        )

        return score

