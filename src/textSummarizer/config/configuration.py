from dataclasses import dataclass
from pathlib import Path

from textSummarizer.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from textSummarizer.utils.common import create_directories, read_yaml
from textSummarizer.entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelEvaluationConfig
)


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    data_path: Path
    model_ckpt: str

    num_train_epochs: int
    warmup_steps: int
    per_device_train_batch_size: int
    weight_decay: float
    logging_steps: int
    evaluation_strategy: str
    eval_steps: int
    save_steps: float
    gradient_accumulation_steps: int


class ConfigurationManager:
    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        cfg = self.config.data_ingestion
        create_directories([cfg.root_dir])
        return DataIngestionConfig(
            root_dir=cfg.root_dir,
            source_URL=cfg.source_URL,
            local_data_file=cfg.local_data_file,
            unzip_dir=cfg.unzip_dir,
        )

    def get_data_validation_config(self) -> DataValidationConfig:
        cfg = self.config.data_validation
        create_directories([cfg.root_dir])
        return DataValidationConfig(
            root_dir=cfg.root_dir,
            STATUS_FILE=cfg.STATUS_FILE,
            ALL_REQUIRED_FILES=cfg.ALL_REQUIRED_FILES,
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        cfg = self.config.data_transformation
        create_directories([cfg.root_dir])
        return DataTransformationConfig(
            root_dir=cfg.root_dir,
            data_path=cfg.data_path,
            tokenizer_name=cfg.tokenizer_name,
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.TrainingArguments   # NOTE: params should come from self.params

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
             root_dir=config.root_dir,
             data_path=config.data_path,
            model_ckpt=config.model_ckpt,

            num_train_epochs=params.num_train_epochs,
            warmup_steps=params.warmup_steps,
            per_device_train_batch_size=params.per_device_train_batch_size,
            weight_decay=params.weight_decay,
            logging_steps=params.logging_steps,
            evaluation_strategy=params.evaluation_strategy,
            eval_steps=params.eval_steps,
            save_steps=params.save_steps,
            gradient_accumulation_steps=params.gradient_accumulation_steps
        )   

        return model_trainer_config
        
    
    def get_model_evaluation_config(self):

        config = self.config.model_evaluation

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            model_path=config.model_path,
            tokenizer_path=config.tokenizer_path,
            metric_file_name=config.metric_file_name
        )

        return model_evaluation_config
        
