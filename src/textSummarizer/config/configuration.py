from dataclasses import dataclass
from pathlib import Path

from textSummarizer.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from textSummarizer.utils.common import create_directories, read_yaml
from textSummarizer.entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
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
        mt = self.config.model_trainer
        ta = self.params.TrainingArguments

        # fix typo in config.yaml (artifcats)
        root_dir = mt.root_dir

        create_directories([root_dir])

        return ModelTrainerConfig(
            root_dir=Path(root_dir),
            data_path=Path(mt.data_path),
            model_ckpt=str(mt.model_ckpt),
            num_train_epochs=int(ta.num_train_epochs),
            warmup_steps=int(ta.warmup_steps),
            per_device_train_batch_size=int(ta.per_device_train_batch_size),
            weight_decay=float(ta.weight_decay),
            logging_steps=int(ta.logging_steps),
            evaluation_strategy=str(ta.evaluation_strategy),
            eval_steps=int(ta.eval_steps),
            save_steps=float(ta.save_steps),
            gradient_accumulation_steps=int(ta.gradient_accumulation_steps),
        )

