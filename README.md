# Text Summarization using FLAN-T5

## Overview

This project implements an end-to-end Text Summarization pipeline using Google's FLAN-T5 model and the SAMSum dataset. The application is designed to automatically generate concise summaries from conversational text.

The project follows a modular machine learning pipeline architecture, including data ingestion, validation, transformation, model training, evaluation, and deployment through FastAPI.

---

## Features

* Automated data ingestion and preprocessing
* Dataset validation pipeline
* Text tokenization and transformation using Hugging Face Transformers
* Fine-tuning of FLAN-T5 on the SAMSum dataset
* ROUGE-based model evaluation
* FastAPI-based deployment
* Modular and production-ready project structure
* YAML-based configuration management
* Logging and exception handling

---

## Tech Stack

### Machine Learning

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* Evaluate

### Backend

* FastAPI
* Uvicorn

### Utilities

* Pandas
* PyYAML
* tqdm

---

## Project Structure

```text
Text-Summerizer-project/
│
├── artifacts/
│   ├── data_ingestion/
│   ├── data_validation/
│   ├── data_transformation/
│   ├── model_trainer/
│   └── model_evaluation/
│
├── config/
│   └── config.yaml
│
├── research/
│
├── src/
│   └── textSummarizer/
│       ├── components/
│       ├── pipeline/
│       ├── config/
│       ├── entity/
│       ├── utils/
│       └── logging/
│
├── app.py
├── main.py
├── params.yaml
├── requirements.txt
└── README.md
```

---

## Dataset

This project uses the SAMSum dataset, which contains human-written summaries of messenger-like conversations.

Dataset Features:

* Dialogue Text
* Human Generated Summary

---

## Installation

### Clone Repository

```bash
git clone <your-repository-url>
cd Text-Summerizer-project
```

### Create Virtual Environment

```bash
python -m venv textS
```

### Activate Environment

#### Windows

```bash
textS\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

### Data Ingestion

```bash
python main.py
```

This will:

* Download dataset
* Validate files
* Transform data
* Train model
* Evaluate model

---

## Model Training

The project uses:

```text
google/flan-t5-small
```

Training parameters can be configured inside:

```text
params.yaml
```

---

## Model Evaluation

Evaluation is performed using ROUGE metrics:

* ROUGE-1
* ROUGE-2
* ROUGE-L
* ROUGE-Lsum

Evaluation results are stored in:

```text
artifacts/model_evaluation/metrics.csv
```

---

## FastAPI Deployment

Run the API server:

```bash
uvicorn app:app --reload
```

API will be available at:

```text
http://127.0.0.1:8000
```

Swagger Documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Example Request

```json
{
  "text": "Hello! How are you today? I wanted to discuss our project meeting scheduled for tomorrow."
}
```

### Example Response

```json
{
  "summary": "Discussion about tomorrow's project meeting."
}
```

---

## Future Improvements

* Deploy using Docker
* Deploy on AWS/GCP/Azure
* Add model monitoring
* Experiment with larger FLAN-T5 variants
* Add authentication to API
* Build Streamlit frontend

---

## Author

Kunal Singh Tanwar

B.Tech - Artificial Intelligence & Data Science

---

## License

This project is intended for educational and learning purposes.
