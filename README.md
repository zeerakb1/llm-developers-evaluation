# Large Language Models as Responders in SE Tasks: A Study on StackExchange

This repository contains the datasets and scripts used in our study exploring the use of Large Language Models (LLMs) as research proxies in software engineering (SE) tasks. The study focuses on evaluating the ability of LLMs to generate human-like responses for technical queries on platforms such as StackExchange. The repository also supports tasks like fine-tuning and retrieval-augmented generation (RAG) for improving response quality.

---

## Dataset

All dataset files are located in the `Dataset/DataExcel/` folder in Excel format. Extract the zip file to access the following:

- `stackExchangeQsAndAnswersDB.xlsx` used as vector database for RAG application
- `stackExchangeQsAndAnswersTest.xlsx` used for evaluating responses
- `stackExchangeQsAndAnswersTrain.xlsx` used for training a LLM for finetuning

These files contain the data used for fine-tuning, evaluation, and response generation.

---

## Virtual Environment Setup

1. **Create a virtual environment:**

   ```bash
   python3 -m venv llm-eval-env
   ```

2. **Activate the virtual environment:**

   - On Linux/Mac:
     ```bash
     source llm-eval-env/bin/activate
     ```
   - On Windows:
     ```bash
     .\llm-eval-env\Scripts\activate
     ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

---

## Script Descriptions and Dataset Usage

### 1. **Base Setting Experiment**: `generateResponsesStackExchange.py`

**Description:**
Generates responses from different models (CodeLlama, Mistral, and Solar-70B) using the Stack Exchange dataset under base settings.

**Dataset Used:** `stackExchangeQsAndAnswersTest.xlsx`

**Usage:**
```bash
python Scripts/generateResponsesStackExchange.py
```

---

### 2. **RAG with FineTune**: `FineTune.py`

**Description:**
Fine-tunes the Solar-70B model using the training dataset from the Stack Exchange data.

**Dataset Used:** `stackExchangeQsAndAnswersTrain.xlsx`

**Usage:**
```bash
python FineTuning/FineTune.py
```

---

### 3. **RAG with FineTune**: `FineTune_and_Rag.py`

**Description:**
Fine-tunes Solar-70B and integrates it with a Retrieval-Augmented Generation (RAG) pipeline.

**Dataset Used:** `stackExchangeQsAndAnswersTrain.xlsx`

**Usage:**
```bash
python FineTuning/FineTune_and_Rag.py
```

---

### 4. **RAG with FineTune**: `FineTuneInference.py`

**Description:**
Uses the fine-tuned Solar-70B model to infer answers based on Stack Exchange queries.

**Dataset Used:** No dataset required for inference.

**Usage:**
```bash
python FineTuning/FineTuneInference.py
```

---

### 5. **RAG Architecture**: `stackExchangeLLMRag.py`

**Description:**
Implements a RAG pipeline using CodeLlama, Mistral, or Solar-70B with a vector database created from the Stack Exchange dataset.

**Dataset Used:** `stackExchangeQsAndAnswersDB.xlsx`

**Usage:**
```bash
python RAG/stackExchangeLLMRag.py
```

---

### 6. **Evaluation of Responses**: `evaluationStackExchange.py`

**Description:**
Evaluates model performance on the Stack Exchange test set. Provides metrics for generated responses.

**Dataset Used:** `stackExchangeQsAndAnswersTest.xlsx`

**Usage:**
```bash
python Scripts/evaluationStackExchange.py
```

---
