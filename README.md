# VLM Evaluation on SROIE2019

This project provides scripts and tools to evaluate Vision-Language Models (VLMs), such as OpenAI's GPT-4o and GPT-4o-mini, on the SROIE2019 dataset. The evaluation focuses on robust normalization and fair calculation of micro-averaged entity-level F1 scores for key fields: company, date, address, and total.

## Features
- **Robust normalization** for dates (including ambiguous formats and month names), currency, and text fields
- **Fuzzy matching** for address and company fields
- **Detailed mismatch reporting** for debugging and model improvement
- **Batch and range-based evaluation**
- **Secure environment management** using `.env` and `.gitignore`

## Getting Started

### Prerequisites
- Python 3.10+
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [python-dotenv](https://pypi.org/project/python-dotenv/)
- An OpenAI API key (set in a `.env` file as `OPENAI_API_KEY`)

### Installation
1. Clone this repository:
   ```sh
   git clone <your-repo-url>
   cd VLM_evaluation
   ```
2. Create and activate a virtual environment:
   ```sh
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   # Or
   source venv/bin/activate  # On Linux/Mac
   ```
3. Download the SROIE2019 dataset (see Acknowledgments below) and place it in the correct folder structure as shown in this repo.

### Usage
- Edit `.env` to include your OpenAI API key:
  ```
  OPENAI_API_KEY=sk-...
  ```
- Run the evaluation script:
  ```sh
  python evaluation_scripts/calculate_f1.py
  ```
- To test a specific file range or debug mismatches, edit the main block in `calculate_f1.py` as described in the comments.

## Folder Structure
```
evaluation_scripts/
    calculate_f1.py
SROIE2019/
    layoutlm-base-uncased/
    test/
        box/
        entities/
        img/
    train/
        box/
        entities/
        img/
```

## Acknowledgments
- **SROIE2019 Dataset**: This project uses the SROIE2019 dataset, which was obtained from [Kaggle](https://www.kaggle.com/datasets/urbikn/sroie-datasetv2). Please cite the original dataset and respect its license and terms of use.
- **OpenAI**: For providing the GPT-4o and GPT-4o-mini models and API.

## License
This project is for research and educational purposes. Please check the licenses of the SROIE2019 dataset and any third-party models or APIs you use.
