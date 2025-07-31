# VLM Evaluation on SROIE2019

This project provides scripts and tools to evaluate Vision-Language Models (VLMs), such as OpenAI's GPT-4o, Gemini (Google Generative AI), and Vertex AI Gemini models, on the SROIE2019 dataset. The evaluation focuses on robust normalization and fair calculation of micro-averaged entity-level F1 scores for key fields: company, date, address, and total.

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
- [LangChain](https://python.langchain.com/)
- [langchain-openai](https://pypi.org/project/langchain-openai/)
- [langchain-google-genai](https://pypi.org/project/langchain-google-genai/) (for Gemini)
- [langchain-google-vertexai](https://pypi.org/project/langchain-google-vertexai/) (for Vertex AI Gemini)
- An API key for your chosen provider (see below)


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
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   # Or install manually:
   pip install openai python-dotenv langchain langchain-openai langchain-google-genai langchain-google-vertexai pydantic
   ```
4. Download the SROIE2019 dataset (see Acknowledgments below) and place it in the correct folder structure as shown in this repo.


### Usage
1. Edit `.env` to include your API credentials for the provider you want to use:
   - **OpenAI**:
     ```
     OPENAI_API_KEY=sk-...
     ```
   - **Gemini (Google Generative AI)**:
     ```
     GOOGLE_API_KEY=your-google-api-key
     ```
   - **Vertex AI Gemini**:
     ```
     GOOGLE_APPLICATION_CREDENTIALS=path/to/your/service-account.json
     # (Service account must have Vertex AI User permissions)
     ```
2. Run the evaluation script:
   ```sh
   python evaluation_scripts/calculate_f1.py
   ```
3. To test a specific file range or debug mismatches, edit the main block in `calculate_f1.py` as described in the comments.

**Note:**
- For Vertex AI, your service account JSON file must exist at the path specified in `.env` and have the correct permissions.
- The script will automatically use the correct credentials based on your `.env` and the model/provider you select in the code.

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
results/
    4o.txt
    gemini-2.0-flash-001.txt
    gemini-2.5-flash.txt
```


## Acknowledgments
- **SROIE2019 Dataset**: This project uses the SROIE2019 dataset, which was obtained from [Kaggle](https://www.kaggle.com/datasets/urbikn/sroie-datasetv2). Please cite the original dataset and respect its license and terms of use.
- **OpenAI**: For providing the GPT-4o and GPT-4o-mini models and API.
- **Google**: For providing Gemini and Vertex AI Gemini models and APIs.

## License
This project is for research and educational purposes. Please check the licenses of the SROIE2019 dataset and any third-party models or APIs you use.
