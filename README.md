# VLM Evaluation on SROIE2019

This project provides scripts and tools to evaluate Vision-Language Models (VLMs), such as OpenAI's GPT-4o, Gemini (Google Generative AI), and Vertex AI Gemini models, on the SROIE2019 dataset. The evaluation focuses on robust normalization and fair calculation of micro-averaged entity-level F1 scores for key fields: company, date, address, and total.

## Features
- **Robust normalization** for dates (including ambiguous formats and month names), currency, and text fields
- **Fuzzy matching** for address and company fields (90% similarity threshold)
- **Detailed mismatch reporting** for debugging and model improvement
- **Multiple provider support**: OpenAI, Google Generative AI, and Vertex AI
- **Jupyter notebook interface** for interactive evaluation
- **Secure environment management** using `.env` and `.gitignore`

## Getting Started

### Prerequisites
- Python 3.10+
- A virtual environment (recommended)
- An API key/credentials for your chosen provider (see Usage section)

### Installation
1. Clone this repository:
   ```sh
   git clone <your-repo-url>
   cd VLM_evaluation
   ```
2. Create and activate a virtual environment:
   ```sh
   python -m venv venv
   .\venv\Scripts\Activate  # On Windows PowerShell
   # Or
   source venv/bin/activate  # On Linux/Mac
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
   
   **Required packages:**
   - `python-dotenv`
   - `langchain`
   - `langchain-openai`
   - `langchain-google-genai` (for Gemini)
   - `langchain-google-vertexai` (for Vertex AI)
   - `pydantic`

4. Download the SROIE2019 dataset (see Acknowledgments below) and place it in the correct folder structure as shown in this repo.

### Usage
1. Create a `.env` file in the project root and add your API credentials:
   
   **For OpenAI:**
   ```
   OPENAI_API_KEY=sk-...
   ```
   
   **For Gemini (Google Generative AI):**
   ```
   GOOGLE_API_KEY=your-google-api-key
   ```
   
   **For Vertex AI Gemini:**
   ```
   GOOGLE_APPLICATION_CREDENTIALS=path/to/your/service-account.json
   ```
   *Note: Service account must have Vertex AI User permissions*

2. **Choose your evaluation method:**
   
   **Option A: Python Script**
   ```sh
   python evaluation_scripts/calculate_f1.py
   ```
   
   **Option B: Jupyter Notebook**
   ```sh
   jupyter notebook evaluation_scripts/calculate_f1_notebook.ipynb
   ```

3. **Configure the model:**
   - Edit the model configuration in the script/notebook
   - Uncomment the desired model provider (OpenAI/Gemini/Vertex AI)
   - Update model names as needed

### Supported Models
- **OpenAI**: `gpt-4o`, `gpt-4o-mini`, `gpt-4-vision-preview`
- **Google Generative AI**: `gemini-1.5-pro-latest`, `gemini-pro-vision`
- **Vertex AI**: `gemini-1.5-pro`, `gemini-1.0-pro-vision`, etc.

## Folder Structure
```
VLM_evaluation/
├── evaluation_scripts/
│   ├── calculate_f1.py
│   └── calculate_f1_notebook.ipynb
├── SROIE2019/
│   └── test/
│       ├── box/
│       ├── entities/
│       └── img/
├── results/
│   ├── 4o.txt
│   ├── gemini-2.0-flash-001.txt
│   └── gemini-2.5-flash.txt
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

## Evaluation Metrics
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1 Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **Per-field analysis** for each entity type
- **Fuzzy matching** for supplier name and address (90% similarity threshold)
- **Exact matching** for dates and amounts

## Troubleshooting
- **Authentication errors**: Ensure your `.env` file contains the correct API keys/credentials
- **Import errors**: Activate your virtual environment and install all requirements
- **Dataset path issues**: Verify the SROIE2019 dataset structure matches the expected format
- **Vertex AI permissions**: Ensure your service account has proper Vertex AI access

## Acknowledgments
- **SROIE2019 Dataset**: This project uses the SROIE2019 dataset, obtained from [Kaggle](https://www.kaggle.com/datasets/urbikn/sroie-datasetv2). Please cite the original dataset and respect its license and terms of use.
- **OpenAI**: For providing GPT-4o and vision models
- **Google**: For providing Gemini and Vertex AI models
- **LangChain**: For the unified interface to multiple LLM providers

## License
This project is for research and educational purposes. Please check the licenses of the SROIE2019 dataset and any third-party models or APIs you use.