import os
import json
import base64
import mimetypes
from dotenv import load_dotenv
from datetime import datetime

# --- LangChain Imports ---
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage, SystemMessage

# --- LangChain Output Parsers ---
from langchain.output_parsers import PydanticOutputParser

# --- Pydantic for schema enforcement ---
from pydantic import BaseModel, Field

# --- Load environment variables from .env file ---
load_dotenv()

# --- Configuration ---
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "SROIE2019", "test")
MODEL_NAME = "gemini-2.0-flash"
FIELDS_TO_EVALUATE = ['supplierName', 'receiptDate', 'supplierAddress', 'totalAmount']
# --- Vertex AI Configuration ---
VERTEX_PROJECT = 'bionic-cosmos-419708'
VERTEX_LOCATION = 'us-west1'
VERTEX_MODEL = 'gemini-2.5-flash'

# --- Helper: Image to base64 ---
def path_2_b64(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None or not mime_type.startswith('image/'):
        raise ValueError(f"Could not determine image MIME type or it's not an image: {image_path}")
    with open(image_path, "rb") as image_file:
        img_b64_str = base64.b64encode(image_file.read()).decode("utf-8")
    return img_b64_str, mime_type.split('/')[-1]

# --- LangChain Model Call ---
def call_vlm_with_langchain(image_path, prompt, api_key):
    img_b64_str, image_type = path_2_b64(image_path)
    # Prepare the message for OpenAI Vision model
    messages = [
        HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{image_type};base64,{img_b64_str}"},
                },
            ]
        )
    ]
    # llm = ChatOpenAI(
    #     model=MODEL_NAME,
    #     openai_api_key=api_key,
    #     max_tokens=1000,
    #     temperature=0,
    #     model_kwargs={"response_format": {"type": "json_object"}},
    # )

    # llm = ChatGoogleGenerativeAI(
    #     model=MODEL_NAME,
    #     google_api_key=api_key,
    #     max_output_tokens=1000,
    #     temperature=0,
    # )

    llm = ChatVertexAI(
        model=VERTEX_MODEL,
        project=VERTEX_PROJECT,
        location=VERTEX_LOCATION,
        max_output_tokens=1000,
        temperature=0,
    )

    llm_with_parser = llm.with_structured_output(SROIEEntity)
    response = llm_with_parser.invoke(messages)
    llm_with_parser = llm.with_structured_output(SROIEEntity)
    response = llm_with_parser.invoke(messages)
    # The response is now a parsed SROIEEntity object
    return response

# --- Pydantic Model for SROIE Entity Extraction ---

class SROIEEntity(BaseModel):
    supplierName: str = Field(description="The supplier name on the receipt")
    receiptDate: str = Field(description="The date on the receipt in DD/MM/YYYY format")
    supplierAddress: str = Field(description="The supplier address on the receipt")
    totalAmount: str = Field(description="The total amount to pay, as a string")

# --- Output Parsers ---
pydantic_parser = PydanticOutputParser(pydantic_object=SROIEEntity)

# --- Normalization function ---
def normalize_value(key, value):
    if value is None:
        return None
    value = str(value).strip()
    if key == "receiptDate":
        import re
        formats = [
            "%Y-%m-%d", "%d-%m-%Y", "%d-%m-%y", "%y-%m-%d",
            "%Y/%m/%d", "%d/%m/%Y", "%d/%m/%y", "%y/%m/%d",
            "%Y.%m.%d", "%d.%m.%Y", "%d.%m.%y", "%y.%m.%d",
            "%d %b %Y", "%d %B %Y", "%d %b %y", "%d %B %y",
            "%d/%b/%Y", "%d/%B/%Y", "%d/%b/%y", "%d/%B/%y",
            "%d-%b-%Y", "%d-%B-%Y", "%d-%b-%y", "%d-%B-%y",
            "%d %m %Y", "%d %m %y",
            "%d%b%Y", "%d%b%y", "%d%B%Y", "%d%B%y",
            "%y%m%d", "%d%m%y", "%d%m%Y",
        ]
        val = value.strip()
        val = re.sub(r"\s+", " ", val)
        for fmt in formats:
            try:
                dt = datetime.strptime(val, fmt)
                return dt.strftime("%d/%m/%Y")
            except Exception:
                continue
        m = re.match(r"(\d{2})[./-](\d{2})[./-](\d{2})$", val)
        if m:
            d, mth, y = m.groups()
            y = int(y)
            y = 2000 + y if y < 30 else 1900 + y
            try:
                dt = datetime(y, int(mth), int(d))
                return dt.strftime("%d/%m/%Y")
            except Exception:
                pass
        parts = val.split("-")
        if len(parts) == 3 and all(part.isdigit() for part in parts):
            yyyy, mm, dd = parts
            return f"{dd.zfill(2)}/{mm.zfill(2)}/{yyyy}"
        return val
    elif key == "totalAmount":
        value = value.replace('$', '').replace('€', '').replace('£', '').replace(',', '').replace('RM', '').strip()
        try:
            return f"{float(value):.2f}"
        except ValueError:
            return value.lower()
    if key in ["supplierAddress", "supplierName"]:
        import string
        value = value.translate(str.maketrans('', '', string.punctuation))
        value = ''.join(value.split())
        return value.lower()
    return value.strip().lower()

# --- Main Evaluation Logic (LangChain version) ---
def evaluate_sroie_dataset_langchain(dataset_path, api_key, fields_to_evaluate):
    all_ground_truths = []
    all_predictions = []
    image_dir = os.path.join(dataset_path, "img")
    entity_dir = os.path.join(dataset_path, "entities")
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(image_files)} images in {image_dir}")
    for i, image_file in enumerate(image_files):
        receipt_id = os.path.splitext(image_file)[0]
        image_path = os.path.join(image_dir, image_file)
        entity_path = os.path.join(entity_dir, f"{receipt_id}.txt")
        if not os.path.exists(entity_path):
            print(f"Warning: Ground truth file not found for {receipt_id} at {entity_path}. Skipping.")
            continue
        print(f"Processing ({i+1}/{len(image_files)}): {receipt_id}")
        try:
            with open(entity_path, "r", encoding="utf-8") as f:
                gt_raw = f.read()
                ground_truth = json.loads(gt_raw)
        except Exception as e:
            print(f"Error reading ground truth for {receipt_id}: {e}. Skipping.")
            continue
        prompt = (
            "You are an expert at extracting information from scanned receipts. "
            "If a field is not present, omit it. Output only these fields as a flat JSON object."
        )
        try:
            prediction = call_vlm_with_langchain(image_path, prompt, api_key)
        except Exception as e:
            print(f"LangChain model call failed for {receipt_id}: {e}")
            prediction = None
        if prediction is None:
            print(f"Could not get valid prediction for {receipt_id}. Skipping.")
            continue
        # Map ground truth keys to new model keys
        gt_key_map = {
            'company': 'supplierName',
            'date': 'receiptDate',
            'address': 'supplierAddress',
            'total': 'totalAmount',
        }
        normalized_gt = {gt_key_map[k]: normalize_value(gt_key_map[k], v) for k, v in ground_truth.items() if k in gt_key_map}
        normalized_pred = {k: normalize_value(k, v) for k, v in prediction.model_dump().items() if k in fields_to_evaluate}
        all_ground_truths.append(normalized_gt)
        all_predictions.append(normalized_pred)
    return all_ground_truths, all_predictions

# --- F1 Score Calculation (same as before) ---
def calculate_f1_score(ground_truths, predictions, fields_to_evaluate):
    total_tp = 0
    total_fp = 0
    total_fn = 0
    import difflib
    # Per-field stats
    field_stats = {field: {"tp": 0, "fp": 0, "fn": 0} for field in fields_to_evaluate}
    for idx, (gt_doc, pred_doc) in enumerate(zip(ground_truths, predictions)):
        for field in fields_to_evaluate:
            gt_value = gt_doc.get(field)
            pred_value = pred_doc.get(field)
            # Fuzzy match for supplierAddress/supplierName
            if field in ["supplierAddress", "supplierName"] and gt_value is not None and pred_value is not None:
                ratio = difflib.SequenceMatcher(None, gt_value, pred_value).ratio()
                if ratio >= 0.90:
                    total_tp += 1
                    field_stats[field]["tp"] += 1
                else:
                    total_fp += 1
                    total_fn += 1
                    field_stats[field]["fp"] += 1
                    field_stats[field]["fn"] += 1
                    print(f"[Doc {idx+1}] Field '{field}' NOT true positive (fuzzy ratio {ratio:.2f} < 0.90):")
                    print(f"  Ground Truth: {repr(gt_value)}")
                    print(f"  Prediction:   {repr(pred_value)}")
            # Exact match for other fields
            elif gt_value is not None and pred_value is not None:
                if gt_value == pred_value:
                    total_tp += 1
                    field_stats[field]["tp"] += 1
                else:
                    total_fp += 1
                    total_fn += 1
                    field_stats[field]["fp"] += 1
                    field_stats[field]["fn"] += 1
                    print(f"[Doc {idx+1}] Field '{field}' NOT true positive (exact mismatch):")
                    print(f"  Ground Truth: {repr(gt_value)}")
                    print(f"  Prediction:   {repr(pred_value)}")
            # Prediction missing
            elif gt_value is not None and pred_value is None:
                total_fn += 1
                field_stats[field]["fn"] += 1
                print(f"[Doc {idx+1}] Field '{field}' NOT true positive (prediction missing):")
                print(f"  Ground Truth: {repr(gt_value)}")
                print(f"  Prediction:   None")
            # Ground truth missing
            elif gt_value is None and pred_value is not None:
                total_fp += 1
                field_stats[field]["fp"] += 1
                print(f"[Doc {idx+1}] Field '{field}' NOT true positive (ground truth missing):")
                print(f"  Ground Truth: None")
                print(f"  Prediction:   {repr(pred_value)}")
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    # Per-field F1
    per_field_scores = {}
    for field in fields_to_evaluate:
        tp = field_stats[field]["tp"]
        fp = field_stats[field]["fp"]
        fn = field_stats[field]["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_field = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        per_field_scores[field] = {
            "precision": prec,
            "recall": rec,
            "f1": f1_field,
            "tp": tp,
            "fp": fp,
            "fn": fn
        }
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "per_field": per_field_scores
    }

# --- Main Execution ---
if __name__ == "__main__":
    # openai_api_key = os.getenv("OPENAI_API_KEY")
    # if not openai_api_key:
    #     raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in a .env file or directly.")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in a .env file or directly.")
    print(f"Starting evaluation on dataset: {DATASET_PATH} using model: {MODEL_NAME} (via LangChain)")
    gt_data, pred_data = evaluate_sroie_dataset_langchain(DATASET_PATH, google_api_key, FIELDS_TO_EVALUATE)
    if not gt_data or not pred_data:
        print("No data processed for evaluation. Check dataset path and file existence.")
    else:
        results = calculate_f1_score(gt_data, pred_data, FIELDS_TO_EVALUATE)
        print("\n--- Micro-Averaged Entity-Level F1 Results ---")
        print(f"Evaluated fields: {', '.join(FIELDS_TO_EVALUATE)}")
        print(f"Total Documents Processed: {len(gt_data)}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall:    {results['recall']:.4f}")
        print(f"F1 Score:  {results['f1']:.4f}")
        print(f"True Positives (TP): {results['total_tp']}")
        print(f"False Positives (FP): {results['total_fp']}")
        print(f"False Negatives (FN): {results['total_fn']}")
        print("\n--- Per-Field F1 Scores ---")
        for field, stats in results["per_field"].items():
            print(f"Field: {field}")
            print(f"  Precision: {stats['precision']:.4f}")
            print(f"  Recall:    {stats['recall']:.4f}")
            print(f"  F1 Score:  {stats['f1']:.4f}")
            print(f"  TP: {stats['tp']}  FP: {stats['fp']}  FN: {stats['fn']}")