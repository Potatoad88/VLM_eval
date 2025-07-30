import os
import json
import base64
from openai import OpenAI
from dotenv import load_dotenv # For loading API key from .env file
import mimetypes # To determine image type from file extension
from datetime import datetime # For date normalization

# --- Load environment variables from .env file ---
load_dotenv()

# --- Configuration ---
# Path to the SROIE dataset folder.
# 'os.path.dirname(__file__)' gets the directory of the current script.
# '..' goes up one level to the parent directory (VLM_evaluation/).
# Then, it navigates into 'SROIE2019/test'.
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "SROIE2019", "test")
# You can change 'test' to 'train' or 'val' if you have one.

# OpenAI model to use
# MODEL_NAME = "gpt-4o-mini" # Or "gpt-4o", "gpt-4-vision-preview"
MODEL_NAME = "gpt-4o"

# Fields to evaluate (must match your prompt and ground truth keys)
FIELDS_TO_EVALUATE = ['company', 'date', 'address', 'total']

# --- VLM Base Class (as per your structure) ---
class VLM:
    """Base class for Vision Language Models."""
    def __init__(self):
        pass

    @staticmethod
    def path_2_b64(image_path, image_size=None):
        """
        Converts an image file to a base64 string and determines its MIME type.
        image_size is not used here but kept for compatibility with your signature.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at: {image_path}")

        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type is None or not mime_type.startswith('image/'):
            raise ValueError(f"Could not determine image MIME type or it's not an image: {image_path}")

        with open(image_path, "rb") as image_file:
            img_b64_str = base64.b64encode(image_file.read()).decode("utf-8")

        return img_b64_str, mime_type.split('/')[-1] # Returns base64 string and type (e.g., 'jpeg', 'png')

# --- GPT4oMini Model Class ---
class GPT4oMini(VLM):
    def __init__(self, api_key):
        super().__init__()
        # Initialize OpenAI client with the provided API key
        self.client = OpenAI(api_key=api_key)

    def predict(self, image_path, prompt, *, image_size=None, **kwargs):
        """
        Sends an image and prompt to the GPT-4o-mini model and returns the raw JSON response content.
        """
        try:
            img_b64_str, image_type = self.path_2_b64(image_path, image_size)
            
            # Construct the messages for the OpenAI API call
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{image_type};base64,{img_b64_str}"},
                        },
                    ],
                }
            ]

            # Make the API call
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=1000, # Limit response length to prevent excessive cost/tokens
                response_format={"type": "json_object"} # Crucial for getting JSON output
            )
            
            # Extract the content from the response
            # The response.choices[0].message.content is already a JSON string because of response_format
            return response.choices[0].message.content

        except FileNotFoundError as e:
            print(f"Error: {e}")
            return None
        except ValueError as e:
            print(f"Error processing image: {e}")
            return None
        except Exception as e:
            print(f"OpenAI API error for {image_path}: {e}")
            return None
        
    @staticmethod
    def get_parsed_output(pred_json_str):
        """
        Parses the JSON string output from the model into a Python dictionary.
        """
        if pred_json_str is None:
            return None
        try:
            return json.loads(pred_json_str)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from model output: {e}\nRaw output: {pred_json_str}")
            return None

# --- Normalization function (critical for accurate evaluation) ---
def normalize_value(key, value):
    """
    Normalizes a value based on its key for consistent comparison.
    This is a basic implementation and might need to be expanded for real-world complexity.
    """
    if value is None:
        return None
    
    # Ensure value is a string for common operations
    value = str(value).strip()

    if key == "date":
        import re
        from datetime import datetime
        # Try to parse using common date formats
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
        # Remove extra spaces
        val = re.sub(r"\s+", " ", val)
        for fmt in formats:
            try:
                dt = datetime.strptime(val, fmt)
                return dt.strftime("%d/%m/%Y")
            except Exception:
                continue
        # Try to handle 2-digit year at end (e.g., 20-11-17 or 23.03.18)
        m = re.match(r"(\d{2})[./-](\d{2})[./-](\d{2})$", val)
        if m:
            d, mth, y = m.groups()
            # Assume 2000+ for year < 30, else 1900+
            y = int(y)
            y = 2000 + y if y < 30 else 1900 + y
            try:
                dt = datetime(y, int(mth), int(d))
                return dt.strftime("%d/%m/%Y")
            except Exception:
                pass
        # If prediction is in YYYY-MM-DD, convert to DD/MM/YYYY
        parts = val.split("-")
        if len(parts) == 3 and all(part.isdigit() for part in parts):
            yyyy, mm, dd = parts
            return f"{dd.zfill(2)}/{mm.zfill(2)}/{yyyy}"
        return val

    elif key == "total":
        # Remove currency symbols, commas, and standardize to float with 2 decimal places
        value = value.replace('$', '').replace('€', '').replace('£', '').replace(',', '').replace('RM', '').strip()
        try:
            return f"{float(value):.2f}"
        except ValueError:
            return value.lower() # Fallback to lowercased original if not a valid number
    
    # For company and address, remove punctuation, spaces, and lowercase for strict matching
    if key in ["address", "company"]:
        import string
        # Remove punctuation
        value = value.translate(str.maketrans('', '', string.punctuation))
        # Remove all whitespace
        value = ''.join(value.split())
        # Lowercase
        return value.lower()
    return value.strip().lower()

# --- Main Evaluation Logic ---
def evaluate_sroie_dataset(dataset_path, model_instance, fields_to_evaluate):
    """
    Evaluates the model's performance on the SROIE dataset.
    """
    all_ground_truths = []
    all_predictions = []

    image_dir = os.path.join(dataset_path, "img")
    entity_dir = os.path.join(dataset_path, "entities")

    # Get list of image files (assuming corresponding entity files exist)
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Found {len(image_files)} images in {image_dir}")

    for i, image_file in enumerate(image_files):
        receipt_id = os.path.splitext(image_file)[0] # e.g., 'invoice_001'
        image_path = os.path.join(image_dir, image_file)
        entity_path = os.path.join(entity_dir, f"{receipt_id}.txt") # SROIE uses .txt for entities

        if not os.path.exists(entity_path):
            print(f"Warning: Ground truth file not found for {receipt_id} at {entity_path}. Skipping.")
            continue

        print(f"Processing ({i+1}/{len(image_files)}): {receipt_id}")

        # Load Ground Truth
        try:
            with open(entity_path, "r", encoding="utf-8") as f:
                # SROIE entities are JSON-like. Load as string then parse.
                gt_raw = f.read()
                ground_truth = json.loads(gt_raw)
        except Exception as e:
            print(f"Error reading ground truth for {receipt_id}: {e}. Skipping.")
            continue

        # Get Model Prediction
        # The prompt is critical for guiding the model's output format
        prompt = (
            "You are an expert at extracting information from scanned receipts. "
            "Extract the 'company', 'date' (format YYYY-MM-DD), 'address', and 'total' from this receipt. "
            "If a field is not present, omit it. Output the information as a JSON object."
        )
        model_raw_output = model_instance.predict(image_path, prompt)
        prediction = model_instance.get_parsed_output(model_raw_output)

        if prediction is None:
            print(f"Could not get valid prediction for {receipt_id}. Skipping.")
            continue

        # Normalize GT and Prediction for comparison (only for fields we want to evaluate)
        normalized_gt = {k: normalize_value(k, v) for k, v in ground_truth.items() if k in FIELDS_TO_EVALUATE}
        normalized_pred = {k: normalize_value(k, v) for k, v in prediction.items() if k in FIELDS_TO_EVALUATE}

        all_ground_truths.append(normalized_gt)
        all_predictions.append(normalized_pred)

    return all_ground_truths, all_predictions

# --- F1 Score Calculation (Micro-averaged Entity-Level F1) ---
def calculate_f1_score(ground_truths, predictions, fields_to_evaluate):
    """
    Calculates micro-averaged entity-level F1 score.
    A match is counted if both the key and the normalized value are identical.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    import difflib
    for gt_doc, pred_doc in zip(ground_truths, predictions):
        for field in fields_to_evaluate:
            gt_value = gt_doc.get(field)
            pred_value = pred_doc.get(field)

            # Fuzzy match for address/company
            if field in ["address", "company"] and gt_value is not None and pred_value is not None:
                ratio = difflib.SequenceMatcher(None, gt_value, pred_value).ratio()
                if ratio >= 0.90:
                    total_tp += 1
                else:
                    total_fp += 1
                    total_fn += 1
            # Exact match for other fields
            elif gt_value is not None and pred_value is not None:
                if gt_value == pred_value:
                    total_tp += 1
                else:
                    total_fp += 1
                    total_fn += 1
            elif gt_value is not None and pred_value is None:
                total_fn += 1
            elif gt_value is None and pred_value is not None:
                total_fp += 1

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn
    }



def print_not_true_positive_fields_for_range(start=0, end=10):
    """
    For image files in the range [start, end), print only the fields that are not true positive (normalized values do not match).
    """
    image_dir = os.path.join(DATASET_PATH, "img")
    entity_dir = os.path.join(DATASET_PATH, "entities")
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files = sorted(image_files)[start:end]
    model = GPT4oMini(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = (
        "You are an expert at extracting information from scanned receipts. "
        "Extract the 'company', 'date' (format YYYY-MM-DD), 'address', and 'total' from this receipt. "
        "If a field is not present, omit it. Output the information as a JSON object."
    )
    import difflib
    for idx, image_filename in enumerate(image_files, start + 1):
        image_path = os.path.join(image_dir, image_filename)
        entity_path = os.path.join(entity_dir, image_filename.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))
        if not os.path.exists(entity_path):
            print(f"[{idx}] {image_filename}: Ground truth file not found: {entity_path}")
            continue
        with open(entity_path, "r", encoding="utf-8") as f:
            ground_truth = json.loads(f.read())
        model_raw_output = model.predict(image_path, prompt)
        prediction = model.get_parsed_output(model_raw_output)
        mismatches = []
        for field in FIELDS_TO_EVALUATE:
            gt_val = ground_truth.get(field)
            pred_val = prediction.get(field) if prediction else None
            norm_gt = normalize_value(field, gt_val)
            norm_pred = normalize_value(field, pred_val)
            if field in ["address", "company"] and norm_gt is not None and norm_pred is not None:
                ratio = difflib.SequenceMatcher(None, norm_gt, norm_pred).ratio()
                if ratio < 0.90:
                    mismatches.append((field, gt_val, pred_val, norm_gt, norm_pred, ratio))
            else:
                if norm_gt != norm_pred:
                    mismatches.append((field, gt_val, pred_val, norm_gt, norm_pred, None))
        if mismatches:
            print(f"\n[{idx}] {image_filename} - Fields NOT true positive:")
            for field, gt_val, pred_val, norm_gt, norm_pred, ratio in mismatches:
                print(f"  Field: {field}")
                print(f"    Ground Truth: {repr(gt_val)}")
                print(f"    Prediction:   {repr(pred_val)}")
                print(f"    Normalized GT: {repr(norm_gt)}")
                print(f"    Normalized Pred: {repr(norm_pred)}")
                if ratio is not None:
                    print(f"    Fuzzy ratio: {ratio:.2f}")

# --- Main Execution ---
if __name__ == "__main__":
    # Get API key from environment variable
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in a .env file or directly.")

    # Initialize the model
    model = GPT4oMini(api_key=openai_api_key)

    print(f"Starting evaluation on dataset: {DATASET_PATH} using model: {MODEL_NAME}")
    
    # To test a single file, uncomment the following line and set the filename:
    # Print fields that are not true positive for the first 10 files
    # print_not_true_positive_fields_for_range(100, 150)

    # # To run full evaluation, comment out the above and uncomment below:
    gt_data, pred_data = evaluate_sroie_dataset(DATASET_PATH, model, FIELDS_TO_EVALUATE)
    if not gt_data or not pred_data:
        print("No data processed for evaluation. Check dataset path and file existence.")
    else:
        # Calculate and print results
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
