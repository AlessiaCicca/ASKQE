import json
import nltk
import argparse
import csv
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import os


# =========================
# MEAN POOLING (SBERT)
# =========================
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


nltk.download("punkt")

# =========================
# ARGS
# =========================
parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_file",
    type=str,
    default="results/sbert_results.csv",
    help="CSV file where SBERT results will be saved",
)
args = parser.parse_args()


# ======================================================================
# ORIGINAL ASKQE (COMMENTED)
# ======================================================================
# languages = ["es", "fr", "hi", "tl", "zh"]
# pipelines = ["vanilla", "semantic", "atomic"]
# perturbations = [
#     "synonym", "word_order", "spelling", "expansion_noimpact",
#     "intensifier", "expansion_impact", "omission", "alteration"
# ]


# ======================================================================
# ADAPTED SETUP — SRC vs BT
# ======================================================================

pipelines = ["vanilla", "atomic", "semantic"]
results_dir = "/content/results"

reference_prefix = "src"          # qa_src-*.jsonl
bt_prefixes = ["mt1", "mt2"]      # qa_bt1-*.jsonl, qa_bt2-*.jsonl


# ======================================================================
# LOAD SBERT MODEL
# ======================================================================
tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)
model = AutoModel.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)
model.eval()


# =========================
# OUTPUT FILE
# =========================
os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

with open(args.output_file, mode="w", newline="", encoding="utf-8") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(
        ["pipeline", "reference", "compared", "avg_sbert_similarity", "num_comparisons"]
    )

    # =========================
    # MAIN LOOP
    # =========================
    for pipeline in pipelines:
        print("=" * 80)
        print(f"Pipeline: {pipeline}")

        reference_file = f"{results_dir}/qa_{reference_prefix}-{pipeline}.jsonl"
        print(f"Reference file: {reference_file}")

        if not os.path.exists(reference_file):
            print(f"[SKIP] Missing reference file for pipeline: {pipeline}")
            continue

        for bt in bt_prefixes:
            predicted_file = f"{results_dir}/qa_{bt}-{pipeline}.jsonl"
            print(f"Compared file: {predicted_file}")

            if not os.path.exists(predicted_file):
                print(f"[SKIP] Missing BT file for {bt} ({pipeline})")
                continue

            total_cosine_similarity = 0.0
            num_comparisons = 0

            with open(predicted_file, "r", encoding="utf-8") as pred_file, \
                 open(reference_file, "r", encoding="utf-8") as ref_file:

                for pred_line, ref_line in zip(pred_file, ref_file):
                    try:
                        pred_data = json.loads(pred_line)
                        ref_data = json.loads(ref_line)

                        pred_answer = pred_data.get("answers", "").strip()
                        ref_answer = ref_data.get("answers", "").strip()

                        if not pred_answer or not ref_answer:
                            continue

                        encoded_pred = tokenizer(
                            pred_answer, padding=True, truncation=True, return_tensors="pt"
                        )
                        encoded_ref = tokenizer(
                            ref_answer, padding=True, truncation=True, return_tensors="pt"
                        )

                        with torch.no_grad():
                            pred_output = model(**encoded_pred)
                            ref_output = model(**encoded_ref)

                        pred_embed = mean_pooling(pred_output, encoded_pred["attention_mask"])
                        ref_embed = mean_pooling(ref_output, encoded_ref["attention_mask"])

                        pred_embed = F.normalize(pred_embed, p=2, dim=1)
                        ref_embed = F.normalize(ref_embed, p=2, dim=1)

                        cos_sim = F.cosine_similarity(pred_embed, ref_embed).item()

                        total_cosine_similarity += cos_sim
                        num_comparisons += 1

                    except json.JSONDecodeError:
                        continue

            if num_comparisons > 0:
                avg_cosine_similarity = total_cosine_similarity / num_comparisons

                print(f"SRC vs {bt}")
                print(f"Num comparisons: {num_comparisons}")
                print(f"Average SBERT similarity: {avg_cosine_similarity:.4f}")

                csv_writer.writerow([
                    pipeline,
                    reference_prefix,
                    bt,
                    avg_cosine_similarity,
                    num_comparisons
                ])
            else:
                print(f"No valid comparisons for SRC vs {bt} ({pipeline})")
