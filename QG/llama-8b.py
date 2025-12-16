from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import argparse
import sys


from prompt import prompts
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

INPUT_PATH = "/content/data/src_mt_bt.jsonl"


def main():
    # =========================
    # LOAD MODEL & TOKENIZER
    # =========================
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto"   # accelerate gestisce CPU / GPU / disk
    )

    # padding corretto per LLaMA
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    # =========================
    # ARGS
    # =========================
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    args = parser.parse_args()

    # =========================
    # LOAD DATASET
    # =========================
    with open(INPUT_PATH, "r", encoding="utf-8") as f_in, \
         open(args.output_path, "a", encoding="utf-8") as f_out:

        for line in f_in:
            data = json.loads(line)
            sentence = data.get("src")

            if not sentence:
                continue

            print(f"[SRC] {sentence}")

            prompt_template = prompts[args.prompt]

            # -------- prompt construction --------
            if args.prompt == "semantic":
                semantic = data.get("semantic_roles")
                if semantic:
                    prompt = prompt_template.replace(
                        "{{sentence}}", sentence
                    ).replace(
                        "{{semantic_roles}}", semantic
                    )
                else:
                    prompt = prompt_template.replace("{{sentence}}", sentence)

            elif args.prompt == "atomic":
                atomics = data.get("atomic_facts")
                if atomics:
                    prompt = prompt_template.replace(
                        "{{sentence}}", sentence
                    ).replace(
                        "{{atomic_facts}}", str(atomics)
                    )
                else:
                    prompt = prompt_template.replace("{{sentence}}", sentence)

            else:  # vanilla
                prompt = prompt_template.replace("{{sentence}}", sentence)

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]

            # =========================
            # CHAT TEMPLATE → TESTO
            # =========================
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # =========================
            # TOKENIZE → DICT
            # =========================
            inputs = tokenizer(
                prompt_text,
                return_tensors="pt"
            ).to(model.device)

            # =========================
            # GENERATE
            # =========================
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=32,              # corto e veloce
                    eos_token_id=tokenizer.eos_token_id
                )

            # =========================
            # DECODE
            # =========================
            response = outputs[0][inputs["input_ids"].shape[-1]:]
            questions = tokenizer.decode(
                response,
                skip_special_tokens=True
            ).strip()

            print(f"> {questions}")
            print("=" * 60)

            data["questions"] = questions
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
