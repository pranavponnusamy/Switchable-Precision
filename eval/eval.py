import sys
sys.path.append("..")


import argparse
import json
import os
import random
import torch
import gc
from tqdm import tqdm

from transformers import AutoTokenizer, GPT2LMHeadModel
from datasets import load_dataset
from google import genai
from google.genai import types

from utils.utils import QuantBlockConfig
from utils import utils
from utils import lora


# ============================================================================
# Quantization Configurations (from train.py)
# ============================================================================

DICT_CONFIGS = {
    # Default 32-bit config (no quantization) - used by train_adv.py
    "default": {
        i: {
            "Attention_W_bit": 32,
            "Attention_A_bit": 32,
            "Attention_KV_bit": 32,
            "MLP_W_bit": 32,
            "MLP_A_bit": 32,
        }
        for i in range(12)
    },
    "8-8-4_uniform": {
        i: {
            "Attention_W_bit": 8,
            "Attention_A_bit": 8,
            "Attention_KV_bit": 4,
            "MLP_W_bit": 8,
            "MLP_A_bit": 8,
        }
        for i in range(12)
    },
    "8-8-16_uniform": {
        i: {
            "Attention_W_bit": 8,
            "Attention_A_bit": 8,
            "Attention_KV_bit": 16,
            "MLP_W_bit": 8,
            "MLP_A_bit": 8,
        }
        for i in range(12)
    },
    "8-8-4_center_reduced": {
        i: {
            "Attention_W_bit": 4 if 5 <= i <= 9 else 8,
            "Attention_A_bit": 8,
            "Attention_KV_bit": 4,
            "MLP_W_bit": 4 if 5 <= i <= 9 else 8,
            "MLP_A_bit": 8,
        }
        for i in range(12)
    },
    "8-8-16_center_reduced": {
        i: {
            "Attention_W_bit": 4 if 5 <= i <= 9 else 8,
            "Attention_A_bit": 8 if 5 <= i <= 9 else 8,
            "Attention_KV_bit": 4 if 5 <= i <= 9 else 16,
            "MLP_W_bit": 4 if 5 <= i <= 9 else 8,
            "MLP_A_bit": 8,
        }
        for i in range(12)
    },
    "8-8-4_ends_reduced": {
        i: {
            "Attention_W_bit": 4 if (i <= 2 or i >= 10) else 8,
            "Attention_A_bit": 8,
            "Attention_KV_bit": 4,
            "MLP_W_bit": 4 if (i <= 2 or i >= 10) else 8,
            "MLP_A_bit": 8,
        }
        for i in range(12)
    },
    "8-8-16_ends_reduced": {
        i: {
            "Attention_W_bit": 4 if (i <= 2 or i >= 10) else 8,
            "Attention_A_bit": 8,
            "Attention_KV_bit": 16,
            "MLP_W_bit": 4 if (i <= 2 or i >= 10) else 8,
            "MLP_A_bit": 8,
        }
        for i in range(12)
    },
    "8-8-4_early_reduced": {
        i: {
            "Attention_W_bit": 4 if i <= 5 else 8,
            "Attention_A_bit": 8,
            "Attention_KV_bit": 4,
            "MLP_W_bit": 4 if i <= 5 else 8,
            "MLP_A_bit": 8,
        }
        for i in range(12)
    },
    "8-8-4_late_reduced": {
        i: {
            "Attention_W_bit": 4 if i >= 6 else 8,
            "Attention_A_bit": 8,
            "Attention_KV_bit": 4,
            "MLP_W_bit": 4 if i >= 6 else 8,
            "MLP_A_bit": 8,
        }
        for i in range(12)
    },
    "8-8-16_early_reduced": {
        i: {
            "Attention_W_bit": 4 if i <= 5 else 8,
            "Attention_A_bit": 8,
            "Attention_KV_bit": 16,
            "MLP_W_bit": 4 if i <= 5 else 8,
            "MLP_A_bit": 8,
        }
        for i in range(12)
    },
    "8-8-16_late_reduced": {
        i: {
            "Attention_W_bit": 4 if i >= 6 else 8,
            "Attention_A_bit": 8,
            "Attention_KV_bit": 16,
            "MLP_W_bit": 4 if i >= 6 else 8,
            "MLP_A_bit": 8,
        }
        for i in range(12)
    },
    "8-8-16_kv4_only_center": {
        i: {
            "Attention_W_bit": 8,
            "Attention_A_bit": 8,
            "Attention_KV_bit": 4 if 5 <= i <= 9 else 16,
            "MLP_W_bit": 8,
            "MLP_A_bit": 8,
        }
        for i in range(12)
    },
    "8-8-16_kv4_only_ends": {
        i: {
            "Attention_W_bit": 8,
            "Attention_A_bit": 8,
            "Attention_KV_bit": 4 if (i <= 2 or i >= 10) else 16,
            "MLP_W_bit": 8,
            "MLP_A_bit": 8,
        }
        for i in range(12)
    },
    "8-8-4_kv16_only_center": {
        i: {
            "Attention_W_bit": 8,
            "Attention_A_bit": 8,
            "Attention_KV_bit": 16 if 5 <= i <= 9 else 4,
            "MLP_W_bit": 8,
            "MLP_A_bit": 8,
        }
        for i in range(12)
    },
}

def get_lora_rank_from_checkpoint(checkpoint_path: str) -> int:
    """Auto-detect LoRA rank from checkpoint file."""
    state_dict = torch.load(checkpoint_path, weights_only=True)
    # Get first precision's weights
    first_precision = list(state_dict.keys())[0]
    weights = state_dict[first_precision]
    # Find a lora_A parameter and get its rank dimension
    for name, param in weights.items():
        if 'lora_A' in name:
            # lora_A shape is [in_features, r]
            return param.shape[1]
    raise ValueError("Could not detect LoRA rank from checkpoint")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate quantized GPT-2 with LoRA using Gemini validation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to LoRA checkpoint file")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to evaluate")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    parser.add_argument("--configs", type=str, nargs="+", default=None, 
                        help="Specific configs to evaluate (default: all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--max_new_tokens", type=int, default=30, help="Max tokens to generate")
    parser.add_argument("--quant_bit", type=int, default=None, 
                        help="Override with uniform quantization bit (e.g., 4, 8, 16, 32)")
    return parser.parse_args()


def load_model_with_lora(checkpoint_path: str, precisions: list, device: torch.device):
    """Load GPT-2, apply quantization and LoRA, then load checkpoint weights."""
    
    # Auto-detect LoRA rank from checkpoint
    lora_r = get_lora_rank_from_checkpoint(checkpoint_path)
    lora_alpha = lora_r * 2.0  # Standard scaling: alpha = 2 * r
    print(f"Detected LoRA rank: {lora_r}, alpha: {lora_alpha}")
    
    print("Loading GPT-2 base model...")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create quantization configs for each layer
    QUANT_CONFIGS = {i: QuantBlockConfig() for i in range(12)}
    
    # Build configs dict
    configs = {}
    for k in DICT_CONFIGS.keys():
        conf = [QuantBlockConfig.from_dict(DICT_CONFIGS[k][i]) for i in range(12)]
        configs[k] = {i: conf[i] for i in range(12)}
    
    print("Applying quantization...")
    utils.quantize_model(model, QUANT_CONFIGS)
    
    print("Applying LoRA adapters...")
    lora.apply_lora_to_model(
        model, precisions, r=lora_r, alpha=lora_alpha, lora_attention=True, lora_mlp=True
    )
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    lora.load_lora(model, checkpoint_path)
    
    model.to(device)
    model.eval()
    
    return model, tokenizer, QUANT_CONFIGS, configs


def sample_validation_data(num_samples: int, seed: int):
    """Sample N random examples from SQuAD validation set."""
    print(f"Loading SQuAD dataset and sampling {num_samples} examples...")
    ds = load_dataset("rajpurkar/squad")
    val_data = ds["validation"]
    
    random.seed(seed)
    indices = random.sample(range(len(val_data)), min(num_samples, len(val_data)))
    
    samples = []
    for idx in indices:
        item = val_data[idx]
        samples.append({
            "context": item["context"],
            "question": item["question"],
            "ground_truth": item["answers"]["text"][0] if item["answers"]["text"] else "",
        })
    
    return samples


def generate_answer(model, tokenizer, context: str, question: str, max_new_tokens: int, device: torch.device) -> str:
    """Generate an answer for a given context and question."""
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=480).to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated part
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Extract answer after "Answer:"
    if "Answer:" in generated_text:
        answer = generated_text.split("Answer:")[-1].strip()
    else:
        answer = generated_text[len(prompt):].strip()
    
    # Clean up - take first line/sentence
    answer = answer.split("\n")[0].strip()
    
    return answer


def validate_with_gemini(context: str, question: str, ground_truth: str, model_answer: str) -> dict:
    """Use Gemini to validate if the model's answer is correct."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    
    prompt = f"""You are evaluating a QA model's answer.

Context: {context[:500]}...
Question: {question}
Ground Truth Answer: {ground_truth}
Model's Answer: {model_answer}

Is the model's answer correct? Consider semantic equivalence, not exact match.
The answer is correct if it conveys the same meaning or information as the ground truth.

Return JSON with this exact format: {{"correct": true, "reasoning": "brief explanation"}} or {{"correct": false, "reasoning": "brief explanation"}}"""

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        ),
    ]
    
    generate_config = types.GenerateContentConfig(
        response_mime_type="application/json",
    )
    
    try:
        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=contents,
            config=generate_config,
        )
        
        result = json.loads(response.text)
        return {
            "correct": result.get("correct", False),
            "reasoning": result.get("reasoning", "No reasoning provided"),
        }
    except Exception as e:
        print(f"Gemini API error: {e}")
        return {"correct": False, "reasoning": f"API error: {str(e)}"}


def evaluate_config(
    model, tokenizer, samples: list, config_name: str, 
    QUANT_CONFIGS: dict, configs: dict, max_new_tokens: int, device: torch.device,
    quant_bit: int = None
) -> list:
    """Evaluate a single quantization config on all samples."""
    # Set active config
    utils.set_active_quant_config(QUANT_CONFIGS, configs[config_name])
    lora.set_active_quant_config(config_name)
    
    # Override with uniform quantization if specified
    if quant_bit is not None:
        utils.uniform_quant_config(QUANT_CONFIGS, quant_bit)
    
    results = []
    for sample in tqdm(samples, desc=f"Generating [{config_name}]", leave=False):
        model_answer = generate_answer(
            model, tokenizer, sample["context"], sample["question"], max_new_tokens, device
        )
        results.append({
            "context": sample["context"],
            "question": sample["question"],
            "ground_truth": sample["ground_truth"],
            "model_answer": model_answer,
        })
    
    return results


def validate_results(results: list, config_name: str) -> list:
    """Validate all results for a config using Gemini."""
    validated = []
    for r in tqdm(results, desc=f"Validating [{config_name}]", leave=False):
        validation = validate_with_gemini(
            r["context"], r["question"], r["ground_truth"], r["model_answer"]
        )
        validated.append({**r, **validation})
    return validated


def main():
    args = parse_args()
    
    # Check for Gemini API key
    if not os.environ.get("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY environment variable not set")
    
    # Determine which configs to evaluate
    all_precisions = list(DICT_CONFIGS.keys())
    if args.configs:
        precisions = [c for c in args.configs if c in all_precisions]
        if not precisions:
            raise ValueError(f"No valid configs found. Available: {all_precisions}")
    else:
        precisions = all_precisions
    
    print(f"Will evaluate {len(precisions)} configs: {precisions}")
    if args.quant_bit:
        print(f"Using uniform {args.quant_bit}-bit quantization override")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model, tokenizer, QUANT_CONFIGS, configs = load_model_with_lora(
        args.checkpoint, precisions, device
    )
    
    # Sample validation data
    samples = sample_validation_data(args.num_samples, args.seed)
    print(f"Sampled {len(samples)} examples for evaluation")
    
    # Evaluate each config
    all_results = {}
    accuracy_summary = {}
    
    for config_name in precisions:
        print(f"\n{'='*60}")
        print(f"Evaluating config: {config_name}")
        print(f"{'='*60}")
        
        # Generate answers
        generation_results = evaluate_config(
            model, tokenizer, samples, config_name,
            QUANT_CONFIGS, configs, args.max_new_tokens, device,
            quant_bit=args.quant_bit
        )
        
        # Validate with Gemini
        validated_results = validate_results(generation_results, config_name)
        
        # Calculate accuracy
        correct_count = sum(1 for r in validated_results if r["correct"])
        accuracy = correct_count / len(validated_results) * 100
        
        all_results[config_name] = validated_results
        accuracy_summary[config_name] = {
            "correct": correct_count,
            "total": len(validated_results),
            "accuracy": accuracy,
        }
        
        print(f"Config {config_name}: {correct_count}/{len(validated_results)} correct ({accuracy:.1f}%)")
        
        # Clean up memory between configs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Print summary
    print("\n" + "="*60)
    print("ACCURACY SUMMARY")
    print("="*60)
    for config_name, stats in accuracy_summary.items():
        print(f"{config_name}: {stats['accuracy']:.1f}% ({stats['correct']}/{stats['total']})")
    
    # Save results to JSON
    output_path = args.output or f"eval_results_{args.num_samples}samples.json"
    output_data = {
        "checkpoint": args.checkpoint,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "quant_bit_override": args.quant_bit,
        "accuracy_summary": accuracy_summary,
        "detailed_results": all_results,
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

