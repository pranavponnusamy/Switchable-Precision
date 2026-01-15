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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fully fine-tuned GPT-2 using Gemini validation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint file")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to evaluate")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--max_new_tokens", type=int, default=30, help="Max tokens to generate")
    parser.add_argument("--quant_bit", type=int, default=8, help="Quantization bit width (default: 8)")
    return parser.parse_args()


def load_model(checkpoint_path: str, quant_bit: int, device: torch.device):
    """Load GPT-2, apply quantization, and load checkpoint weights."""
    print("Loading GPT-2 base model...")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create quantization configs for each layer
    QUANT_CONFIGS = {i: QuantBlockConfig() for i in range(12)}
    
    print("Applying quantization...")
    utils.quantize_model(model, QUANT_CONFIGS)
    utils.uniform_quant_config(QUANT_CONFIGS, quant_bit)
    print(f"Using {quant_bit}-bit quantization")
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # Handle DDP-wrapped checkpoints (keys start with 'module.')
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)
    
    model.to(device)
    model.eval()
    
    return model, tokenizer


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


def main():
    args = parse_args()
    
    # Check for Gemini API key
    if not os.environ.get("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY environment variable not set")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model, tokenizer = load_model(args.checkpoint, args.quant_bit, device)
    
    # Sample validation data
    samples = sample_validation_data(args.num_samples, args.seed)
    print(f"Sampled {len(samples)} examples for evaluation")
    
    # Generate answers
    print("\nGenerating answers...")
    results = []
    for sample in tqdm(samples, desc="Generating"):
        model_answer = generate_answer(
            model, tokenizer, sample["context"], sample["question"], 
            args.max_new_tokens, device
        )
        results.append({
            "context": sample["context"],
            "question": sample["question"],
            "ground_truth": sample["ground_truth"],
            "model_answer": model_answer,
        })
    
    # Validate with Gemini
    print("\nValidating with Gemini...")
    validated_results = []
    for r in tqdm(results, desc="Validating"):
        validation = validate_with_gemini(
            r["context"], r["question"], r["ground_truth"], r["model_answer"]
        )
        validated_results.append({**r, **validation})
    
    # Calculate accuracy
    correct_count = sum(1 for r in validated_results if r["correct"])
    accuracy = correct_count / len(validated_results) * 100
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Accuracy: {accuracy:.1f}% ({correct_count}/{len(validated_results)})")
    
    # Save results to JSON
    output_path = args.output or f"eval_full_results_{args.num_samples}samples.json"
    output_data = {
        "checkpoint": args.checkpoint,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "quant_bit": args.quant_bit,
        "accuracy": accuracy,
        "correct": correct_count,
        "total": len(validated_results),
        "detailed_results": validated_results,
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

