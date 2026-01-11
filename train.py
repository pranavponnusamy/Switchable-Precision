from utils.utils import QuantBlockConfig
from utils import utils
from _transformers.src.transformers.models.gpt2.modeling_gpt2 import (
    GPT2MLPQ,
    GPT2AttentionQ,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel
from utils import lora
from transformers import GPT2Model
import torch
from utils import lora
import torch
import gc
from datasets import load_dataset
from torch.utils.data import DataLoader
import wandb


gc.collect()
torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
QUANT_CONFIGS = {i: utils.QuantBlockConfig() for i in range(0, 12)}

# otherwise 4-8-4
dict_configs = {
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
}
precisions = [k for k in dict_configs.keys()]

configs = {}
for k, v in dict_configs.items():
    conf = [QuantBlockConfig.from_dict(dict_configs[k][i]) for i in range(12)]
    quant_configs = {i: conf[i] for i in range(12)}
    configs[k] = quant_configs

LORA_R = 32
LORA_ALPHA = 64.0

utils.quantize_model(model, QUANT_CONFIGS)
lora.apply_lora_to_model(
    model, precisions, r=LORA_R, alpha=LORA_ALPHA, lora_attention=True, lora_mlp=True
)
lora.load_lora(model, "lora_epoch_9.pt")

# lora.load_lora(model, "lora_final.pt")
utils.set_active_quant_config(QUANT_CONFIGS, configs["8-8-4_uniform"])
lora.set_active_quant_config("8-8-4_uniform")

print(model, flush=True)
tokenizer.pad_token = tokenizer.eos_token
model_inputs = tokenizer(
    ["Context: Bob killed rob. Rob killed charlie. Charlie killed linda. Question: Who killed Rob? Answer: "], return_tensors="pt", padding=True
).to(model.device)
print(model_inputs.input_ids.shape, flush=True)

with torch.no_grad():
    generated_ids = model.generate(
        input_ids=model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_new_tokens=30,
        do_sample=False, 
        pad_token_id=tokenizer.eos_token_id,
    )


print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0], flush=True)

utils.set_active_quant_config(QUANT_CONFIGS, configs["8-8-16_uniform"])
lora.set_active_quant_config("8-8-16_uniform")

model.to(device)

with torch.no_grad():
    generated_ids = model.generate(
        input_ids=model_inputs.input_ids.to(device),
        attention_mask=model_inputs.attention_mask.to(device),
        max_new_tokens=30,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )


print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0], flush=True)

def collate_fn(batch):
    """Custom collate for SQuAD dataset"""
    return {
        "context": [item["context"] for item in batch],
        "question": [item["question"] for item in batch],
        "answers": [
            item["answers"]["text"][0] if item["answers"]["text"] else ""
            for item in batch
        ],
    }

ds = load_dataset("rajpurkar/squad")
train_data = ds["train"]
val_data = ds["validation"]

train_loader = DataLoader(
    train_data,
    batch_size=64, 
    shuffle=True,
    collate_fn=collate_fn,
)

val_loader = DataLoader(
    val_data,
    batch_size=128,  
    shuffle=False,
    collate_fn=collate_fn,
)

print(f"Training samples: {len(train_data)}", flush=True)
print(f"Validation samples: {len(val_data)}", flush=True)
print(f"Batches per epoch: {len(train_loader)}", flush=True)


def create_masked_labels(tokenizer, contexts, questions, answers, max_length=512):
    """
    Create input_ids and labels where only answer tokens contribute to loss.
    Labels use -100 for masked positions (ignored by CrossEntropyLoss).

    IMPORTANT: We truncate the CONTEXT (left side), not the answer, to ensure
    the model always sees and learns from the answer tokens.
    """
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for context, question, answer in zip(contexts, questions, answers):
        # First, tokenize the question and answer parts (these must NOT be truncated)
        qa_part = f"\nQuestion: {question}\nAnswer: {answer}{tokenizer.eos_token}"
        qa_tokens = tokenizer(qa_part, add_special_tokens=False)
        qa_length = len(qa_tokens.input_ids)

        # Calculate how much space is left for context
        # Reserve tokens for "Context: " prefix
        context_prefix = "Context: "
        prefix_tokens = tokenizer(context_prefix, add_special_tokens=True)
        prefix_length = len(prefix_tokens.input_ids)

        # Max tokens available for context content
        max_context_tokens = max_length - qa_length - prefix_length

        if max_context_tokens < 10:
            # If there's not enough room, use minimal context
            max_context_tokens = 10

        # Tokenize and truncate context from the RIGHT (keep beginning)
        context_tokens = tokenizer(
            context,
            add_special_tokens=False,
            max_length=max_context_tokens,
            truncation=True,
        )
        truncated_context = tokenizer.decode(
            context_tokens.input_ids, skip_special_tokens=True
        )

        # Build the full prompt with truncated context
        prompt = f"Context: {truncated_context}\nQuestion: {question}\nAnswer:"
        full_text = f"{prompt} {answer}{tokenizer.eos_token}"

        # Tokenize prompt separately to get its length (for masking)
        prompt_tokens = tokenizer(prompt, add_special_tokens=True)
        prompt_length = len(prompt_tokens.input_ids)

        # Tokenize full sequence (should fit within max_length now)
        full_tokens = tokenizer(
            full_text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = full_tokens.input_ids.squeeze(0)
        attention_mask = full_tokens.attention_mask.squeeze(0)

        # Create labels: -100 for prompt, actual token ids for answer
        labels = input_ids.clone()
        labels[:prompt_length] = -100  # Mask context + question

        # Also mask padding tokens
        labels[attention_mask == 0] = -100

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)

    return {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "labels": torch.stack(labels_list),
    }


num_epochs = 15  # Increased for center_reduced configs to converge


def validate(model, val_loader, tokenizer, configs, precisions, device):
    """Run validation and return average loss per config."""
    model.eval()
    val_losses = {p: 0.0 for p in precisions}
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch_data = create_masked_labels(
                tokenizer, batch["context"], batch["question"], batch["answers"]
            )
            input_ids = batch_data["input_ids"].to(device)
            attention_mask = batch_data["attention_mask"].to(device)
            labels = batch_data["labels"].to(device)
            
            for precision in precisions:
                utils.set_active_quant_config(QUANT_CONFIGS, configs[precision])
                lora.set_active_quant_config(precision)
                
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                val_losses[precision] += outputs.loss.item()
                del outputs
            
            num_batches += 1
    
    # Average losses
    val_losses = {p: v / num_batches for p, v in val_losses.items()}
    model.train()
    return val_losses

# Optional: weight each config's contribution to the loss
loss_scale = {k: 1.0 for k in precisions}

# Get ALL LoRA parameters (for all configs)
lora_params = [p for n, p in model.named_parameters() if "lora_" in n]
for p in model.parameters():
    p.requires_grad = False
for p in lora_params:
    p.requires_grad = True

# Higher learning rate with scaling=1.0, and add warmup scheduler for stability
optimizer = torch.optim.AdamW(lora_params, lr=1e-3, weight_decay=0.01)

# Optional: Learning rate scheduler for better convergence
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(
    optimizer, T_max=num_epochs * len(train_loader), eta_min=1e-5
)

wandb_run = wandb.init(
    entity="gatech_pranav",
    project="eic-test",
    config={
        "model_name": "openai-community/gpt2",
        "num_epochs": num_epochs,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "train_batch_size": train_loader.batch_size,
        "val_batch_size": val_loader.batch_size,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "weight_decay": optimizer.param_groups[0]["weight_decay"],
        "scheduler": "CosineAnnealingLR",
        "quant_precisions": precisions,
        "loss_scale": loss_scale,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "dataset": "rajpurkar/squad",
        "t_max": num_epochs * len(train_loader),
    },
)
wandb.watch(model, log="gradients", log_freq=10)

model.train()
global_step = 0
best_val_loss = {p: float('inf') for p in precisions}

for epoch in range(num_epochs):
    print(f"\n{'='*60}", flush=True)
    print(f"Epoch {epoch + 1}/{num_epochs}", flush=True)
    print(f"{'='*60}", flush=True)
    
    epoch_losses = {p: 0.0 for p in precisions}
    num_batches = 0
    
    for batch in train_loader:
        # Prepare inputs (same for all configs)
        batch_data = create_masked_labels(
            tokenizer, batch["context"], batch["question"], batch["answers"]
        )
        input_ids = batch_data["input_ids"].to(model.device)
        attention_mask = batch_data["attention_mask"].to(model.device)
        labels = batch_data["labels"].to(model.device)

        optimizer.zero_grad()  # Zero grads once at the start

        loss_values = {}

        # Forward + backward for EACH precision config
        for precision in precisions:
            utils.set_active_quant_config(QUANT_CONFIGS, configs[precision])
            lora.set_active_quant_config(precision)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            # Scale the loss for this config
            loss = outputs.loss * loss_scale[precision]

            # Accumulate gradients (don't step yet!)
            loss.backward()

            loss_values[precision] = outputs.loss.item()
            epoch_losses[precision] += outputs.loss.item()

            # Free memory
            del outputs, loss

        num_batches += 1

        # Gradient clipping - important for stability with quantization
        # torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)

        # Single optimizer step after all configs
        optimizer.step()
        scheduler.step()

        global_step += 1
        wandb_metrics = {f"train/batch_loss/{p}": v for p, v in loss_values.items()}
        wandb_metrics.update(
            {
                "lr": scheduler.get_last_lr()[0],
                "epoch": epoch + 1,
                "global_step": global_step,
            }
        )
        wandb_run.log(wandb_metrics, step=global_step)

        # Logging every 100 batches
        if num_batches % 100 == 0:
            loss_str = " | ".join([f"{p}: {v:.4f}" for p, v in loss_values.items()])
            print(f"[Batch {num_batches}/{len(train_loader)}] Losses: {loss_str}", flush=True)

    # Epoch summary - average training loss
    avg_train_losses = {p: v / num_batches for p, v in epoch_losses.items()}
    train_str = " | ".join([f"{p}: {v:.4f}" for p, v in avg_train_losses.items()])
    print(f"\n[Epoch {epoch + 1}] Avg Train Loss: {train_str}", flush=True)
    wandb_run.log(
        {
            **{f"train/epoch_loss/{p}": v for p, v in avg_train_losses.items()},
            "epoch": epoch + 1,
            "global_step": global_step,
        },
        step=global_step,
    )
    
    # Validation
    print(f"Running validation...", flush=True)
    val_losses = validate(model, val_loader, tokenizer, configs, precisions, device)
    val_str = " | ".join([f"{p}: {v:.4f}" for p, v in val_losses.items()])
    print(f"[Epoch {epoch + 1}] Val Loss: {val_str}", flush=True)
    wandb_run.log(
        {
            **{f"val/loss/{p}": v for p, v in val_losses.items()},
            "epoch": epoch + 1,
            "global_step": global_step,
        },
        step=global_step,
    )
    
    # Track best validation loss
    improved = []
    for p in precisions:
        if val_losses[p] < best_val_loss[p]:
            best_val_loss[p] = val_losses[p]
            improved.append(p)
    if improved:
        print(f"[Epoch {epoch + 1}] Improved: {', '.join(improved)}", flush=True)
        wandb_run.log(
            {f"val/best/{p}": best_val_loss[p] for p in improved},
            step=global_step,
        )
    
    # Save checkpoint after each epoch
    lora.save_lora(model, f"lora_epoch_{epoch + 1}.pt", precisions)
    print(f"Saved checkpoint: lora_epoch_{epoch + 1}.pt", flush=True)

# Save final model
lora.save_lora(model, "lora_final.pt", precisions)
print(f"\nTraining complete! Final model saved to lora_final.pt", flush=True)
print(f"Best validation losses: {best_val_loss}", flush=True)
wandb_run.summary["best_val_loss"] = best_val_loss
wandb_run.finish()
