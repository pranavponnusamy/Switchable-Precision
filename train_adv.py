import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.utils import QuantBlockConfig
from utils import utils
from _transformers.src.transformers.models.gpt2.modeling_gpt2 import (
    GPT2MLPQ,
    GPT2AttentionQ,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel
from utils import lora
from transformers import GPT2Model
import gc
from datasets import load_dataset
from torch.utils.data import DataLoader
import wandb

num_epochs = 100
train_batch_size = 1
val_batch_size = 1
checkpoint_dir = "adv/checkpoints"

def setup_ddp():
    """Initialize distributed training."""
    if "RANK" in os.environ:
        # Launched via torchrun
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl")
    else:
        # Single GPU fallback
        rank = 0
        local_rank = 0
        world_size = 1
    
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    
    return rank, local_rank, world_size, device


def cleanup_ddp():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    """Check if this is the main process (rank 0)."""
    return rank == 0


def print_rank0(msg, rank):
    """Print only from rank 0."""
    if is_main_process(rank):
        print(msg, flush=True)


# Initialize DDP
rank, local_rank, world_size, device = setup_ddp()
is_distributed = world_size > 1

gc.collect()
torch.cuda.empty_cache()

if (not os.path.exists(checkpoint_dir) and is_main_process(rank)):
    os.makedirs(checkpoint_dir)

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
QUANT_CONFIGS = {i: utils.QuantBlockConfig() for i in range(0, 12)}


dict_configs = {
    "default": {
        i: {
            "Attention_W_bit": 32,
            "Attention_A_bit": 32,
            "Attention_KV_bit": 32,
            "MLP_W_bit": 32,
            "MLP_A_bit": 32,
        }
        for i in range(12)
    }
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

utils.set_active_quant_config(QUANT_CONFIGS, configs["default"])
lora.set_active_quant_config("default")

print_rank0(f"Model: {model}", rank)
tokenizer.pad_token = tokenizer.eos_token


model.to(device)


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
train_data = ds["train"].select(range(20))
val_data = ds["validation"].select(range(2))


train_sampler = DistributedSampler(
    train_data, 
    num_replicas=world_size, 
    rank=rank, 
    shuffle=True
) if is_distributed else None

val_sampler = DistributedSampler(
    val_data, 
    num_replicas=world_size, 
    rank=rank, 
    shuffle=False
) if is_distributed else None

train_loader = DataLoader(
    train_data,
    batch_size=train_batch_size,  
    shuffle=(train_sampler is None), 
    sampler=train_sampler,
    collate_fn=collate_fn,
)

val_loader = DataLoader(
    val_data,
    batch_size=val_batch_size,  
    shuffle=False,
    sampler=val_sampler,
    collate_fn=collate_fn,
)

print_rank0(f"Training samples: {len(train_data)}", rank)
print_rank0(f"Validation samples: {len(val_data)}", rank)
print_rank0(f"Batches per epoch: {len(train_loader)}", rank)
print_rank0(f"World size: {world_size}", rank)
print_rank0(f"Effective batch size: {train_batch_size * world_size}", rank)


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


def validate(model, val_loader, tokenizer, device, rank, world_size):
    eval_model = model.module if hasattr(model, 'module') else model
    eval_model.eval()
    
    val_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch_data = create_masked_labels(
                tokenizer, batch["context"], batch["question"], batch["answers"]
            )
            input_ids = batch_data["input_ids"].to(device)
            attention_mask = batch_data["attention_mask"].to(device)
            labels = batch_data["labels"].to(device)
            
            outputs = eval_model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            val_loss += outputs.loss.item()
            del outputs
            
            num_batches += 1
    
    if world_size > 1:
        loss_tensor = torch.tensor([val_loss, num_batches], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        val_loss = loss_tensor[0].item() / loss_tensor[1].item()
    else:
        val_loss = val_loss / num_batches
    
    eval_model.train()
    return val_loss



loss_scale = {k: 1.0 for k in precisions}

lora_params = [p for n, p in model.named_parameters() if "lora_" in n]
for p in model.parameters():
    p.requires_grad = False
for p in lora_params:
    p.requires_grad = True

if is_distributed:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)


optimizer = torch.optim.AdamW(lora_params, lr=1e-3, weight_decay=0.01)
scheduler = CosineAnnealingLR(
    optimizer, T_max=num_epochs * len(train_loader), eta_min=1e-5
)

wandb_run = None
if is_main_process(rank):
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
            "effective_batch_size": train_loader.batch_size * world_size,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "weight_decay": optimizer.param_groups[0]["weight_decay"],
            "scheduler": "CosineAnnealingLR",
            "quant_precisions": precisions,
            "loss_scale": loss_scale,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "dataset": "rajpurkar/squad",
            "t_max": num_epochs * len(train_loader),
            "world_size": world_size,
            "distributed": is_distributed,
        },
    )
    wandb.watch(model, log="gradients", log_freq=20)

base_model = model.module if hasattr(model, 'module') else model

model.train()
global_step = 0
best_val_loss = {p: float('inf') for p in precisions}

global_step = 0
for epoch in range(num_epochs):
    model.train()
    
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    epoch_loss = 0.0
    num_batches = 0
    for batch in train_loader:
        batch_data = create_masked_labels(
            tokenizer, batch["context"], batch["question"], batch["answers"]
        )
        input_ids = batch_data["input_ids"].to(device)
        attention_mask = batch_data["attention_mask"].to(device)
        labels = batch_data["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        num_batches += 1

        if is_main_process(rank) and wandb_run is not None:
            wandb_run.log(
                {
                    "train/step_loss": loss.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                    "global_step": global_step,
                },
                step=global_step,
            )

        if global_step % 10 == 0 and is_main_process(rank):
            print_rank0(f"Epoch {epoch}, Step {global_step}, Loss {loss.item()}", rank)

        global_step += 1
    
    avg_train_loss = epoch_loss / num_batches
    if is_main_process(rank) and wandb_run is not None:
        wandb_run.log(
            {
                "train/loss": avg_train_loss,
                "epoch": epoch,
                "global_step": global_step,
            },
            step=global_step,
        )
    
    val_loss = validate(model, val_loader, tokenizer, device, rank, world_size)
    if is_main_process(rank):
        print_rank0(f"Epoch {epoch}, Val Loss {val_loss}", rank)
    if is_main_process(rank) and wandb_run is not None:
        wandb_run.log(
            {
                "val/loss": val_loss,
                "epoch": epoch,
                "global_step": global_step,
            },
            step=global_step,
        )
    
    # if is_main_process(rank):
        # torch.save(model.state_dict(), f"{checkpoint_dir}/model_epoch_{epoch}.pt")
        # wandb.save(f"{checkpoint_dir}/model_epoch_{epoch}.pt")
        # print(f"Saved checkpoint: model_epoch_{epoch}.pt", flush=True)
    if is_distributed:
        dist.barrier()

if is_main_process(rank) and wandb_run is not None:
    wandb_run.finish()
cleanup_ddp()