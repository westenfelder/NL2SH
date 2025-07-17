import torch
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


# wandb logging
run = wandb.init(
    project="nl2sh",
    name="run01",
)


# load model
model_id = "Qwen/Qwen2.5-Coder-3B-Instruct"
# clean_up_tokenization_spaces to prevent stripping of whitespace
tokenizer = AutoTokenizer.from_pretrained(model_id, clean_up_tokenization_spaces=False)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", torch_dtype=torch.bfloat16)

# uncomment for llama models
# tokenizer.pad_token = '<|finetune_right_pad_id|>' # NOT tokenizer.pad_token = tokenizer.eos_token 

# load dataset
def apply_chat_template(row):
    messages = [
        {"role": "system", "content": "Your task is to translate a natural language instruction to a Bash command. You will receive an instruction in English and output a Bash command that can be run in a Linux terminal."},
        {"role": "user", "content": row['nl']},
        {"role": "assistant", "content": row['bash']}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=False,
    )
    return {"prompt": prompt}

def tokenize_rows(row):
    # truncation to 150 tokens prunes 676 items in the dataset
    tokens = tokenizer(row['prompt'], padding="max_length", truncation=True, max_length=150)
    # padding tokens should not be considered in the loss calculation
    tokens['labels'] = [-100 if token == tokenizer.pad_token_id else token for token in tokens['input_ids']]
    return tokens

train_dataset = load_dataset("westenfelder/NL2SH-ALFA", "train", split="train")
test_dataset = load_dataset("westenfelder/NL2SH-ALFA", "test", split="train")
formatted_train_dataset = train_dataset.map(apply_chat_template)
formatted_test_dataset = test_dataset.map(apply_chat_template)
tokenized_train_dataset = formatted_train_dataset.map(tokenize_rows)
tokenized_test_dataset = formatted_test_dataset.map(tokenize_rows)
final_train_dataset = tokenized_train_dataset.remove_columns(['nl', 'bash', 'prompt'])
final_test_dataset = tokenized_test_dataset.remove_columns(['nl', 'bash', 'prompt'])


# train
model.train()
training_args = TrainingArguments(
    output_dir="checkpoints",
    eval_strategy="steps",
    eval_steps=1000,
    logging_steps=100,
    save_steps = 5000,
    # max_steps=5,
    per_device_train_batch_size=15,
    per_device_eval_batch_size=15,
    gradient_accumulation_steps=5,
    num_train_epochs=10,
    report_to="wandb",
    log_level="info",
    learning_rate=1e-5,
    max_grad_norm=2,
    weight_decay = 0.01,
    seed = 123,
    bf16 = True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=final_train_dataset,
    eval_dataset=final_test_dataset,
    tokenizer=tokenizer
)

trainer.train(resume_from_checkpoint = False)
wandb.finish()
trainer.save_model("final")
tokenizer.save_pretrained("final")

model.push_to_hub("westenfelder/NL2SH")
tokenizer.push_to_hub("westenfelder/NL2SH")

# clear memory
del model
del tokenizer
del trainer
torch.cuda.empty_cache()
