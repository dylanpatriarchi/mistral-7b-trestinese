"""
Main training script for fine-tuning Mistral-7B on Trestinese dialect.
Uses LoRA/QLoRA for efficient fine-tuning.
"""

import os
import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_dataset
from utils import (
    setup_logging,
    load_config,
    set_seed,
    print_model_info,
    ensure_dir,
    print_gpu_utilization
)


logger = setup_logging()


def load_and_prepare_model(config: dict):
    """
    Load the base model with quantization and prepare for LoRA training.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (model, tokenizer)
    """
    model_config = config['model']
    lora_config = config['lora']
    
    logger.info(f"Loading model: {model_config['name']}")
    
    # Configure quantization
    quantization_config = None
    if model_config.get('load_in_4bit', False):
        logger.info("Using 4-bit quantization (QLoRA)")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if model_config.get('torch_dtype') == 'bfloat16' else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif model_config.get('load_in_8bit', False):
        logger.info("Using 8-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_config['name'],
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model
    logger.info("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_config['name'],
        quantization_config=quantization_config,
        device_map=model_config.get('device_map', 'auto'),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if model_config.get('torch_dtype') == 'bfloat16' else torch.float16,
    )
    
    # Prepare model for k-bit training
    if quantization_config is not None:
        logger.info("Preparing model for k-bit training...")
        model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    logger.info("Configuring LoRA...")
    peft_config = LoraConfig(
        r=lora_config['r'],
        lora_alpha=lora_config['lora_alpha'],
        target_modules=lora_config['target_modules'],
        lora_dropout=lora_config['lora_dropout'],
        bias=lora_config['bias'],
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Apply LoRA
    logger.info("Applying LoRA adapters...")
    model = get_peft_model(model, peft_config)
    
    # Print model information
    print_model_info(model, logger)
    
    return model, tokenizer


def preprocess_dataset(examples, tokenizer, max_length=512):
    """
    Tokenize the dataset.
    
    Args:
        examples: Dataset examples
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        
    Returns:
        Tokenized examples
    """
    # Tokenize
    tokenized = tokenizer(
        examples['text'],
        truncation=True,
        max_length=max_length,
        padding='max_length',
        return_tensors=None,
    )
    
    # Set labels (for causal LM, labels are the same as input_ids)
    tokenized['labels'] = tokenized['input_ids'].copy()
    
    return tokenized


def load_datasets(data_dir: str, tokenizer, config: dict):
    """
    Load and preprocess training and validation datasets.
    
    Args:
        data_dir: Directory containing train.jsonl and validation.jsonl
        tokenizer: Tokenizer instance
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    logger.info(f"Loading datasets from {data_dir}")
    
    # Load datasets
    dataset = load_dataset(
        'json',
        data_files={
            'train': f'{data_dir}/train.jsonl',
            'validation': f'{data_dir}/validation.jsonl'
        }
    )
    
    max_length = config['dataset'].get('max_length', 512)
    
    # Tokenize datasets
    logger.info("Tokenizing datasets...")
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_dataset(examples, tokenizer, max_length),
        batched=True,
        remove_columns=dataset['train'].column_names,
        desc="Tokenizing"
    )
    
    return tokenized_dataset['train'], tokenized_dataset['validation']


def setup_training_arguments(config: dict) -> TrainingArguments:
    """
    Set up training arguments from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        TrainingArguments instance
    """
    train_config = config['training']
    wandb_config = config.get('wandb', {})
    
    # Ensure output directory exists
    ensure_dir(train_config['output_dir'])
    
    # Set up W&B if configured
    os.environ['WANDB_PROJECT'] = wandb_config.get('project', 'mistral-7b-trestinese')
    os.environ['WANDB_LOG_MODEL'] = 'false'
    
    args = TrainingArguments(
        output_dir=train_config['output_dir'],
        num_train_epochs=train_config['num_train_epochs'],
        per_device_train_batch_size=train_config['per_device_train_batch_size'],
        per_device_eval_batch_size=train_config['per_device_eval_batch_size'],
        gradient_accumulation_steps=train_config['gradient_accumulation_steps'],
        learning_rate=train_config['learning_rate'],
        weight_decay=train_config['weight_decay'],
        warmup_ratio=train_config['warmup_ratio'],
        max_grad_norm=train_config['max_grad_norm'],
        
        # Optimizer and scheduler
        optim=train_config['optim'],
        lr_scheduler_type=train_config['lr_scheduler_type'],
        
        # Logging and saving
        logging_steps=train_config['logging_steps'],
        save_steps=train_config['save_steps'],
        eval_steps=train_config['eval_steps'],
        save_total_limit=train_config['save_total_limit'],
        
        # Evaluation
        evaluation_strategy=train_config['evaluation_strategy'],
        load_best_model_at_end=train_config['load_best_model_at_end'],
        metric_for_best_model=train_config['metric_for_best_model'],
        greater_is_better=train_config['greater_is_better'],
        
        # Mixed precision
        fp16=train_config['fp16'],
        bf16=train_config['bf16'],
        
        # Other
        remove_unused_columns=train_config['remove_unused_columns'],
        report_to=train_config['report_to'],
        push_to_hub=train_config['push_to_hub'],
        
        # Additional settings for better training
        gradient_checkpointing=True,
        save_safetensors=True,
    )
    
    return args


def compute_metrics(eval_pred):
    """
    Compute metrics for evaluation.
    
    Args:
        eval_pred: Evaluation predictions
        
    Returns:
        Dictionary of metrics
    """
    predictions, labels = eval_pred
    
    # For now, we'll just return perplexity based on the loss
    # More sophisticated metrics will be computed in evaluate.py
    return {}


def train(config_path: str = "config.yaml", data_dir: str = "./data", resume_from_checkpoint: str = None):
    """
    Main training function.
    
    Args:
        config_path: Path to configuration file
        data_dir: Directory containing processed datasets
        resume_from_checkpoint: Path to checkpoint to resume from
    """
    # Load configuration
    logger.info("Loading configuration...")
    config = load_config(config_path)
    
    # Set seed for reproducibility
    set_seed(config['dataset'].get('seed', 42))
    
    # Print GPU information
    if torch.cuda.is_available():
        logger.info("\nGPU Information:")
        print_gpu_utilization()
    
    # Load model and tokenizer
    model, tokenizer = load_and_prepare_model(config)
    
    # Load datasets
    train_dataset, eval_dataset = load_datasets(data_dir, tokenizer, config)
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(eval_dataset)}")
    
    # Setup training arguments
    training_args = setup_training_arguments(config)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train
    logger.info("\n" + "=" * 60)
    logger.info("STARTING TRAINING")
    logger.info("=" * 60 + "\n")
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Save final model
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    
    final_model_path = os.path.join(config['training']['output_dir'], 'final_model')
    logger.info(f"Saving final model to {final_model_path}")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    logger.info("Training complete!")
    
    # Print final GPU utilization
    if torch.cuda.is_available():
        logger.info("\nFinal GPU Utilization:")
        print_gpu_utilization()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Mistral-7B on Trestinese dialect"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory containing processed datasets"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    
    args = parser.parse_args()
    
    train(
        config_path=args.config,
        data_dir=args.data_dir,
        resume_from_checkpoint=args.resume_from_checkpoint
    )


if __name__ == "__main__":
    main()

