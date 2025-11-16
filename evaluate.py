"""
Evaluation script for Mistral-7B Trestinese model.
Computes various metrics including BLEU, ROUGE, and perplexity.
"""

import os
import json
import argparse
import torch
from tqdm import tqdm
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from evaluate import load
from utils import (
    setup_logging,
    load_config,
    create_inference_prompt,
    extract_output_from_response,
    set_seed
)


logger = setup_logging()


class TrestineseEvaluator:
    """Evaluator for Trestinese translation model."""
    
    def __init__(self, model, tokenizer, config: dict):
        """
        Initialize evaluator.
        
        Args:
            model: Fine-tuned model
            tokenizer: Tokenizer instance
            config: Configuration dictionary
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.generation_config = config.get('generation', {})
        
        # Load metrics
        logger.info("Loading evaluation metrics...")
        self.bleu_metric = load("bleu")
        self.rouge_metric = load("rouge")
        
    def generate_translation(self, italian_text: str) -> str:
        """
        Generate Trestinese translation for Italian text.
        
        Args:
            italian_text: Italian text to translate
            
        Returns:
            Generated Trestinese translation
        """
        prompt = create_inference_prompt(italian_text)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.generation_config.get('max_new_tokens', 256),
                temperature=self.generation_config.get('temperature', 0.7),
                top_p=self.generation_config.get('top_p', 0.9),
                top_k=self.generation_config.get('top_k', 50),
                repetition_penalty=self.generation_config.get('repetition_penalty', 1.1),
                do_sample=self.generation_config.get('do_sample', True),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        translation = extract_output_from_response(generated_text)
        
        return translation.strip()
    
    def evaluate_dataset(self, dataset_path: str) -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            dataset_path: Path to JSONL dataset file
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating on {dataset_path}")
        
        # Load dataset
        dataset = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                dataset.append(json.loads(line.strip()))
        
        logger.info(f"Loaded {len(dataset)} examples")
        
        predictions = []
        references = []
        
        # Generate translations
        logger.info("Generating translations...")
        for example in tqdm(dataset, desc="Evaluating"):
            italian = example.get('italiano', example.get('input', ''))
            reference = example.get('trestinese', example.get('output', ''))
            
            prediction = self.generate_translation(italian)
            
            predictions.append(prediction)
            references.append(reference)
        
        # Compute metrics
        logger.info("Computing metrics...")
        metrics = self.compute_metrics(predictions, references)
        
        return metrics, predictions, references
    
    def compute_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            predictions: List of predicted translations
            references: List of reference translations
            
        Returns:
            Dictionary of metrics
        """
        # BLEU score
        bleu_results = self.bleu_metric.compute(
            predictions=predictions,
            references=[[ref] for ref in references]
        )
        
        # ROUGE scores
        rouge_results = self.rouge_metric.compute(
            predictions=predictions,
            references=references
        )
        
        # Character Error Rate (CER)
        cer = self.compute_cer(predictions, references)
        
        # Word accuracy
        word_accuracy = self.compute_word_accuracy(predictions, references)
        
        metrics = {
            'bleu': bleu_results['bleu'],
            'rouge1': rouge_results['rouge1'],
            'rouge2': rouge_results['rouge2'],
            'rougeL': rouge_results['rougeL'],
            'cer': cer,
            'word_accuracy': word_accuracy,
        }
        
        return metrics
    
    @staticmethod
    def compute_cer(predictions: List[str], references: List[str]) -> float:
        """
        Compute Character Error Rate.
        
        Args:
            predictions: List of predicted translations
            references: List of reference translations
            
        Returns:
            CER score
        """
        total_chars = 0
        total_errors = 0
        
        for pred, ref in zip(predictions, references):
            # Simple character-level distance (Levenshtein)
            errors = sum(1 for a, b in zip(pred, ref) if a != b)
            errors += abs(len(pred) - len(ref))
            
            total_errors += errors
            total_chars += len(ref)
        
        return total_errors / total_chars if total_chars > 0 else 0
    
    @staticmethod
    def compute_word_accuracy(predictions: List[str], references: List[str]) -> float:
        """
        Compute word-level accuracy.
        
        Args:
            predictions: List of predicted translations
            references: List of reference translations
            
        Returns:
            Word accuracy score
        """
        total_words = 0
        correct_words = 0
        
        for pred, ref in zip(predictions, references):
            pred_words = pred.lower().split()
            ref_words = ref.lower().split()
            
            # Count matching words (in order)
            matches = sum(1 for p, r in zip(pred_words, ref_words) if p == r)
            
            correct_words += matches
            total_words += len(ref_words)
        
        return correct_words / total_words if total_words > 0 else 0


def load_model_for_evaluation(model_path: str, base_model_name: str = None):
    """
    Load fine-tuned model for evaluation.
    
    Args:
        model_path: Path to fine-tuned model
        base_model_name: Base model name (if loading LoRA adapters)
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model from {model_path}")
    
    # Check if this is a PEFT model or full model
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    is_peft_model = os.path.exists(adapter_config_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    if is_peft_model:
        # Load base model and adapters
        if base_model_name is None:
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
                base_model_name = adapter_config.get('base_model_name_or_path')
        
        logger.info(f"Loading base model: {base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        logger.info("Loading LoRA adapters...")
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()  # Merge adapters for faster inference
    else:
        # Load full model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    
    model.eval()
    
    return model, tokenizer


def print_metrics(metrics: Dict[str, float], title: str = "EVALUATION METRICS"):
    """
    Print metrics in a formatted table.
    
    Args:
        metrics: Dictionary of metrics
        title: Title for the metrics table
    """
    logger.info("\n" + "=" * 60)
    logger.info(title.center(60))
    logger.info("=" * 60)
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name.upper():20s}: {value:.4f}")
    logger.info("=" * 60 + "\n")


def save_results(
    metrics: Dict[str, float],
    predictions: List[str],
    references: List[str],
    output_dir: str
):
    """
    Save evaluation results to files.
    
    Args:
        metrics: Dictionary of metrics
        predictions: List of predictions
        references: List of references
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")
    
    # Save predictions
    results_path = os.path.join(output_dir, "predictions.jsonl")
    with open(results_path, 'w', encoding='utf-8') as f:
        for pred, ref in zip(predictions, references):
            f.write(json.dumps({
                'prediction': pred,
                'reference': ref
            }, ensure_ascii=False) + '\n')
    logger.info(f"Saved predictions to {results_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate Mistral-7B Trestinese model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data/validation.jsonl",
        help="Path to evaluation dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model name (for loading LoRA adapters)"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    set_seed(config['dataset'].get('seed', 42))
    
    # Load model
    model, tokenizer = load_model_for_evaluation(args.model_path, args.base_model)
    
    # Create evaluator
    evaluator = TrestineseEvaluator(model, tokenizer, config)
    
    # Evaluate
    metrics, predictions, references = evaluator.evaluate_dataset(args.dataset_path)
    
    # Print metrics
    print_metrics(metrics)
    
    # Save results
    save_results(metrics, predictions, references, args.output_dir)
    
    # Print some example predictions
    logger.info("Sample predictions:")
    logger.info("-" * 60)
    for i in range(min(5, len(predictions))):
        logger.info(f"\nExample {i+1}:")
        logger.info(f"Reference:  {references[i]}")
        logger.info(f"Prediction: {predictions[i]}")
    logger.info("-" * 60)
    
    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()

