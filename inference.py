"""
Inference script for Mistral-7B Trestinese model.
Allows interactive translation from Italian to Trestinese.
"""

import os
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from utils import (
    setup_logging,
    load_config,
    create_inference_prompt,
    extract_output_from_response
)


logger = setup_logging()


class TrestineseTranslator:
    """Interactive translator for Italian to Trestinese."""
    
    def __init__(self, model, tokenizer, config: dict):
        """
        Initialize translator.
        
        Args:
            model: Fine-tuned model
            tokenizer: Tokenizer instance
            config: Configuration dictionary
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.generation_config = config.get('generation', {})
        
        logger.info("Translator ready!")
    
    def translate(self, italian_text: str, verbose: bool = False) -> str:
        """
        Translate Italian text to Trestinese.
        
        Args:
            italian_text: Italian text to translate
            verbose: Whether to print generation details
            
        Returns:
            Trestinese translation
        """
        # Create prompt
        prompt = create_inference_prompt(italian_text)
        
        if verbose:
            logger.info(f"Prompt:\n{prompt}")
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.model.device)
        
        # Generate
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
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if verbose:
            logger.info(f"Full output:\n{generated_text}")
        
        # Extract translation
        translation = extract_output_from_response(generated_text)
        
        return translation.strip()
    
    def translate_batch(self, italian_texts: list) -> list:
        """
        Translate multiple Italian texts to Trestinese.
        
        Args:
            italian_texts: List of Italian texts
            
        Returns:
            List of Trestinese translations
        """
        translations = []
        for text in italian_texts:
            translation = self.translate(text)
            translations.append(translation)
        return translations
    
    def interactive_mode(self):
        """Run interactive translation mode."""
        logger.info("\n" + "=" * 60)
        logger.info("INTERACTIVE TRANSLATION MODE".center(60))
        logger.info("=" * 60)
        logger.info("Enter Italian text to translate to Trestinese.")
        logger.info("Type 'quit', 'exit', or 'q' to stop.")
        logger.info("=" * 60 + "\n")
        
        while True:
            try:
                # Get input
                italian_text = input("\n🇮🇹 Italian: ").strip()
                
                # Check for exit commands
                if italian_text.lower() in ['quit', 'exit', 'q', '']:
                    logger.info("Goodbye!")
                    break
                
                # Translate
                translation = self.translate(italian_text)
                
                # Print result
                print(f"🗣️  Trestinese: {translation}")
                
            except KeyboardInterrupt:
                logger.info("\n\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error: {e}")


def load_model_for_inference(model_path: str, base_model_name: str = None):
    """
    Load fine-tuned model for inference.
    
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
    logger.info("Loading tokenizer...")
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
            device_map="auto",
            trust_remote_code=True
        )
        
        logger.info("Loading LoRA adapters...")
        model = PeftModel.from_pretrained(base_model, model_path)
        
        # Option to merge adapters for faster inference
        logger.info("Merging adapters for faster inference...")
        model = model.merge_and_unload()
    else:
        # Load full model
        logger.info("Loading full model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    
    model.eval()
    logger.info("Model loaded successfully!")
    
    return model, tokenizer


def translate_from_file(translator: TrestineseTranslator, input_file: str, output_file: str):
    """
    Translate texts from a file.
    
    Args:
        translator: TrestineseTranslator instance
        input_file: Input file with Italian texts (one per line or JSONL)
        output_file: Output file for translations
    """
    logger.info(f"Reading from {input_file}")
    
    # Read input
    italian_texts = []
    if input_file.endswith('.jsonl'):
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                italian_texts.append(data.get('italiano', data.get('text', '')))
    else:
        with open(input_file, 'r', encoding='utf-8') as f:
            italian_texts = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Found {len(italian_texts)} texts to translate")
    
    # Translate
    logger.info("Translating...")
    translations = translator.translate_batch(italian_texts)
    
    # Save results
    logger.info(f"Saving results to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for italian, trestinese in zip(italian_texts, translations):
            result = {
                'italiano': italian,
                'trestinese': trestinese
            }
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    logger.info("Translation complete!")


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description="Translate Italian to Trestinese using fine-tuned Mistral-7B"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model"
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
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Italian text to translate"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Input file with Italian texts"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="translations.jsonl",
        help="Output file for translations"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print generation details"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Load model
    model, tokenizer = load_model_for_inference(args.model_path, args.base_model)
    
    # Create translator
    translator = TrestineseTranslator(model, tokenizer, config)
    
    # Run appropriate mode
    if args.interactive:
        translator.interactive_mode()
    elif args.input_file:
        translate_from_file(translator, args.input_file, args.output_file)
    elif args.text:
        translation = translator.translate(args.text, verbose=args.verbose)
        print(f"\nItalian:    {args.text}")
        print(f"Trestinese: {translation}\n")
    else:
        logger.error("Please provide --text, --input_file, or use --interactive mode")
        parser.print_help()


if __name__ == "__main__":
    main()

