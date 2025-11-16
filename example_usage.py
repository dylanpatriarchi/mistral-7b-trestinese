"""
Example usage script demonstrating how to use the fine-tuned Mistral-7B Trestinese model.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from utils import create_inference_prompt, extract_output_from_response


def load_model(model_path: str, base_model: str = "mistralai/Mistral-7B-v0.1"):
    """
    Load the fine-tuned model.
    
    Args:
        model_path: Path to the fine-tuned model
        base_model: Base model name
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print("Loading LoRA adapters...")
    model = PeftModel.from_pretrained(base, model_path)
    model = model.merge_and_unload()
    model.eval()
    
    print("✅ Model loaded!\n")
    return model, tokenizer


def translate(model, tokenizer, italian_text: str) -> str:
    """
    Translate Italian text to Trestinese.
    
    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        italian_text: Italian text to translate
        
    Returns:
        Trestinese translation
    """
    prompt = create_inference_prompt(italian_text)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return extract_output_from_response(result).strip()


def main():
    """Main function with example usage."""
    # Configuration
    MODEL_PATH = "./output/final_model"  # Change this to your model path
    BASE_MODEL = "mistralai/Mistral-7B-v0.1"
    
    # Load model
    print("=" * 60)
    print("LOADING MODEL")
    print("=" * 60 + "\n")
    
    model, tokenizer = load_model(MODEL_PATH, BASE_MODEL)
    
    # Example translations
    examples = [
        "Ciao, come stai?",
        "Andiamo a bere un caffè",
        "Dove vai oggi?",
        "Non ho capito niente",
        "Che bella giornata!",
        "Mi fa male la testa",
        "Grazie mille per tutto",
        "Ci vediamo domani",
        "Hai voglia di uscire?",
        "Sono molto stanco",
    ]
    
    print("=" * 60)
    print("EXAMPLE TRANSLATIONS")
    print("=" * 60 + "\n")
    
    for italian in examples:
        trestinese = translate(model, tokenizer, italian)
        print(f"🇮🇹 {italian}")
        print(f"🗣️  {trestinese}\n")
    
    print("=" * 60)
    print("CUSTOM TRANSLATION")
    print("=" * 60 + "\n")
    
    # Try your own translation
    custom_text = "Buongiorno! Come posso aiutarti oggi?"
    custom_translation = translate(model, tokenizer, custom_text)
    
    print(f"🇮🇹 {custom_text}")
    print(f"🗣️  {custom_translation}")
    
    print("\n" + "=" * 60)
    print("✅ Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()

