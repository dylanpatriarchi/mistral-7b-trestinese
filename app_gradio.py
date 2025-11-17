"""
Gradio web interface for Mistral-7B Trestinese translator.
Can be run on Google Colab to create a public web interface.
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from utils import create_inference_prompt, extract_output_from_response


class TrestineseTranslatorApp:
    """Web application for Italian to Trestinese translation."""
    
    def __init__(self, model_path: str, base_model: str = "mistralai/Mistral-7B-v0.1"):
        """
        Initialize the translator app.
        
        Args:
            model_path: Path to fine-tuned model
            base_model: Base model name
        """
        print("Loading model...")
        self.load_model(model_path, base_model)
        print("✅ Model loaded!")
    
    def load_model(self, model_path: str, base_model: str):
        """Load the fine-tuned model with 4-bit quantization for Colab."""
        from transformers import BitsAndBytesConfig
        import os
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Check if model is already merged (no adapter_config.json)
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        is_merged = not os.path.exists(adapter_config_path)
        
        if is_merged:
            # Model is already merged, load directly
            print("Loading merged model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        else:
            # Model has adapters, use 4-bit quantization to save memory
            print("Loading model with 4-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # Load base model with 4-bit quantization
            base = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=quantization_config,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
            # Load adapters (no merge needed with quantized model)
            self.model = PeftModel.from_pretrained(base, model_path)
        
        self.model.eval()
    
    def translate(
        self,
        italian_text: str,
        temperature: float = 0.7,
        max_tokens: int = 256,
        top_p: float = 0.9
    ) -> str:
        """
        Translate Italian text to Trestinese.
        
        Args:
            italian_text: Italian text to translate
            temperature: Sampling temperature (higher = more creative)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            
        Returns:
            Trestinese translation
        """
        if not italian_text.strip():
            return "⚠️ Inserisci del testo italiano da tradurre!"
        
        # Create prompt
        prompt = create_inference_prompt(italian_text)
        
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
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=50,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Decode
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        translation = extract_output_from_response(result)
        
        return translation.strip()


def create_interface(model_path: str) -> gr.Interface:
    """
    Create Gradio interface.
    
    Args:
        model_path: Path to fine-tuned model
        
    Returns:
        Gradio Interface
    """
    # Initialize translator
    app = TrestineseTranslatorApp(model_path)
    
    # Example phrases
    examples = [
        ["Ciao, come stai?"],
        ["Andiamo a bere un caffè"],
        ["Dove vai oggi?"],
        ["Non ho capito niente"],
        ["Che bella giornata!"],
        ["Mi fa male la testa"],
        ["Grazie mille per tutto"],
        ["Ci vediamo domani"],
        ["Hai voglia di uscire?"],
        ["Sono molto stanco"]
    ]
    
    # Create interface
    interface = gr.Interface(
        fn=app.translate,
        inputs=[
            gr.Textbox(
                label="🇮🇹 Italiano",
                placeholder="Inserisci una frase in italiano...",
                lines=3
            ),
            gr.Slider(
                minimum=0.1,
                maximum=1.5,
                value=0.7,
                step=0.1,
                label="Temperature (creatività)"
            ),
            gr.Slider(
                minimum=50,
                maximum=512,
                value=256,
                step=50,
                label="Max Tokens"
            ),
            gr.Slider(
                minimum=0.5,
                maximum=1.0,
                value=0.9,
                step=0.05,
                label="Top P"
            )
        ],
        outputs=gr.Textbox(
            label="🗣️ Trestinese",
            lines=3
        ),
        title="🗣️ Traduttore Italiano → Trestinese",
        description="""
        Traduttore basato su **Mistral-7B** fine-tuned per il dialetto trestinese.
        
        Inserisci una frase in italiano e ottieni la traduzione in trestinese!
        """,
        examples=examples,
        theme=gr.themes.Soft(),
        analytics_enabled=False
    )
    
    return interface


def main():
    """Main function to launch the app."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch Gradio web interface")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./output/final_model",
        help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public link (for Colab)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run on"
    )
    
    args = parser.parse_args()
    
    # Create and launch interface
    interface = create_interface(args.model_path)
    
    print("\n" + "="*60)
    print("🚀 LAUNCHING WEB INTERFACE")
    print("="*60)
    
    interface.launch(
        share=args.share,
        server_port=args.port,
        server_name="0.0.0.0"
    )


if __name__ == "__main__":
    main()

