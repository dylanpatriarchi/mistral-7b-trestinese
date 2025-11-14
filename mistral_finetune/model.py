# mistral_finetune/model.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from configs.config import TrainingConfig, LoRAConfig

class ModelLoader:
    """
    Handles loading the model and tokenizer.
    """

    def __init__(self, model_name: str):
        """
        Initializes the ModelLoader.

        Args:
            model_name (str): The name of the model to load from Hugging Face.
        """
        self.model_name = model_name

    def load_model_and_tokenizer(self):
        """
        Loads the model and tokenizer with quantization and LoRA configuration.

        Returns:
            A tuple containing the model and tokenizer.
        """
        # Quantization configuration
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map={"": 0} # Automatically select the device
        )
        model.config.use_cache = False

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)

        # LoRA configuration
        peft_config = LoraConfig(
            r=LoRAConfig.R,
            lora_alpha=LoRAConfig.LORA_ALPHA,
            lora_dropout=LoRAConfig.LORA_DROPOUT,
            target_modules=LoRAConfig.TARGET_MODULES,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, peft_config)

        return model, tokenizer
