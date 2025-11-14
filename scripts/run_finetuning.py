# scripts/run_finetuning.py

import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mistral_finetune.data_loader import DatasetLoader
from mistral_finetune.model import ModelLoader
from mistral_finetune.trainer import FineTuner
from configs.config import TrainingConfig

def main():
    """
    The main function to run the fine-tuning process.
    """
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model_loader = ModelLoader(model_name=TrainingConfig.MODEL_NAME)
    model, tokenizer = model_loader.load_model_and_tokenizer()

    # Load and prepare dataset
    print("Loading and preparing dataset...")
    dataset_loader = DatasetLoader(dataset_path=TrainingConfig.DATASET_PATH, tokenizer=tokenizer)
    train_dataset = dataset_loader.load_and_prepare_dataset()

    # Fine-tune the model
    print("Starting fine-tuning...")
    fine_tuner = FineTuner(model=model, tokenizer=tokenizer, train_dataset=train_dataset)
    fine_tuner.train()

    print("Fine-tuning finished.")

if __name__ == "__main__":
    main()
