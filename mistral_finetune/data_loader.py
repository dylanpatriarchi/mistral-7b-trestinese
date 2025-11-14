# mistral_finetune/data_loader.py

from datasets import load_dataset
from configs.config import TrainingConfig, PromptConfig

class DatasetLoader:
    """
    Handles loading and preparing the dataset for fine-tuning.
    """

    def __init__(self, dataset_path: str, tokenizer):
        """
        Initializes the DatasetLoader.

        Args:
            dataset_path (str): The path to the dataset file.
            tokenizer: The tokenizer to use for preprocessing.
        """
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer

    def load_and_prepare_dataset(self):
        """
        Loads the dataset and prepares it for training.

        Returns:
            The preprocessed dataset.
        """
        dataset = load_dataset("json", data_files=self.dataset_path, split="train")
        dataset = dataset.map(self._generate_and_tokenize_prompt)
        return dataset

    def _generate_and_tokenize_prompt(self, data_point):
        """
        Generates a prompt from a data point and tokenizes it.

        Args:
            data_point (dict): A dictionary containing 'italian' and 'trestinese' keys.

        Returns:
            The tokenized prompt.
        """
        full_prompt = PromptConfig.PROMPT_TEMPLATE.format(
            italian=data_point["italiano"],
            trestinese=data_point["trestinese"],
        )
        
        # Set pad_token if it's not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        tokenized_full_prompt = self.tokenizer(
            full_prompt,
            truncation=True,
            max_length=TrainingConfig.MAX_SEQ_LENGTH,
            padding="max_length",
        )
        return tokenized_full_prompt
