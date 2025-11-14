# mistral_finetune/trainer.py

import transformers
from configs.config import TrainingConfig

class FineTuner:
    """
    Handles the fine-tuning process of the model.
    """

    def __init__(self, model, tokenizer, train_dataset):
        """
        Initializes the FineTuner.

        Args:
            model: The model to be fine-tuned.
            tokenizer: The tokenizer associated with the model.
            train_dataset: The dataset to be used for training.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset

    def train(self):
        """
        Executes the fine-tuning process.
        """
        training_args = transformers.TrainingArguments(
            output_dir=TrainingConfig.OUTPUT_DIR,
            num_train_epochs=TrainingConfig.NUM_TRAIN_EPOCHS,
            per_device_train_batch_size=TrainingConfig.PER_DEVICE_TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=4,
            learning_rate=TrainingConfig.LEARNING_RATE,
            fp16=TrainingConfig.FP16,
            logging_dir=TrainingConfig.LOGGING_DIR,
            logging_steps=TrainingConfig.LOGGING_STEPS,
            optim="paged_adamw_8bit",
            save_total_limit=3,
            save_strategy="epoch",
        )

        trainer = transformers.Trainer(
            model=self.model,
            train_dataset=self.train_dataset,
            args=training_args,
            data_collator=transformers.DataCollatorForLanguageModeling(
                self.tokenizer, mlm=False
            ),
        )
        
        self.model.config.use_cache = False
        trainer.train()
