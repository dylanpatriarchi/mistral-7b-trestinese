# Mistral-7B Trestinese Fine-Tuning

This project fine-tunes the Mistral-7B model to translate from Italian to Trestinese, a local dialect.

## Project Structure

```
.
├───dataset.jsonl
├───requirements.txt
├───README.md
├───mistral_finetune/
│   ├───__init__.py
│   ├───data_loader.py
│   ├───model.py
│   └───trainer.py
├───configs/
│   └───config.py
└───scripts/
    └───run_finetuning.py
```

- `dataset.jsonl`: The dataset containing Italian-Trestinese sentence pairs.
- `requirements.txt`: The list of Python dependencies.
- `README.md`: This file.
- `mistral_finetune/`: The main Python package for the fine-tuning logic.
  - `data_loader.py`: Handles loading and preprocessing of the dataset.
  - `model.py`: Manages the Mistral-7B model and tokenizer.
  - `trainer.py`: Contains the logic for the fine-tuning process.
- `configs/`: Configuration files.
  - `config.py`: Main configuration for the project (e.g., model name, hyperparameters).
- `scripts/`: Scripts to run the project.
  - `run_finetuning.py`: The entry point to start the fine-tuning process.

## How to Use

1.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the fine-tuning:**

    ```bash
    python scripts/run_finetuning.py
    ```

## Note

The provided `dataset.jsonl` is small. For better results, a larger and more diverse dataset is recommended.
