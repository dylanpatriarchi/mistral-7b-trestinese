# configs/config.py

class TrainingConfig:
    """
    Configuration for the training process.
    """
    MODEL_NAME = "mistralai/Mistral-7B-v0.1"
    DATASET_PATH = "dataset.jsonl"
    OUTPUT_DIR = "./results"
    NUM_TRAIN_EPOCHS = 3
    PER_DEVICE_TRAIN_BATCH_SIZE = 4
    PER_DEVICE_EVAL_BATCH_SIZE = 4
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 0.01
    LOGGING_DIR = "./logs"
    LOGGING_STEPS = 10
    LEARNING_RATE = 2e-4
    FP16 = False
    MAX_SEQ_LENGTH = 512

class LoRAConfig:
    """
    Configuration for LoRA (Low-Rank Adaptation).
    """
    R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    TARGET_MODULES = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]

class PromptConfig:
    """
    Configuration for the prompt format.
    """
    PROMPT_TEMPLATE = """### Instruction:
Translate the following Italian text to Trestinese.

### Italian:
{italian}

### Trestinese:
{trestinese}"""
