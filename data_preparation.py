"""
Data preparation script for Mistral-7B Trestinese fine-tuning.
Converts the dataset into a format suitable for training.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
from utils import setup_logging, load_config, create_prompt, set_seed


logger = setup_logging()


def load_jsonl(file_path: str) -> List[Dict[str, str]]:
    """
    Load data from JSONL file.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of dictionaries containing the data
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def save_jsonl(data: List[Dict], file_path: str) -> None:
    """
    Save data to JSONL file.
    
    Args:
        data: List of dictionaries to save
        file_path: Output file path
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def create_training_example(item: Dict[str, str], instruction: str) -> Dict[str, str]:
    """
    Create a training example with instruction-input-output format.
    
    Args:
        item: Dictionary with 'italiano' and 'trestinese' keys
        instruction: Task instruction
        
    Returns:
        Dictionary with formatted prompt
    """
    prompt = create_prompt(
        instruction=instruction,
        input_text=item['italiano'],
        output_text=item['trestinese']
    )
    
    return {
        'text': prompt,
        'italiano': item['italiano'],
        'trestinese': item['trestinese']
    }


def prepare_dataset(
    input_file: str,
    output_dir: str,
    train_split: float = 0.9,
    seed: int = 42,
    instruction: str = "Translate the following Italian text to Trestinese dialect."
) -> Tuple[int, int]:
    """
    Prepare the dataset by splitting into train and validation sets.
    
    Args:
        input_file: Path to input JSONL file
        output_dir: Directory to save processed datasets
        train_split: Proportion of data for training
        seed: Random seed for reproducibility
        instruction: Task instruction to use
        
    Returns:
        Tuple of (train_size, validation_size)
    """
    # Set seed for reproducibility
    set_seed(seed)
    
    # Load data
    logger.info(f"Loading data from {input_file}")
    data = load_jsonl(input_file)
    logger.info(f"Loaded {len(data)} examples")
    
    # Split data
    train_data, val_data = train_test_split(
        data,
        train_size=train_split,
        random_state=seed,
        shuffle=True
    )
    
    logger.info(f"Split data: {len(train_data)} train, {len(val_data)} validation")
    
    # Create formatted examples
    logger.info("Creating formatted training examples...")
    train_examples = [create_training_example(item, instruction) for item in train_data]
    val_examples = [create_training_example(item, instruction) for item in val_data]
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save processed datasets
    train_path = Path(output_dir) / "train.jsonl"
    val_path = Path(output_dir) / "validation.jsonl"
    
    logger.info(f"Saving training data to {train_path}")
    save_jsonl(train_examples, str(train_path))
    
    logger.info(f"Saving validation data to {val_path}")
    save_jsonl(val_examples, str(val_path))
    
    # Print statistics
    logger.info("\n" + "=" * 60)
    logger.info("DATASET STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Total examples:      {len(data)}")
    logger.info(f"Training examples:   {len(train_examples)}")
    logger.info(f"Validation examples: {len(val_examples)}")
    logger.info(f"Train split:         {train_split * 100:.1f}%")
    logger.info(f"Validation split:    {(1 - train_split) * 100:.1f}%")
    logger.info("=" * 60)
    
    # Print example
    logger.info("\nExample training sample:")
    logger.info("-" * 60)
    logger.info(train_examples[0]['text'])
    logger.info("-" * 60)
    
    return len(train_examples), len(val_examples)


def analyze_dataset(data: List[Dict[str, str]]) -> None:
    """
    Analyze and print statistics about the dataset.
    
    Args:
        data: List of data examples
    """
    italiano_lengths = [len(item['italiano'].split()) for item in data]
    trestinese_lengths = [len(item['trestinese'].split()) for item in data]
    
    logger.info("\n" + "=" * 60)
    logger.info("DETAILED DATASET ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Total examples: {len(data)}")
    logger.info("\nItalian text statistics:")
    logger.info(f"  Avg words: {sum(italiano_lengths) / len(italiano_lengths):.2f}")
    logger.info(f"  Min words: {min(italiano_lengths)}")
    logger.info(f"  Max words: {max(italiano_lengths)}")
    logger.info("\nTrestinese text statistics:")
    logger.info(f"  Avg words: {sum(trestinese_lengths) / len(trestinese_lengths):.2f}")
    logger.info(f"  Min words: {min(trestinese_lengths)}")
    logger.info(f"  Max words: {max(trestinese_lengths)}")
    logger.info("=" * 60 + "\n")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Prepare dataset for Mistral-7B Trestinese fine-tuning"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="dataset.jsonl",
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Directory to save processed datasets"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Perform detailed dataset analysis"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    dataset_config = config.get('dataset', {})
    
    # Override with command line arguments
    input_file = args.input_file or dataset_config.get('path', 'dataset.jsonl')
    train_split = dataset_config.get('train_split', 0.9)
    seed = dataset_config.get('seed', 42)
    
    # Analyze dataset if requested
    if args.analyze:
        data = load_jsonl(input_file)
        analyze_dataset(data)
    
    # Prepare dataset
    prepare_dataset(
        input_file=input_file,
        output_dir=args.output_dir,
        train_split=train_split,
        seed=seed
    )
    
    logger.info("Dataset preparation complete!")


if __name__ == "__main__":
    main()

