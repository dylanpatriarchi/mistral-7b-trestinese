# Mistral-7B Trestinese Fine-Tuning

A professional fine-tuning pipeline for Mistral-7B to translate Italian to Trestinese dialect using LoRA/QLoRA for efficient training.

## 🌟 Features

- **Efficient Training**: Uses QLoRA (4-bit quantization) with LoRA adapters for memory-efficient fine-tuning
- **Professional Pipeline**: Complete data preparation, training, evaluation, and inference scripts
- **Comprehensive Metrics**: BLEU, ROUGE, Character Error Rate (CER), and Word Accuracy
- **Google Colab Support**: Ready-to-use Colab notebook with GPU support
- **Monitoring**: Weights & Biases integration for experiment tracking
- **Interactive Mode**: Test your model with an interactive translation interface

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Google Colab Tutorial](#google-colab-tutorial)
- [Dataset Format](#dataset-format)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Performance Metrics](#performance-metrics)
- [Troubleshooting](#troubleshooting)

## 🚀 Installation

### Local Installation

```bash
# Clone the repository
git clone https://github.com/dylanpatriarchi/mistral-7b-trestinese
cd mistral-7b-trestinese

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### GPU Requirements

- **Minimum**: 16GB VRAM (with 4-bit quantization)
- **Recommended**: 24GB+ VRAM
- **Colab**: T4 GPU (free tier) or A100 (Pro tier)

## ⚡ Quick Start

### 1. Prepare Your Dataset

```bash
python data_preparation.py \
    --input_file dataset.jsonl \
    --output_dir ./data \
    --analyze
```

This will:
- Split dataset into train (90%) and validation (10%) sets
- Create formatted prompts for training
- Analyze dataset statistics

### 2. Train the Model

```bash
python train.py \
    --config config.yaml \
    --data_dir ./data
```

### 3. Evaluate the Model

```bash
python evaluate.py \
    --model_path ./output/final_model \
    --dataset_path ./data/validation.jsonl \
    --output_dir ./evaluation_results
```

### 4. Run Inference

**Interactive Mode:**
```bash
python inference.py \
    --model_path ./output/final_model \
    --interactive
```

**Single Translation:**
```bash
python inference.py \
    --model_path ./output/final_model \
    --text "Ciao, come stai?"
```

**Batch Translation:**
```bash
python inference.py \
    --model_path ./output/final_model \
    --input_file input.txt \
    --output_file translations.jsonl
```

## 📓 Google Colab Tutorial

### Step-by-Step Colab Setup

1. **Open the Colab Notebook**
   - Upload `colab_notebook.ipynb` to Google Drive
   - Open with Google Colab
   - Or use the direct link: [Open in Colab](#)

2. **Enable GPU**
   ```
   Runtime → Change runtime type → Hardware accelerator → GPU (T4)
   ```

3. **Install Dependencies**
   ```python
   !pip install -q transformers peft datasets accelerate bitsandbytes
   !pip install -q evaluate rouge-score sacrebleu wandb
   ```

4. **Clone Your Repository**
   ```python
   !git clone <your-repo-url>
   %cd mistral-7b-trestinese
   ```

5. **Upload Your Dataset**
   ```python
   from google.colab import files
   uploaded = files.upload()  # Upload dataset.jsonl
   ```

6. **Prepare Data**
   ```python
   !python data_preparation.py --input_file dataset.jsonl --output_dir ./data
   ```

7. **Configure Training**
   Edit `config.yaml` or use Colab-optimized settings:
   ```python
   # For free T4 GPU (15GB)
   config_overrides = {
       'model': {'load_in_4bit': True},
       'training': {
           'per_device_train_batch_size': 2,
           'gradient_accumulation_steps': 8,
           'num_train_epochs': 3
       }
   }
   ```

8. **Start Training**
   ```python
   !python train.py --config config.yaml --data_dir ./data
   ```

9. **Monitor Training**
   - Check output logs
   - View W&B dashboard (if configured)
   - Monitor GPU usage: `!nvidia-smi`

10. **Evaluate**
    ```python
    !python evaluate.py \
        --model_path ./output/final_model \
        --dataset_path ./data/validation.jsonl
    ```

11. **Test Your Model**
    ```python
    !python inference.py \
        --model_path ./output/final_model \
        --text "Ciao, come stai?"
    ```

12. **Download Your Model**
    ```python
    from google.colab import files
    !zip -r fine_tuned_model.zip ./output/final_model
    files.download('fine_tuned_model.zip')
    ```

### Colab Memory Optimization Tips

**For Free Tier (T4 15GB):**
```yaml
model:
  load_in_4bit: true
  
training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  gradient_checkpointing: true
```

**For Pro Tier (A100 40GB):**
```yaml
model:
  load_in_4bit: false  # Can use full precision
  
training:
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 2
```

## 📊 Dataset Format

Your dataset should be in JSONL format with Italian-Trestinese pairs:

```jsonl
{"italiano": "Ciao, come stai?", "trestinese": "Bona, com'è?"}
{"italiano": "Andiamo a bere un caffè.", "trestinese": "Gimo a be ncaffé."}
{"italiano": "Cosa fai oggi?", "trestinese": "Che fe oggi?"}
```

The training pipeline will automatically format these into instruction-following prompts:

```
### Instruction:
Translate the following Italian text to Trestinese dialect.

### Input:
Ciao, come stai?

### Output:
Bona, com'è?
```

## 🎯 Training

### Configuration

Edit `config.yaml` to customize training parameters:

```yaml
model:
  name: "mistralai/Mistral-7B-v0.1"
  load_in_4bit: true  # Use QLoRA

lora:
  r: 16  # LoRA rank
  lora_alpha: 32
  lora_dropout: 0.05

training:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  learning_rate: 2.0e-4
  gradient_accumulation_steps: 4
```

### Training Arguments

- `--config`: Path to configuration file
- `--data_dir`: Directory with prepared datasets
- `--resume_from_checkpoint`: Resume from a checkpoint

### Monitoring Training

The training script provides:
- Real-time loss tracking
- Validation metrics every `eval_steps`
- Model checkpoints saved every `save_steps`
- GPU utilization monitoring
- W&B integration (optional)

### Expected Training Time

- **Dataset**: 303 examples (90/10 split)
- **T4 GPU**: ~30-45 minutes for 3 epochs
- **A100 GPU**: ~15-20 minutes for 3 epochs

## 📈 Evaluation

The evaluation script computes multiple metrics:

### Metrics

1. **BLEU Score**: Translation quality (0-1, higher is better)
2. **ROUGE-1/2/L**: N-gram overlap with references
3. **Character Error Rate (CER)**: Character-level accuracy
4. **Word Accuracy**: Word-level matching

### Running Evaluation

```bash
python evaluate.py \
    --model_path ./output/final_model \
    --dataset_path ./data/validation.jsonl \
    --output_dir ./evaluation_results \
    --base_model mistralai/Mistral-7B-v0.1
```

### Example Output

```
==============================================================
                   EVALUATION METRICS
==============================================================
BLEU                : 0.7523
ROUGE1              : 0.8234
ROUGE2              : 0.7156
ROUGEL              : 0.8012
CER                 : 0.0876
WORD_ACCURACY       : 0.8567
==============================================================
```

## 🔮 Inference

### Interactive Mode

Start an interactive translation session:

```bash
python inference.py \
    --model_path ./output/final_model \
    --interactive
```

```
🇮🇹 Italian: Ciao, come stai?
🗣️  Trestinese: Bona, com'è?

🇮🇹 Italian: Andiamo a mangiare
🗣️  Trestinese: Gimo a magnà
```

### Single Translation

```bash
python inference.py \
    --model_path ./output/final_model \
    --text "Buongiorno, vuoi un caffè?"
```

### Batch Translation

Create `input.txt`:
```
Ciao, come stai?
Dove vai?
Andiamo a mangiare.
```

Run:
```bash
python inference.py \
    --model_path ./output/final_model \
    --input_file input.txt \
    --output_file translations.jsonl
```

## ⚙️ Configuration

### Model Configuration

```yaml
model:
  name: "mistralai/Mistral-7B-v0.1"  # Base model
  load_in_4bit: true                   # 4-bit quantization
  torch_dtype: "bfloat16"              # Data type
```

### LoRA Configuration

```yaml
lora:
  r: 16                    # Rank (higher = more parameters)
  lora_alpha: 32          # Scaling factor
  target_modules:         # Modules to apply LoRA to
    - q_proj
    - k_proj
    - v_proj
    - o_proj
  lora_dropout: 0.05      # Dropout rate
```

### Training Configuration

```yaml
training:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  warmup_ratio: 0.03
  evaluation_strategy: "steps"
  eval_steps: 100
  save_steps: 100
```

### Generation Configuration

```yaml
generation:
  max_new_tokens: 256
  temperature: 0.7        # Randomness (0-1)
  top_p: 0.9             # Nucleus sampling
  top_k: 50              # Top-k sampling
  repetition_penalty: 1.1
```

## 📁 Project Structure

```
mistral-7b-trestinese/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config.yaml              # Configuration file
├── dataset.jsonl            # Your dataset
├── data_preparation.py      # Dataset preparation script
├── train.py                 # Training script
├── evaluate.py              # Evaluation script
├── inference.py             # Inference script
├── utils.py                 # Utility functions
├── colab_notebook.ipynb     # Google Colab notebook
├── data/                    # Prepared datasets
│   ├── train.jsonl
│   └── validation.jsonl
├── output/                  # Training outputs
│   ├── checkpoint-*/
│   └── final_model/
└── evaluation_results/      # Evaluation results
    ├── metrics.json
    └── predictions.jsonl
```

## 📊 Performance Metrics

### Model Size

- **Base Model**: 7B parameters (~14GB)
- **Trainable Parameters** (LoRA): ~4M (0.06%)
- **Memory Usage**: 8-10GB with 4-bit quantization

### Training Efficiency

- **Effective Batch Size**: `batch_size × gradient_accumulation_steps × num_gpus`
- **Example**: 4 × 4 × 1 = 16 effective batch size

### Expected Performance

With 303 training examples:
- **BLEU**: 0.70-0.80
- **ROUGE-L**: 0.75-0.85
- **Word Accuracy**: 0.80-0.90

Performance improves with:
- More training data
- Longer training time
- Larger LoRA rank
- Better hyperparameter tuning

## 🔧 Troubleshooting

### Out of Memory (OOM)

**Solutions:**
1. Enable 4-bit quantization: `load_in_4bit: true`
2. Reduce batch size: `per_device_train_batch_size: 2`
3. Increase gradient accumulation: `gradient_accumulation_steps: 8`
4. Enable gradient checkpointing (already enabled)

### Slow Training

**Solutions:**
1. Increase batch size if memory allows
2. Reduce `logging_steps` and `eval_steps`
3. Use mixed precision training: `bf16: true`
4. Use a more powerful GPU

### Poor Translation Quality

**Solutions:**
1. Increase training epochs: `num_train_epochs: 5`
2. Add more training data
3. Increase LoRA rank: `r: 32`
4. Adjust learning rate: try `1.0e-4` or `3.0e-4`
5. Adjust generation parameters: lower `temperature` for more deterministic outputs

### Model Not Loading

**Check:**
1. Model path is correct
2. All adapter files are present (`adapter_config.json`, `adapter_model.bin`)
3. Tokenizer files are saved with the model
4. Base model name matches training configuration

### W&B Connection Issues

```python
# Disable W&B if needed
export WANDB_MODE=offline
# Or in config.yaml:
training:
  report_to: "none"
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 Citation

If you use this code, please cite:

```bibtex
@software{mistral7b_trestinese,
  title = {Mistral-7B Trestinese Fine-Tuning},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/dylanpatriarchi/mistral-7b-trestinese}
}
```

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [Mistral AI](https://mistral.ai/) for the Mistral-7B model
- [Hugging Face](https://huggingface.co/) for the transformers library
- [Microsoft](https://github.com/microsoft/LoRA) for LoRA
- [Tim Dettmers](https://github.com/TimDettmers/bitsandbytes) for bitsandbytes

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com].

---

**Happy Fine-Tuning! 🚀**

