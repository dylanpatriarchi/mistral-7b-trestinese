# Quick Start Guide

## 🚀 Fastest Way to Get Started

### Option 1: Google Colab (Recommended for Beginners)

1. **Upload to Google Drive**
   - Upload all project files to Google Drive
   - Open `colab_notebook.ipynb` with Google Colab

2. **Follow the Notebook**
   - The notebook guides you through each step
   - Training takes ~30-45 minutes on free T4 GPU

3. **Done!**
   - Your model will be ready to translate Italian to Trestinese

### Option 2: Local Training (For Advanced Users)

```bash
# 1. Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Prepare data
python data_preparation.py --input_file dataset.jsonl --output_dir ./data

# 3. Train
python train.py --config config.yaml --data_dir ./data

# 4. Test
python inference.py --model_path ./output/final_model --interactive
```

## 📋 What You Need

- **For Colab**: Free Google account (no GPU required on your computer!)
- **For Local**: 
  - Python 3.8+
  - 16GB+ RAM
  - NVIDIA GPU with 16GB+ VRAM
  - CUDA 11.8+

## 🎯 Expected Results

After training on 303 examples:
- **BLEU Score**: 0.70-0.80
- **Translation Quality**: Good for common phrases
- **Training Time**: 30-45 minutes (T4 GPU)

## 💡 Tips

1. **Use Google Colab** if you don't have a powerful GPU
2. **Mount Google Drive** to save your model permanently
3. **Start with default settings** - they're already optimized
4. **Add more data** if you want better quality

## 🆘 Need Help?

- Check the full [README.md](README.md) for detailed instructions
- Look at [Troubleshooting](README.md#troubleshooting) section
- Open an issue on GitHub

---

**Ready? Let's start with the Colab notebook! 🚀**

