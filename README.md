# 🖼️ Image Caption Generator using Deep Learning (NLP + Computer Vision)

This project implements an end-to-end **Image Captioning** model that generates natural language descriptions for input images using deep learning techniques. It combines **Convolutional Neural Networks (CNNs)** for image feature extraction and **Recurrent Neural Networks (RNNs)** or **Transformer-based models** for sequence generation.

## 🚀 Features

- Uses **CNNs** (e.g., InceptionV3 or ResNet) for image feature extraction
- Applies **NLP techniques** with **RNNs/LSTMs** or **Transformer decoders** for caption generation
- Trained on the **Flickr8k** dataset (can be extended to Flickr30k or MSCOCO)
- Includes **image preprocessing**, **tokenization**, and **caption beam search decoding**
- Flask REST API ready for inference deployment

## 🛠️ Tech Stack

- Python
- TensorFlow / PyTorch
- Flask (for serving the model)
- TQDM (for progress bar)
- Numpy, Matplotlib, PIL, etc.

## 📁 Dataset

- [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)

## 📸 Example Output

> Input: ![Sample Image](./sample.jpg)  
> Output: `"a group of people walking down the street"`

## 🔧 How to Use

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/image-captioning-app.git
cd image-captioning-app

###IInstall Requirements
pip install -r requirements.txt


/model
  ├── feature_extractor.keras
  ├── tokenizer.pkl
  └── model.keras

