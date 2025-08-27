#  Image Captioning with Neural Networks

This project implements an **Image Captioning System** using a **Convolutional Neural Network (CNN)** for image feature extraction and an **Artificial Neural Network (ANN/LSTM)** for sequence modeling.  

It generates **human-like captions** for input images. The system is wrapped inside a **Streamlit app** for an interactive UI.

---

##  Tech Stack
- **Frontend/UI:** Streamlit
- **Feature Extraction:** Xception (pre-trained CNN on ImageNet)
- **Caption Generator:** ANN + LSTM (Keras/TensorFlow)
- **Dataset:** [Flickr8k](https://github.com/jbrownlee/Datasets/releases/tag/Flickr8k_Dataset) (images + text captions)
- **Language Processing:** Tokenization + Word Indexing

---

## How It Works

1. **Image Feature Extraction**
   - Input image is passed through a pre-trained **Xception CNN** (with top layer removed).
   - Produces a **2048-dim feature vector** representing the image.

2. **Text Preprocessing**
   - Captions are cleaned (lowercase, remove punctuation, numbers).
   - Tokenized using Keras `Tokenizer`.
   - Vocabulary of ~7k words created.
   - Special tokens `startseq` and `endseq` added.

3. **Sequence Modeling**
   - Model takes two inputs:
     - **Image features** (from CNN).
     - **Partial caption sequence** (words so far).
   - Uses **Embedding + LSTM** layers to predict the **next word**.
   - Trained using `sparse_categorical_crossentropy`.

4. **Caption Generation**
   - At inference, start with `startseq`.
   - Iteratively predict next word until `endseq` is generated or max length is reached.
   - Produces natural captions like:  
     > *"a dog running across a field with a ball"*

---


