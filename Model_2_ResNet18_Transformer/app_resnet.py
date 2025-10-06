import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter 
import torchvision
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F
import pickle
import gc
import random

# ---- Load Required Items ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load vocabulary
with open("word_to_index.pkl", "rb") as f:
    word_to_index = pickle.load(f)

with open("index_to_word.pkl", "rb") as f:
    index_to_word = pickle.load(f)

vocab_size = len(word_to_index)
max_seq_len = 33
start_token = word_to_index["<start>"]
pad_token = word_to_index["<pad>"]

# ---- Image Preprocessing ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---- Feature Extractor ----
resnet = models.resnet18(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-2])  # Remove last pooling and FC layers
resnet.eval()
resnet.to(device)

# ---- Positional Encoding ----
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=max_seq_len):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return self.dropout(x)

# ---- Model Definition ----
class ImageCaptionModel(nn.Module):
    def __init__(self, n_head, n_decoder_layer, vocab_size, embedding_size):
        super().__init__()
        self.pos_encoder = PositionalEncoding(embedding_size, 0.1)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_size, nhead=n_head)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layer)
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.fc_out = nn.Linear(embedding_size, vocab_size)
        self.embedding_size = embedding_size

    def forward(self, encoded_image, decoder_input):
        memory = encoded_image.permute(1, 0, 2)  # (seq_len, batch, embed)
        tgt_emb = self.embedding(decoder_input) * np.sqrt(self.embedding_size)
        tgt_emb = self.pos_encoder(tgt_emb)
        tgt = tgt_emb.permute(1, 0, 2)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)
        out = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
        return self.fc_out(out)

# ---- Load Trained Model ----
model = ImageCaptionModel(16, 4, vocab_size, 512).to(device)
state_dict = torch.load("resent_transformer.pth", map_location=device)
# Clean up the state dict to skip positional encoding mismatch
state_dict = {k: v for k, v in state_dict.items() if 'pos_encoder.pe' not in k}
model.load_state_dict(state_dict, strict=False)
model.eval()

# ---- Caption Generator ----
def generate_caption(model, img_tensor):
    model.eval()
    with torch.no_grad():
        # Extract features
        features = resnet(img_tensor).to(device)  # (1, 512, 7, 7)
        features = features.permute(0, 2, 3, 1).reshape(1, -1, 512)

        input_seq = [pad_token] * max_seq_len
        input_seq[0] = start_token
        input_tensor = torch.tensor(input_seq).unsqueeze(0).to(device)

        output_caption = []
        for t in range(max_seq_len - 1):
            outputs = model(features, input_tensor)
            next_token_logits = outputs[t, 0, :]

            next_word_idx = torch.argmax(next_token_logits).item()
            next_word = index_to_word.get(next_word_idx, "<unk>")

            input_tensor[0, t + 1] = next_word_idx
            if next_word == "<end>":
                break
            output_caption.append(next_word)

    return " ".join(output_caption)


# ---- Streamlit UI ----
st.set_page_config(page_title="🖼️ Image Captioning with Transformers", layout="centered")
st.title("🧠📸 Image Captioning App")
st.markdown("Upload an image and get a generated caption using a Transformer decoder over ResNet features.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Generating caption..."):
        img_tensor = transform(img).unsqueeze(0).to(device)
        caption = generate_caption(model, img_tensor)

    st.success("Generated Caption:")
    st.markdown(f"> {caption}")
