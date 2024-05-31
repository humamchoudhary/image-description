import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.nn import (
    TransformerEncoder,
    TransformerEncoderLayer,
    TransformerDecoder,
    TransformerDecoderLayer,
)
import torch.optim as optim
import math
from const import *
from hyperparm import *


IMAGE_SIZE = (224, 224)
EMBED_DIM = 1280
FF_DIM = 2048
SEQ_LENGTH = 20
VOCAB_SIZE = 10000


# Define the CNN model using EfficientNetB7
class EfficientNetB1CNN(nn.Module):
    def __init__(self, image_size):
        super(EfficientNetB1CNN, self).__init__()
        self.base_model = models.efficientnet_b1(
            weights=models.EfficientNet_B1_Weights.DEFAULT
        )
        self.base_model.features[0][0] = nn.Conv2d(
            3, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.base_model.classifier = nn.Identity()
        self.output_shape = self.get_output_shape(image_size)

    def get_output_shape(self, image_size):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, *image_size)
            features = self.base_model.features(dummy_input)
            return features.size()

    def forward(self, x):
        x = self.base_model.features(x)
        x = x.permute(0, 2, 3, 1)  # Change to (B, H, W, C)
        x = x.reshape(x.size(0), -1, x.size(-1))  # Flatten to (B, H*W, C)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0).transpose(0, 1)

    def forward(self, x):
        return x + self.pe[: x.size(0), :]


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super(TransformerEncoderBlock, self).__init__()
        self.encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim
        )
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=1)

    def forward(self, x, mask=None):
        print(x.shape)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, vocab_size, seq_length):
        super(TransformerDecoderBlock, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, seq_length)
        self.decoder_layer = TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim
        )
        self.transformer_decoder = TransformerDecoder(self.decoder_layer, num_layers=1)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt = self.embedding(tgt) * math.sqrt(tgt.size(-1))
        tgt = self.positional_encoding(tgt)
        output = self.transformer_decoder(
            tgt, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_mask
        )
        output = self.fc(output)
        return output


class ImageCaptioningModel(nn.Module):
    def __init__(self, cnn_model, encoder, decoder, num_captions_per_image=1):
        super(ImageCaptioningModel, self).__init__()
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder
        self.num_captions_per_image = num_captions_per_image

    def forward(self, images, captions):
        cnn_features = self.cnn_model(images)
        encoder_output = self.encoder(cnn_features)
        decoder_output = self.decoder(captions, encoder_output)
        return decoder_output


# Instantiate the models
cnn_model = EfficientNetB1CNN(IMAGE_SIZE)
encoder = TransformerEncoderBlock(embed_dim=EMBED_DIM, num_heads=2, ff_dim=FF_DIM)
decoder = TransformerDecoderBlock(
    embed_dim=EMBED_DIM,
    num_heads=8,
    ff_dim=FF_DIM,
    vocab_size=VOCAB_SIZE,
    seq_length=SEQ_LENGTH,
)

model = ImageCaptioningModel(cnn_model=cnn_model, encoder=encoder, decoder=decoder)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=pad_index)
# Print model summaries
if __name__ == "__main__":
    print(model)
