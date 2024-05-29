import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from torch import nn
import torch
import random
from const import *
from hyperparm import *
import torch.optim as optim
from torchvision import models

from torchvision.models import resnet18, ResNet18_Weights


class Encoder(nn.Module):
    def __init__(self, encoder_dim, decoder_dim):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        for param in resnet.parameters():
            param.requires_grad_(False)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(encoder_dim, embedding_dim)

    def forward(self, images):
        features = self.resnet(images)
        features = features.permute(0, 2, 3, 1)
        outputs = features.view(features.size(0), -1, features.size(-1))
        outputs = self.linear(outputs)
        return outputs


class Decoder(nn.Module):
    def __init__(
        self,
        output_dim,
        embedding_dim,
        n_heads,
        num_decoder_layers,
        dim_feedforward,
        dropout,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 500, embedding_dim))
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer_decoder = nn.TransformerDecoder(
            self.transformer_decoder_layer, num_layers=num_decoder_layers
        )
        self.fc_out = nn.Linear(embedding_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, memory):
        trg = self.embedding(trg) + self.positional_encoding[:, : trg.size(0), :]
        trg = self.dropout(trg)
        memory = memory.permute(1, 0, 2)
        trg = trg.permute(1, 0, 2)
        output = self.transformer_decoder(trg, memory)
        output = output.permute(1, 0, 2)
        output = self.fc_out(output)
        return output


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        encoder_outputs = self.encoder(src)
        batch_size = encoder_outputs.size(0)
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # Prepare the encoder outputs for the decoder
        encoder_outputs = encoder_outputs.permute(
            1, 0, 2
        )  # Reshape to [seq_len, batch_size, hidden_size]

        input = trg[0, :]

        for t in range(1, trg_len):
            output = self.decoder(input.unsqueeze(0), encoder_outputs)
            outputs[t] = output.squeeze(0)
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(2)
            input = trg[t] if teacher_force else top1.squeeze(0)

        return outputs


n_heads = 8
num_decoder_layers = 6
dim_feedforward = 1024

encoder = Encoder(encoder_dim, embedding_dim).to(device)
decoder = Decoder(
    output_dim,
    embedding_dim,
    n_heads,
    num_decoder_layers,
    dim_feedforward,
    decoder_dropout,
).to(device)
model = Seq2Seq(encoder, decoder, device).to(device)


# model = Seq2Seq(encoder, decoder, device).to(device)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=pad_index)


if __name__ == "__main__":
    print("Model Architecture:\n")
    print(model)
    print("\nModel Parameters:\n")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")
