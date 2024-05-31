# Model 7

import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(parent_dir)
sys.path.insert(0, parent_dir)

from torch import nn
import torch
import random
from const import *
from hyperparm import *
import torch.optim as optim

# from torchvision import models
from torch.optim import Adam
from torchvision.models import resnet18, ResNet18_Weights


class Encoder(nn.Module):
    def __init__(self, encoder_dim, decoder_dim):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # for param in resnet.parameters():
        #     param.requires_grad_(False)

        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

    def forward(self, images):
        features = self.resnet(images)
        features = features.permute(0, 2, 3, 1)
        outputs = features.view(features.size(0), -1, features.size(-1))

        mean_encoder_out = outputs.mean(dim=1)
        hidden = self.init_h(mean_encoder_out)
        cell = self.init_c(mean_encoder_out)
        return outputs, (hidden.unsqueeze(0), cell.unsqueeze(0))


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()

        self.attention_dim = attention_dim

        self.W = nn.Linear(decoder_dim, attention_dim)
        self.U = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.A = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, encoder_outputs, hidden):
        u_hs = self.U(encoder_outputs)  # (batch_size, seq_len, attention_dim)
        w_ah = self.W(hidden).unsqueeze(1)  # (batch_size, 1, attention_dim)
        combined_states = u_hs + w_ah
        combined_states = torch.tanh(combined_states)

        attention_scores = self.A(combined_states)  # (batch_size, seq_len, 1)
        attention_scores = attention_scores.squeeze(2)

        return torch.softmax(attention_scores, dim=1)


class Decoder(nn.Module):
    def __init__(
        self,
        output_dim,
        embedding_dim,
        encoder_dim,
        decoder_dim,
        dropout,
        attention,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.LSTM(encoder_dim + embedding_dim, decoder_dim)
        self.fc_out = nn.Linear(encoder_dim + decoder_dim + embedding_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, encoder_outputs, hidden):
        input = input.unsqueeze(0)

        embedded = self.dropout(self.embedding(input))

        a = self.attention(
            encoder_outputs, hidden[0][-1]
        )  # Use the last layer of LSTM hidden state

        a = a.unsqueeze(1)

        weighted = torch.bmm(a, encoder_outputs)

        weighted = weighted.permute(1, 0, 2)

        rnn_input = torch.cat((embedded, weighted), dim=2)

        output, hidden = self.rnn(rnn_input, hidden)

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))

        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        trg_length = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_length, batch_size, trg_vocab_size).to(self.device)

        encoder_outputs, hidden = self.encoder(src)

        input = trg[0]

        for t in range(1, trg_length):
            output, hidden = self.decoder(input, encoder_outputs, hidden)

            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio

            top1 = output.argmax(1)

            input = trg[t] if teacher_force else top1

        return outputs


encoder = Encoder(encoder_dim, decoder_dim)

attention = Attention(encoder_dim, decoder_dim, attention_dim)

decoder = Decoder(
    output_dim, embedding_dim, encoder_dim, decoder_dim, decoder_dropout, attention
)


model = Seq2Seq(encoder, decoder, device).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss(ignore_index=pad_index)
# criterion = nn.MSELoss(reduction="mean")


if __name__ == "__main__":
    print("Model Architecture:\n")
    print(model)
    print("\nModel Parameters:\n")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")
