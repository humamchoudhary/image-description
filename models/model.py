import torch
import torch.nn as nn
import random
import pandas as pd

# import torchvision.models as models
import spacy
from torchtext import vocab



from torchvision.models import resnet50, ResNet50_Weights


class Encoder(nn.Module):
    def __init__(self, encoder_dim, decoder_dim):
        super().__init__()
        # resnet = ResNet18(10000)
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)

    def forward(self, images):
        features = self.resnet(images)  # (batch_size,2048,7,7)
        features = features.permute(0, 2, 3, 1)  # (batch_size,7,7,2048)
        outputs = features.view(
            features.size(0), -1, features.size(-1)
        )  # (batch_size,49,2048) #(batch_size , num_layers , encoder hidden dim)

        mean_encoder_out = outputs.mean(dim=1)
        hidden = self.init_h(mean_encoder_out)  # (batch_size, decoder hidden dim)
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()

        self.attention_dim = attention_dim

        self.W = nn.Linear(decoder_dim, attention_dim)
        self.U = nn.Linear(encoder_dim, attention_dim)

        self.A = nn.Linear(attention_dim, 1)

    def forward(self, encoder_outputs, hidden):

        u_hs = self.U(encoder_outputs)
        w_ah = self.W(hidden)

        combined_states = torch.tanh(u_hs + w_ah.unsqueeze(1))

        attention_scores = self.A(combined_states)
        attention_scores = attention_scores.squeeze(2)

        return torch.softmax(attention_scores, dim=1)

        # class Decoder(nn.Module):
        #     def __init__(
        #         self,
        #         output_dim,
        #         embedding_dim,
        #         encoder_dim,
        #         decoder_dim,
        #         dropout,
        #         attention,
        #     ):
        #         super().__init__()
        #         self.output_dim = output_dim
        #         self.attention = attention
        #         self.embedding = nn.Embedding(output_dim, embedding_dim)
        #         self.rnn = nn.GRU(encoder_dim + embedding_dim, decoder_dim)
        #         self.fc_out = nn.Linear(encoder_dim + decoder_dim + embedding_dim, output_dim)
        #         self.dropout = nn.Dropout(dropout)

        #     def forward(self, input, encoder_outputs, hidden):

        input = input.unsqueeze(0)

        embedded = self.dropout(self.embedding(input))

        a = self.attention(encoder_outputs, hidden)

        a = a.unsqueeze(1)

        weighted = torch.bmm(a, encoder_outputs)

        weighted = weighted.permute(1, 0, 2)

        rnn_input = torch.cat((embedded, weighted), dim=2)

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        assert (output == hidden).all()
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))

        return prediction, hidden.squeeze(0), a.squeeze(1)


class Decoder(nn.Module):
    def __init__(
        self, output_dim, embedding_dim, encoder_dim, decoder_dim, dropout, attention
    ):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.LSTM(encoder_dim + embedding_dim, decoder_dim, bidirectional=True)
        self.fc_out = nn.Linear(
            encoder_dim + decoder_dim * 2 + embedding_dim, output_dim
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, encoder_outputs, hidden, cell):
        input = input.unsqueeze(0)  # input: [batch size] -> [1, batch size]
        embedded = self.dropout(
            self.embedding(input)
        )  # embedded: [1, batch size, embedding dim]
        a = self.attention(encoder_outputs, hidden[-1])  # a: [batch size, num_layers]
        a = a.unsqueeze(1)  # a: [batch size, 1, num_layers]
        weighted = torch.bmm(
            a, encoder_outputs
        )  # weighted: [batch size, 1, encoder dim]
        weighted = weighted.permute(1, 0, 2)  # weighted: [1, batch size, encoder dim]
        rnn_input = torch.cat(
            (embedded, weighted), dim=2
        )  # rnn_input: [1, batch size, encoder dim + embedding dim]
        output, (hidden, cell) = self.rnn(
            rnn_input, (hidden, cell)
        )  # output: [1, batch size, decoder dim * 2]
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        prediction = self.fc_out(
            torch.cat((output, weighted, embedded), dim=1)
        )  # prediction: [batch size, output dim]
        return prediction, hidden, cell, a.squeeze(1)


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

        # Initialize cell state for LSTM
        hidden_dim = hidden.size(-1)
        hidden = hidden.unsqueeze(0).repeat(2, 1, 1)  # Bidirectional LSTM: 2 layers
        cell = torch.zeros(2, batch_size, hidden_dim).to(self.device)

        input = trg[0]

        for t in range(1, trg_length):
            output, hidden, cell, _ = self.decoder(input, encoder_outputs, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1

        return outputs


df_caption = pd.read_csv("../../dataset_cleaned.csv")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


en_nlp = spacy.load("en_core_web_sm")


def tokenize_commnet(example):
    en_tokens = [token.text for token in en_nlp.tokenizer(example)][:100]
    en_tokens = [token.lower() for token in en_tokens]
    en_tokens = ["<sos>"] + en_tokens + ["<eos>"]
    return en_tokens


df_caption["text"] = df_caption["text"].apply(tokenize_commnet)

min_freq = 2

unk_token = "<unk>"
pad_token = "<pad>"
sos_token = "<sos>"
eos_token = "<eos>"

special_tokens = [
    unk_token,
    pad_token,
    sos_token,
    eos_token,
]

vocab = vocab.build_vocab_from_iterator(
    df_caption["text"],
    min_freq=min_freq,
    specials=special_tokens,
)

unk_index = vocab[unk_token]
pad_index = vocab[pad_token]

vocab.set_default_index(unk_index)

output_dim = 3490
embedding_dim = 256
encoder_dim = 2048
decoder_dim = 512
attention_dim = 512
decoder_dropout = 0.25

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder(encoder_dim, decoder_dim)

attention = Attention(encoder_dim, decoder_dim, attention_dim)


decoder = Decoder(
    output_dim, embedding_dim, encoder_dim, decoder_dim, decoder_dropout, attention
)


model = Seq2Seq(encoder, decoder, device).to(device)
