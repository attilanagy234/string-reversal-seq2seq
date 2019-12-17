import torch
import random
from torch import nn
from torch.nn import functional as F
from models.modules.attention import BahdanauAttention


class Encoder(nn.Module):

    def __init__(self, opts, num_tokens, emb_dim, encoder_hidden_dim):
        super().__init__()
        self.opts = opts
        self.hidden_size = encoder_hidden_dim
        self.embedding = nn.Embedding(embedding_dim=emb_dim, num_embeddings=num_tokens)
        self.rnn = nn.LSTM(hidden_size=encoder_hidden_dim // 2,
                           bidirectional=True,
                           input_size=emb_dim,
                           batch_first=True)

    def forward(self, input):
        input = self.embedding(input)

        # TODO: multilayer
        init_hidden = (torch.zeros((2, input.shape[0], self.hidden_size // 2)).to(self.opts.device),
                       torch.zeros((2, input.shape[0], self.hidden_size // 2)).to(self.opts.device))
        output, _ = self.rnn(input, init_hidden)

        return output


class Decoder(nn.Module):
    def __init__(self, opts, num_tokens, emb_dim, encoder_dim, decoder_hidden_dim, attn_dim):
        super().__init__()
        self.opts = opts
        self.num_tokens = num_tokens
        self.encoder_dim = encoder_dim
        self.rnn = nn.LSTM(hidden_size=decoder_hidden_dim, input_size=encoder_dim + emb_dim, batch_first=True)
        self.attention = BahdanauAttention(attn_dim, key_size=encoder_dim, query_size=encoder_dim)
        self.embedding_encoder = nn.Embedding(num_tokens, emb_dim)
        self.embedding_decoder = nn.Linear(decoder_hidden_dim, num_tokens)

        self.SOS_ID = 0  # TODO
        self.EOS_ID = 1  # TODO

    def forward(self, input, input_lengths, target, max_steps, teacher_forcing_ratio):
        mask = get_mask(input_lengths).to(self.opts.device)
        batch_size, seq_len, features_num = input.shape
        prev_token = torch.tensor([self.SOS_ID] * batch_size, device=input.device).to(self.opts.device)
        hidden = (torch.zeros((1, batch_size, self.encoder_dim), device=input.device).to(self.opts.device),
                  torch.zeros((1, batch_size, self.encoder_dim), device=input.device).to(self.opts.device))

        key = self.attention.key_layer(input)

        # Decoding
        prev_tokens = []
        attentions = []
        for s in range(max_steps):
            prev_token, hidden, attention = self.step(input, mask, key, hidden, prev_token)
            prev_tokens.append(prev_token)

            attentions.append(attention.detach().cpu().squeeze(0).squeeze(0))

            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
            if use_teacher_forcing and target is not None:
                prev_token = target[:, s]
            else:
                prev_token = prev_token.detach().argmax(1)

            if batch_size == 1 and target is None and prev_token[0] == self.EOS_ID:
                break

        return torch.stack(prev_tokens, 1), torch.stack(attentions, 1)

    def step(self, input, mask, key, hidden, prev_token):
        context, attention = self.attention(query=hidden[0], proj_key=key, value=input, mask=mask)

        embs = self.embedding_encoder(prev_token)
        context = context.squeeze(1)
        combined = torch.cat((context, embs), 1).unsqueeze(1)

        output, hidden = self.rnn(combined, hidden)

        # output shape bs x seq_len x emb_dim
        output = self.embedding_decoder(output).squeeze(1)
        output = F.log_softmax(output, dim=-1)
        return output, hidden, attention


def get_mask(lengths):
    max_len = lengths.max()
    mask = torch.zeros((lengths.shape[0], max_len))
    for i, l in enumerate(lengths):
        mask[i, l:] = 1
    return mask


class MySeq2Seq(nn.Module):
    def __init__(self, num_tokens, emb_dim, encoder_hidden_dim, decoder_hidden_dim, attn_dim, opts):
        super().__init__()
        self.opts = opts
        self.encoder = Encoder(opts=opts, num_tokens=num_tokens, emb_dim=emb_dim, encoder_hidden_dim=encoder_hidden_dim)
        self.decoder = Decoder(opts=opts, num_tokens=num_tokens, emb_dim=emb_dim, encoder_dim=encoder_hidden_dim,
                               decoder_hidden_dim=decoder_hidden_dim, attn_dim=attn_dim)

    def forward(self, input, input_lengths, targets=None, max_length=256, teacher_forcing_ratio=1.):
        encoded = self.encoder(input)
        output, attention = self.decoder(encoded, input_lengths, targets, max_length, teacher_forcing_ratio)
        return output, attention
