import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, mask, attention_mask=None, head_mask=None):

    matmul_qk = torch.matmul(q, k.permute(0,1,3,2))

    dk = k.shape[-1]
    scaled_attention_logits = matmul_qk / np.sqrt(dk)

    if mask is not None:
        nd, ns = scaled_attention_logits.size(-2), scaled_attention_logits.size(-1)
        scaled_attention_logits += (mask[ns-nd:ns, :ns] * -1e4)

    if attention_mask is not None:
        scaled_attention_logits = scaled_attention_logits + attention_mask

    attention_weights = torch.softmax(scaled_attention_logits, dim=-1)

    if head_mask is not None:
        attention_weights = attention_weights * head_mask

    output = torch.matmul(attention_weights, v)

    return output, attention_weights


class BahdanauAttention(nn.Module):

    def __init__(self, hidden_size, key_size=None, query_size=None):
        super(BahdanauAttention, self).__init__()

        # key_size is 2*hidden_size
        key_size = 2 * hidden_size if key_size is None else key_size
        query_size = hidden_size if query_size is None else query_size

        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)

        self.alphas = None

    def forward(self, query=None, proj_key=None, value=None, mask=None):
        assert mask is not None, "mask is required"


        query = self.query_layer(query)

        query = query.permute(1,0,2)

        scores = self.energy_layer(torch.tanh(query + proj_key))
        scores = scores.squeeze(2).unsqueeze(1)

        scores.data.masked_fill_(mask.unsqueeze(1) == 1, -float('inf'))

        alphas = F.softmax(scores, dim=-1)
        self.alphas = alphas

        context = torch.bmm(alphas, value)

        return context, alphas
