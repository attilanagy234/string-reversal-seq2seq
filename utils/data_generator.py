import numpy as np
import torch

from torch.utils import data
from random import choice, randrange
from itertools import zip_longest


def pad_tensor(vec, pad, value=0, dim=0):

    pad_size = pad - vec.shape[0]

    if len(vec.shape) == 2:
        zeros = torch.ones((pad_size, vec.shape[-1])) * value
    elif len(vec.shape) == 1:
        zeros = torch.ones((pad_size,)) * value
    else:
        raise NotImplementedError
    return torch.cat([torch.Tensor(vec), zeros], dim=dim)


def collate_function(batch, values=(0, 0), dim=0):

    sequence_lengths = torch.Tensor([int(x[0].shape[dim]) for x in batch])
    sequence_lengths, xids = sequence_lengths.sort(descending=True)
    target_lengths = torch.Tensor([int(x[1].shape[dim]) for x in batch])
    target_lengths, yids = target_lengths.sort(descending=True)
    src_max_len = max(map(lambda x: x[0].shape[dim], batch))
    tgt_max_len = max(map(lambda x: x[1].shape[dim], batch))
    batch = [(pad_tensor(x, pad=src_max_len, dim=dim, value=3),
              pad_tensor(y, pad=tgt_max_len, dim=dim, value=3))
             for (x, y) in batch]

    batch = torch.stack([torch.stack([torch.Tensor(b[0]).long(), torch.Tensor(b[1]).long()]) for b in batch])
    xs = batch[:,0]
    ys = batch[:,1]
    return xs, ys, sequence_lengths.int(), target_lengths.int()


class ReversedStringData(data.Dataset):

    def __init__(self, dataset_size, min_length, max_length, token_num):
        assert 3 < token_num <= 26+3

        self.SOS = "<s>"
        self.EOS = "</s>"
        self.PAD = "<pad>"
        self.characters = ''.join([chr(ord('a') + i) for i in range(token_num-3)])
        self.int2char = [self.SOS, self.EOS, self.PAD] + list(self.characters)
        self.char2int = {c: i+3 for i, c in enumerate(self.characters)}
        self.char2int[self.SOS] = 0
        self.char2int[self.EOS] = 1
        self.char2int[self.PAD] = 2
        print(self.char2int)
        self.VOCAB_SIZE = len(self.characters)
        self.min_length = min_length
        self.max_length = max_length

        self.set = [self._sample() for _ in range(dataset_size)]

    def __len__(self):
        return len(self.set)

    def __getitem__(self, item):
        return self.set[item]

    def _sample(self):
        random_length = randrange(self.min_length, self.max_length)

        random_char_list = [choice(self.characters) for _ in range(random_length)]
        random_string = ''.join(random_char_list)
        input = [self.char2int.get(x) for x in random_string]
        input = np.array([self.char2int[self.SOS]] + input + [self.char2int[self.EOS]])
        output = [self.char2int.get(x) for x in random_string[::-1]]
        output = np.array([self.char2int[self.SOS]] + output + [self.char2int[self.EOS]])

        return input, output

    def translate(self, input):
        text = ""
        for c in input:
            if c not in self.characters:
                print(f"[{c}] character was not represented in character set")
            else:
                text = text + c

        input = [self.char2int[c] for c in text]
        res = np.array([self.char2int[self.SOS]] + input + [self.char2int[self.EOS]])
        return res

    def reverse_translate(self, logits):
        output_text = ""
        for l in logits:
            output_text = output_text + self.int2char[l]
            if l == self.char2int[self.EOS]:
                break
        return output_text
