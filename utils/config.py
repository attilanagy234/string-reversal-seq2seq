import torch


class Config:
    def __init__(self):
        self.device = torch.device("cpu")
        torch.cuda.set_device(self.device)

    def set_cuda(self):
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)

