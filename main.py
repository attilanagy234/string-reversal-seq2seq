import argparse
import logging
import sys
import torch
import json
import os

from models.modules.model import MySeq2Seq
from training import train, evaluate
from torch.utils import data
from utils.data_generator import ReversedStringData, collate_function
from torch.nn import functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-9s %(message)s'))

log = logging.getLogger(__name__)
log.addHandler(handler)
log.setLevel(logging.INFO)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=str)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--train_size', default=256*100, type=int)
    parser.add_argument('--eval_size', default=256*10, type=int)
    parser.add_argument('--cuda', dest='use_cuda', default=True, type=bool)
    parser.add_argument('--device', default=None, type=object)
    parser.add_argument('--batch_size', dest='batch_size', default=256, type=int)
    parser.add_argument('--emb_dim', default=7, type=int)
    parser.add_argument('--num_tokens', default=11, type=int)
    parser.add_argument('--encoder_hidden_dim', default=32, type=int)
    parser.add_argument('--decoder_hidden_dim', default=32, type=int)
    parser.add_argument('--attn_dim', default=32, type=int)
    parser.add_argument('--min_len', default=5, type=int)
    parser.add_argument('--max_len', default=15, type=int)
    opts = parser.parse_args()

    def get_data_loaders(train_dataset_size, test_dataset_size):
        train_dataset = ReversedStringData(train_dataset_size, opts.min_len, opts.max_len, opts.num_tokens)
        test_dataset = ReversedStringData(test_dataset_size, opts.min_len, opts.max_len, opts.num_tokens)
        train_loader = data.DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=False,
                                       collate_fn=collate_function, drop_last=True)
        test_loader = data.DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=False,
                                      collate_fn=collate_function, drop_last=True)

        return train_loader, test_loader


    cuda = opts.use_cuda
    if cuda:
        if not torch.cuda.is_available():
            log.info('Cuda is not available, using CPU instead')
            cuda = False
    if cuda:
        opts.device = torch.device("cuda:0")
        torch.cuda.set_device(opts.device)
    else:
        opts.device = torch.device("cpu")

    config_path = os.path.join("experiments", opts.config)

    if not os.path.exists(config_path):
        raise FileNotFoundError

    with open(config_path, "r") as f:
        config = json.load(f)

    train_loader, test_loader = get_data_loaders(opts.train_size, opts.eval_size)
    config["batch_size"] = opts.batch_size

    model = MySeq2Seq(num_tokens=opts.num_tokens,
                      emb_dim=opts.emb_dim,
                      encoder_hidden_dim=opts.encoder_hidden_dim,
                      decoder_hidden_dim=opts.decoder_hidden_dim,
                      attn_dim=opts.attn_dim,
                      opts=opts)

    loss = nn.NLLLoss(ignore_index=2)
    writer = SummaryWriter()

    log.info(model)

    model = model.to(opts.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.007)

    for k, v in sorted(config.items(), key=lambda i: i[0]):
        print(" (" + k + ") : " + str(v))
    print()
    print("=" * 60)

    print("\nInitializing weights...")
    for name, param in model.named_parameters():
        if 'bias' in name:
            torch.nn.init.constant_(param, 0.0)
        elif 'weight' in name:
            torch.nn.init.xavier_normal_(param) # TODO gain

    for epoch in range(opts.epochs):
        run_state = (epoch, opts.epochs, opts.train_size)

        model, optimizer = train(model, optimizer, train_loader, run_state, loss_func=loss, writer=writer, epoch_num=epoch, opts=opts)
        evaluate(model, test_loader, loss_func=loss, opts=opts)

        torch.save({
            'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            'opts': opts
        }, f"saved_models/model_{epoch}.pth")


