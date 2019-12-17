import logging
import torch
import sys

from models.modules.model import MySeq2Seq
from utils.attention_plot import show_attention
from utils.data_generator import ReversedStringData


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-9s %(message)s'))

log = logging.getLogger(__name__)
log.addHandler(handler)
log.setLevel(logging.INFO)


if __name__ == '__main__':
    model_path = sys.argv[1]

    checkpoint = torch.load(model_path)
    opts = checkpoint['opts']

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

    model = MySeq2Seq(num_tokens=opts.num_tokens,
                      emb_dim=opts.emb_dim,
                      encoder_hidden_dim=opts.encoder_hidden_dim,
                      decoder_hidden_dim=opts.decoder_hidden_dim,
                      attn_dim=opts.attn_dim,
                      opts=opts)
    model.load_state_dict(checkpoint['model_state_dict'])

    log.info(model)

    model = model.to(opts.device)
    model.eval()

    input_text = input()

    dataset = ReversedStringData(0, opts.min_len, opts.max_len, opts.num_tokens)
    input = torch.LongTensor(dataset.translate(input_text)).to(opts.device)
    input_length = torch.LongTensor([input.shape[0]])
    input = input.unsqueeze(0)

    preds, attention = model(input, input_length, None, max_length=input_length.max()*2, teacher_forcing_ratio=0)  # Output lenght can be max 2 times the input length
    logits = preds.argmax(2)
    logits = logits.squeeze(0)

    output_text = dataset.reverse_translate(logits)
    print(output_text)

    input_print = [dataset.reverse_translate([i]) for i in input.detach().cpu().tolist()[0]]
    output_print = [dataset.reverse_translate([i]) for i in logits.detach().cpu().tolist()]
    show_attention(input_print, output_print, attention.transpose(0, 1), "experiments/dummy.png")
