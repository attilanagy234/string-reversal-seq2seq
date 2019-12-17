import torch
import numpy as np
import matplotlib.pyplot as plt
import tqdm


def train(model, optimizer, train_loader, state, loss_func, writer, epoch_num, opts):
    epoch, n_epochs, train_steps = state

    losses = []

    # t = tqdm.tqdm(total=min(len(train_loader), train_steps))
    t = tqdm.tqdm(train_loader)
    model.train()

    for batch_count, batch in enumerate(t):
        t.set_description("Epoch {:.0f}/{:.0f} (train={})".format(epoch, n_epochs, model.training))
        inputs, targets, input_lengths, target_lengths = batch

        inputs = inputs.to(opts.device)
        targets = targets.to(opts.device)
        target_lengths = target_lengths.to(opts.device)

        max_len = target_lengths.max().item()
        preds, attention = model(inputs, input_lengths, targets, max_length=max_len, teacher_forcing_ratio=1.)
        loss = loss_func(preds.view(-1, preds.shape[2]), targets.reshape(-1))
        losses.append(loss.detach().cpu().item())

        writer.add_scalar('Loss/train', loss.item(), epoch_num*len(train_loader)+batch_count)

        optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=.5)
        optimizer.step()
        t.set_postfix(loss='{:05.3f}'.format(loss.item()), avg_loss='{:05.3f}'.format(np.mean(losses)))
        t.update()

    return model, optimizer


def evaluate(model, eval_loader, loss_func, opts):

    losses = []
    accs = []

    t = tqdm.tqdm(eval_loader)
    model.eval()

    with torch.no_grad():
        for batch in t:
            t.set_description(" Evaluating... (train={})".format(model.training))

            inputs, targets, input_lengths, target_lengths = batch

            inputs = inputs.to(opts.device)
            targets = targets.to(opts.device)
            target_lengths = target_lengths.to(opts.device)

            max_len = target_lengths.max().item()
            preds, _ = model(inputs, input_lengths, targets, max_length=max_len, teacher_forcing_ratio=1.)
            logits = preds.argmax(2)
            loss = loss_func(preds.view(-1, preds.shape[2]), targets.reshape(-1))
            acc = 100 * (logits == targets).detach().cpu().float().mean()
            losses.append(loss.detach().cpu().item())
            accs.append(acc)
            t.set_postfix(avg_acc='{:05.3f}'.format(np.mean(accs)), avg_loss='{:05.3f}'.format(np.mean(losses)))
            t.update()
        # align = alignments.detach().cpu().numpy()[:, :, 0]

    # Uncomment if you want to visualise weights
    # fig, ax = plt.subplots(1, 1)
    # ax.pcolormesh(align)
    # fig.savefig("data/att.png")
    print("  End of evaluation : loss {:05.3f} , acc {:03.1f}".format(np.mean(losses), np.mean(accs)))
    # return {'loss': np.mean(losses), 'cer': np.mean(accs)*100}


if __name__ == '__main__':
    train(1, 10)
