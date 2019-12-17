import matplotlib.pyplot as plt
from matplotlib import ticker


def show_attention(input_tokens, output_tokens, attentions, filename):
    # Set up figure with colorbar
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)

    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)
    # Set up axes
    # Weird matplotlib bug - not showing first word -> shift
    ax.set_xticklabels(["<SOS>"]+input_tokens)
    ax.set_yticklabels(["<SOS>"]+output_tokens)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig(filename)
    plt.show()
