import matplotlib.pyplot as plt
from recombinant_tree import RecombinantTree
import numpy as np


def plot_tree(data_tree: RecombinantTree, title="Binomial tree", val_color_map=None):
    """
    Plots a recombinant binomial tree with geometrically constant spacing 
    and displays the numerical values at each node.

    :param title: Tytuł wykresu
    :param data_tree: generated date tree from @generate_stock_tree function
    :return: tree plot
    """
    plt.figure(figsize=(14, 8))

    N = data_tree.N

    for t in range(N):
        for i in range(t + 1):
            x = t
            y = (t / 2) - i

            x_next = t + 1
            y_up = ((t + 1) / 2) - i
            y_down = ((t + 1) / 2) - (i + 1)

            plt.plot([x, x_next], [y, y_up], 'g-', alpha=0.3, linewidth=1)
            plt.plot([x, x_next], [y, y_down], 'r-', alpha=0.3, linewidth=1)

    for t in range(N + 1):
        for i in range(t + 1):
            x = t
            y = (t / 2) - i

            node_value = data_tree[i, t]

            plt.scatter(x, y, color='blue', s=40, zorder=5)

            if np.issubdtype(data_tree.dtype, np.floating):
                label_text = f"{node_value:.2f}"
            else:
                label_text = f"{node_value}"
            plt.text(x + 0.05, y + 0.05, label_text,
                     fontsize=9,
                     fontweight='bold',
                     ha='left',
                     va='bottom',
                     zorder=6,
                     color=val_color_map(node_value) if val_color_map else 'black')

    plt.title(title, fontsize=14)
    plt.xlabel("Time step", fontsize=12)

    plt.xticks(range(N + 1))
    plt.yticks([])

    plt.xlim(-0.5, N + 0.7)
    plt.ylim(-N/2 - 0.5, N/2 + 0.5)

    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()
