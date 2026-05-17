import matplotlib.pyplot as plt


def plot_tree(data_tree, title="Drzewo dwumianowe"):
    """
    :param title: Tytuł wykresu
    :param data_tree: generated date tree from @generate_stock_tree function
    :return: tree plot
    """
    plt.figure(figsize=(14, 8))

    # TODO this will break sooner or later, create a tree class
    N = data_tree.shape[0] - 1
    for t in range(N):
        for i in range(t + 1):
            x = t
            y = data_tree[i, t]

            # Wyznaczamy ceny w następnym kroku
            # Ruch w górę wiersz pozostaje ten sam ruszamy się o jeden w góre
            y_up = data_tree[i, t + 1]
            # Ruch w dół oznacza zmiane wiesza i zmiane kolumny
            y_down = data_tree[i + 1, t + 1]

            # Rysowanie linii
            plt.plot([x, x + 1], [y, y_up], 'g-', alpha=0.3, linewidth=1)
            plt.plot([x, x + 1], [y, y_down], 'r-', alpha=0.3, linewidth=1)

    # Rysowanie węzłów
    for t in range(N + 1):
        for i in range(t + 1):
            x = t
            y = data_tree[i, t]
            plt.scatter(x, y, color='blue', s=30, zorder=5)

    plt.title(title, fontsize=14)
    plt.xlabel("Krok czasu (t)", fontsize=12)
    plt.ylabel("Cena instrumentu", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
