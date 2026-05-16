import numpy as np


class BinomialModel:
    def __init__(self, S0, N, dt, u, d, r):
        self.S0 = S0
        self.N = N
        self.dt = dt
        self.r = r
        self.u = u
        self.d = d
        self.p = (np.exp(r * dt) - self.d) / (self.u - self.d)

    def generate_stock_tree(self):
        tree = np.zeros(shape=(self.N + 1, self.N + 1))
        for t in range(self.N + 1):
            for i in range(t + 1):
                tree[i, t] = self.d**i * self.u**(t - i) * self.S0

        return tree

    def plot_tree(self, data_tree, title="Drzewo dwumianowe"):
        """
        :param title: Tytuł wykresu
        :param data_tree: generated date tree from @generate_stock_tree function
        :return: tree plot
        """
        plt.figure(figsize=(14, 8))

        for t in range(self.N):
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
        for t in range(self.N + 1):
            for i in range(t + 1):
                x = t
                y = data_tree[i, t]
                plt.scatter(x, y, color='blue', s=30, zorder=5)


        plt.title(title, fontsize=14)
        plt.xlabel("Krok czasu (t)", fontsize=12)
        plt.ylabel("Cena instrumentu", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()
