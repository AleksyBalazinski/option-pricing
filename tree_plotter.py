import matplotlib.pyplot as plt
from recombinant_tree import RecombinantTree
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from recombinant_tree import RecombinantTree


def plot_tree(data_tree: RecombinantTree, title="Binomial tree", val_color_map=None, show_labels=True):
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

            plt.scatter(x, y, color=val_color_map(node_value)
                        if val_color_map else 'black', s=40, zorder=5)

            if np.issubdtype(data_tree.dtype, np.floating):
                label_text = f"{node_value:.2f}"
            else:
                label_text = f"{node_value}"
            if show_labels:
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


def plot_american_option_exercise(
    stock_tree: RecombinantTree,
    exercise_tree: RecombinantTree,
    strike_price: float,
    title="American Put Optimal Exercise Boundary"
):
    """
    Plots the binomial tree using actual stock prices on the Y-axis with a log scale.
    Displays the Strike Price and dynamically calculates/plots the Optimal Exercise Boundary.

    :param stock_tree: RecombinantTree containing stock prices S_t at each node
    :param exercise_tree: RecombinantTree containing boolean/integer flags 
                          (e.g., 1 for Exercise/Green, 0 for Hold/Red)
    :param strike_price: The strike price (K) of the option
    :param title: Chart title
    """
    fig, ax = plt.subplots(figsize=(15, 9))
    N = stock_tree.N

    # 1. Colors & Design Constants (Softer, modern financial palette)
    color_hold = '#D9534F'      # Soft crimson red
    color_exercise = '#27AE60'  # Soft emerald green
    color_edge = '#7F8C8D'      # Muted grey for tree branches

    # Track boundary points for the line equation later: (time_step, boundary_price)
    boundary_x = []
    boundary_y = []

    # 2. Plot Branches (Lines connecting nodes)
    # Using true stock price coordinates
    for t in range(N):
        for i in range(t + 1):
            x = t
            y = stock_tree[i, t]

            x_next = t + 1
            y_up = stock_tree[i, t + 1]        # Up move
            y_down = stock_tree[i + 1, t + 1]    # Down move

            ax.plot([x, x_next], [y, y_up], color=color_edge,
                    alpha=0.15, linewidth=1, zorder=1)
            ax.plot([x, x_next], [y, y_down], color=color_edge,
                    alpha=0.15, linewidth=1, zorder=1)

    # 3. Plot Nodes & Track Exercise Boundary
    for t in range(N + 1):
        highest_exercise_s = None
        lowest_hold_s = None

        for i in range(t + 1):
            x = t
            s_val = stock_tree[i, t]
            # Assuming truthy value for green nodes
            is_exercised = exercise_tree[i, t]

            # Assign color based on early exercise state
            node_color = color_exercise if is_exercised else color_hold
            ax.scatter(x, s_val, color=node_color, s=35,
                       zorder=3, edgecolors='none')

            # Track thresholds to find the midpoint boundary
            if is_exercised:
                if highest_exercise_s is None or s_val > highest_exercise_s:
                    highest_exercise_s = s_val
            else:
                if lowest_hold_s is None or s_val < lowest_hold_s:
                    lowest_hold_s = s_val

        # Calculate optimal boundary midpoint for this time step
        if highest_exercise_s is not None and lowest_hold_s is not None:
            boundary_x.append(t)
            # Geometric mean handles log-scaled midpoints better financially
            boundary_y.append(np.sqrt(highest_exercise_s * lowest_hold_s))

    # 4. Plot Optimal Exercise Boundary Line
    if boundary_x:
        ax.plot(boundary_x, boundary_y, color='#2C3E50', linestyle='-', linewidth=2.5,
                label='Optimal Exercise Boundary ($S^*$)', zorder=4)

    # 5. Plot Strike Price Horizontal Line
    ax.axhline(y=strike_price, color='#2C3E50', linestyle='--', alpha=0.7, linewidth=1.5,
               label=f'Strike Price ($K = {strike_price}$)', zorder=2)

    # 6. Formatting Axes & Labels
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel("Time Step ($t$)", fontsize=12, labelpad=10)
    ax.set_ylabel("Stock Price ($S$)", fontsize=12, labelpad=10)

    # Use Log Scale so geometric spacing remains intact visually
    ax.set_yscale('log')

    # Clean formatting for Y axis ticks instead of scientific notation
    from matplotlib.ticker import ScalarFormatter
    ax.yaxis.set_major_formatter(ScalarFormatter())

    ax.set_xticks(range(N + 1))
    ax.set_xlim(-0.5, N + 0.5)

    # Clean styling: Remove top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, which="both", linestyle=':', alpha=0.3)

    # 7. Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Hold (No Exercise)',
               markerfacecolor=color_hold, markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Early Exercise',
               markerfacecolor=color_exercise, markersize=8),
        Line2D([0], [0], color='#2C3E50', linestyle='--',
               label=f'Strike Price ($K={strike_price}$)'),
        Line2D([0], [0], color='#2C3E50', linestyle='-',
               linewidth=2.5, label='Exercise Boundary ($S^*_t$)')
    ]
    ax.legend(handles=legend_elements, loc='upper left',
              frameon=True, facecolor='white', edgecolor='none')

    plt.tight_layout()
    plt.show()


def plot_european_call_tree(
    stock_tree: RecombinantTree,
    option_tree: RecombinantTree,
    strike_price: float,
    title="European Call Option Value Propagation"
):
    """
    Plots a European Call binomial tree using actual stock prices on the Y-axis (log scale).
    Applies a continuous color gradient to all nodes based on the option's value.

    :param stock_tree: RecombinantTree containing stock prices S_t at each node
    :param option_tree: RecombinantTree containing option values V_t at each node
    :param strike_price: The strike price (K) of the option
    :param title: Chart title
    """
    fig, ax = plt.subplots(figsize=(16, 9))
    N = stock_tree.N

    # 1. Colors & Colormap Setup
    # Using a sequential plasma or viridis map shows premium concentration beautifully.
    # 'YlOrRd' or 'magma' work wonders for financial "heat" maps.
    cmap_gradient = plt.cm.magma
    color_edge = '#7F8C8D'  # Muted grey for tree branches

    # Normalize option values across the entire tree for the color mapping
    max_option_val = np.max([option_tree[i, t]
                            for t in range(N+1) for i in range(t+1)])
    # Avoid division by zero if max value is 0
    from matplotlib.ticker import ScalarFormatter
    norm = Normalize(
        vmin=0, vmax=max_option_val if max_option_val > 0 else 1.0)

    # 2. Plot Tree Branches
    for t in range(N):
        for i in range(t + 1):
            x = t
            y = stock_tree[i, t]
            ax.plot([x, x + 1], [y, stock_tree[i, t + 1]],
                    color=color_edge, alpha=0.12, linewidth=1, zorder=1)
            ax.plot([x, x + 1], [y, stock_tree[i + 1, t + 1]],
                    color=color_edge, alpha=0.12, linewidth=1, zorder=1)

    # 3. Plot Nodes mapped to the Option Premium Gradient
    for t in range(N + 1):
        for i in range(t + 1):
            s_val = stock_tree[i, t]
            opt_val = option_tree[i, t]

            # Map the option value to the color scale
            node_color = cmap_gradient(norm(opt_val))

            # Draw the node
            ax.scatter(t, s_val, color=node_color, s=40,
                       zorder=3, edgecolors='none')

    # 4. Plot Strike Price Horizontal Line
    # For a Call, nodes ABOVE this line land In-The-Money at maturity
    ax.axhline(y=strike_price, color='#2C3E50', linestyle='--', alpha=0.8, linewidth=1.5,
               label=f'Strike Price ($K = {strike_price}$)', zorder=2)

    # 5. Formatting Axes, Labels & Scale
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel("Time Step ($t$)", fontsize=12, labelpad=10)
    ax.set_ylabel("Stock Price ($S$)", fontsize=12, labelpad=10)

    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ScalarFormatter())

    ax.set_xticks(range(N + 1))
    ax.set_xlim(-0.5, N + 0.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, which="both", linestyle=':', alpha=0.3)

    # 6. Add Colorbar Sidebar to quantify the gradient values
    sm = plt.cm.ScalarMappable(cmap=cmap_gradient, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, shrink=0.7)
    cbar.set_label("Call Option Value ($V$)", fontsize=11, labelpad=10)

    # 7. Legend
    legend_elements = [
        Line2D([0], [0], color='#2C3E50', linestyle='--',
               label=f'Strike Price ($K={strike_price}$)')
    ]
    ax.legend(handles=legend_elements, loc='upper left',
              frameon=True, facecolor='white', edgecolor='none')

    plt.tight_layout()
    plt.show()
