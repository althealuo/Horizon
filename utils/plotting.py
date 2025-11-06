import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ============================================================
#  Global Plotting Styles
# ============================================================

# --- Shared color palette (color-blind friendly & print-safe) ---
PLOT_COLORS = ["#ff7e73", "#6eb8f0",  "#a0a0a0", "#4daea1", "#e6a95b", "#b58fe6"]

# --- Global Seaborn Theme ---
sns.set_theme(style="white", font="serif", context="paper", font_scale=1.1)
sns.set_palette(sns.color_palette(PLOT_COLORS))

# --- Shared Figure Style Parameters ---
SPINE_COLOR = "#f0f0f0"
GRID_COLOR = "#f0f0f0"
LABEL_FONT = {'fontfamily': 'sans-serif', 'fontsize': 8}
TITLE_FONT = {'fontfamily': 'serif', 'fontsize': 10}
TICK_FONT_FAMILY = 'sans-serif'
TICK_FONT_SIZE = 8

# ============================================================
#  Plotting Functions
# ============================================================

def plot_test_loss(model_dict):
    """Plot test loss progression for each model."""
    fig, ax = plt.subplots(figsize=(6, 3.5))

    # --- Plot each model’s loss curve ---
    for model_name, data in model_dict.items():
        loss_prog = data["test_loss_prog"]
        final_epoch = data["final_epoch"]
        ax.plot(range(final_epoch), loss_prog, label=model_name, linewidth=1.2)

    # --- Labels and Legend ---
    ax.set_xlabel("Epoch", **LABEL_FONT)
    ax.set_ylabel("Test Loss", **LABEL_FONT)
    ax.legend(
        prop={'family': 'sans-serif', 'size': 9},
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15),
        ncol=3,
        handletextpad=0.5,
        columnspacing=1.0,
    )

    # --- Ticks ---
    for label in ax.get_yticklabels() + ax.get_xticklabels():
        label.set_fontfamily(TICK_FONT_FAMILY)
        label.set_fontsize(TICK_FONT_SIZE)

    # --- Box + Grid styling ---
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
        spine.set_linewidth(0.8)

    ax.grid(True, axis='x', color=GRID_COLOR, linestyle='-', linewidth=0.6)
    ax.grid(False, axis='y')

    plt.tight_layout()
    plt.show()

def plot_error_bars(model_dict, title=None, xlabel="Accuracy", col1="test_acc_h1_prog", col2="test_acc_h6_prog"):
    """Plot mean ± std accuracy for H1 vs H6 across models."""
    h1_means, h6_means = [], []
    h1_stds, h6_stds = [], []

    # --- Aggregate data ---
    for model_name, data in model_dict.items():
        acc_h1 = np.array(data[col1])
        acc_h6 = np.array(data[col2])
        h1_means.append(acc_h1.mean())
        h6_means.append(acc_h6.mean())
        h1_stds.append(acc_h1.std())
        h6_stds.append(acc_h6.std())

    names = list(model_dict.keys())
    y_pos = np.arange(len(names))

    # --- Figure setup ---
    fig, ax = plt.subplots(figsize=(6, 3.5))

    # --- Errorbar plots ---
    ax.errorbar(h1_means, y_pos - 0.1, xerr=h1_stds, fmt='o',
                color=PLOT_COLORS[0], label="H1", capsize=3, linewidth=1.0)
    ax.errorbar(h6_means, y_pos + 0.1, xerr=h6_stds, fmt='o',
                color=PLOT_COLORS[1], label="H6", capsize=3, linewidth=1.0)

    # --- Labels and Legend ---
    ax.set_title(f"{title}", **TITLE_FONT)
    # ax.set_ylabel("Models", **LABEL_FONT)
    ax.set_xlabel(xlabel, **LABEL_FONT)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.legend(
        prop={'family': 'sans-serif', 'size': 7},
        loc='lower right',          # anchor the legend’s lower-right corner
        bbox_to_anchor=(1.0, 1.0), # place just above the top-right corner
        ncol=2,
        handletextpad=0.5,
        columnspacing=1.0
    )

    ax.invert_yaxis()

    # --- Sans-serif tick labels ---
    for label in ax.get_yticklabels() + ax.get_xticklabels():
        label.set_fontfamily(TICK_FONT_FAMILY)
        label.set_fontsize(TICK_FONT_SIZE)

    # --- Box + Grid styling ---
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
        spine.set_linewidth(0.8)

    ax.grid(True, axis='x', color=GRID_COLOR, linestyle='-', linewidth=0.6)
    ax.grid(False, axis='y')

    plt.tight_layout()
    plt.show()


def plot_h6_stepwise(history, model_name="Seq2SeqGRU"):
    """
    Plot stepwise accuracy for H6 (c4–c9) vs overall H1 accuracy
    in the same visual style as other paper-quality plots.
    """

    # --- Data prep ---
    final_acc_h6 = history["h6_step_acc"][-1]   # e.g., [acc_c4,...,acc_c9]
    final_acc_h1 = history["h1_acc"][-1]        # single value
    trials = [f"c{i}" for i in range(4, 10)]

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(6, 3.5))

    # H6 line
    ax.plot(trials, final_acc_h6, marker="o", linestyle="-",
            color=PLOT_COLORS[0], label="H6", linewidth=1.2, markersize=4)

    # H1 reference line
    ax.axhline(y=final_acc_h1, color=PLOT_COLORS[1],
               linestyle="--", label="H1", linewidth=1.0)

    # --- Labels and Title ---
    ax.set_xlabel("Game Trial", fontfamily="sans-serif", fontsize=8)
    ax.set_ylabel("Accuracy", fontfamily="sans-serif", fontsize=8)
    ax.set_title(f"H6 trial progression", fontfamily="serif", fontsize=11, pad=12)

    # --- Legend (top-right, above plot box) ---
    ax.legend(
        prop={'family': 'sans-serif', 'size': 7},
        loc='lower right',          # anchor the legend’s lower-right corner
        bbox_to_anchor=(1.0, 1.05), # place just above the top-right corner
        ncol=2,
        handletextpad=0.5,
        columnspacing=1.0
    )

    # --- Ticks ---
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("sans-serif")
        label.set_fontsize(8)

    # --- Styling ---
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
        spine.set_linewidth(0.8)

    ax.grid(True, axis="y", color=GRID_COLOR, linestyle="-", linewidth=0.6)
    ax.grid(False, axis="x")

    ax.set_ylim(0.7, 0.85)

    plt.tight_layout()
    plt.show()
