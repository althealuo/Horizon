import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats # added for later use in error bars
from styles import THEME, apply_base_style, COLORS
from matplotlib.ticker import MaxNLocator

def plot_loss(model_dict, title="Train Loss Progression"):
    """
    Plot Train loss progression for each model using the style.py system.
    """
    # We increase the figure size slightly to give the larger fonts breathing room
    fig, ax = plt.subplots(figsize=(12, 7))

    # --- Plot each model’s loss curve ---
    for i, (model_name, data) in enumerate(model_dict.items()):
        loss_prog = data["test_loss_prog"]
        final_epoch = data["final_epoch"]
        
        # Use the updated color palette from styles.py
        color = COLORS[i % len(COLORS)]
        
        ax.plot(range(final_epoch), loss_prog, 
                label=model_name, 
                linewidth=2, 
                color=color)

    # --- Set Text Labels ---
    # Note: apply_base_style will handle the actual font family, size, and bolding
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Loss")

    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))

    # --- Setup Legend ---
    # We maintain the original compact layout, but the style function
    # will automatically scale the font to size 16 and remove the frame.
    ax.legend(
        frameon=True,
        loc='upper right',
        bbox_to_anchor=(1.0, 0.98),
        ncol=2,
        handletextpad=0.6,
        columnspacing=0.6,
        borderpad=0.6,
        labelspacing=0.6
    )

    # --- Apply Global Visual DNA ---
    # This single call replaces all the manual tick and spine loops
    apply_base_style(ax)

    # --- Grid Styling ---
    # We keep the grid for diagnostic plots, but use the theme's spine color
    ax.grid(True, axis='x', color=THEME['colors']['spine'], linestyle='-', linewidth=0.6, alpha=0.15)
    ax.grid(False, axis='y')

    plt.tight_layout()
    plt.show()

def plot_error_bars(model_dict, title=None, xlabel="Accuracy", col1="test_acc_h1_prog", col2="test_acc_h6_prog", baseline=None):
    """
    Plot mean ± std accuracy for H1 vs H6 across models with updated styles.
    """
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
    # Increased width to 12 to accommodate the larger fonts and labels
    fig, ax = plt.subplots(figsize=(12, 8))

    # --- Colors from THEME ---
    # Mapping H1 to your Red and H6 to your Blue from styles.py
    color_h1 = THEME['colors'][1]
    color_h6 = THEME['colors'][6]

    # --- Errorbar plots ---
    # H6 (Blue) and H1 (Red) with slightly thicker lines and larger markers
    ax.errorbar(h6_means, y_pos + 0.1, xerr=h6_stds, fmt='o',
                color=color_h6, label="H6", capsize=8, linewidth=3, markersize=12)

    ax.errorbar(h1_means, y_pos - 0.1, xerr=h1_stds, fmt='o',
                color=color_h1, label="H1", capsize=8, linewidth=3, markersize=12)

    # --- Baseline markers (Optional dashed lines) ---
    if baseline is not None and len(baseline) == 2:
        h1_base, h6_base = baseline
        ax.axvline(x=h1_base, color=color_h1, linestyle="--", alpha=0.6, linewidth=3)
        ax.axvline(x=h6_base, color=color_h6, linestyle="--", alpha=0.6, linewidth=3)
        
        # Baseline text labels
        ax.text(h1_base - 0.001, y_pos[-1] - 1.4, f"H1 baseline={h1_base:.2f}", 
                color=color_h1, fontsize=THEME['fonts']['label']['size'], rotation=270, 
                ha="right", va="bottom", alpha=0.8)
            
        ax.text(h6_base - 0.001, y_pos[-1] - 1.4, f"H6 baseline={h6_base:.2f}", 
                color=color_h6, fontsize=THEME['fonts']['label']['size'], rotation=270, 
                ha="right", va="bottom", alpha=0.8)

    # --- Labels and Title ---
    ax.set_title(title, pad=40)
    ax.set_xlabel(xlabel, labelpad=15)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)

    # --- Legend ---
    # Placed above the box for a clean, academic look
    ax.legend(loc='lower right', bbox_to_anchor=(1.0, 1.02), ncol=2,
        frameon=True, handletextpad=0.5, columnspacing=1.0, borderpad=0.6)

    ax.invert_yaxis()

    # --- Apply THEME Visual DNA ---
    # This automatically handles the Bold Title, Large Labels, and Dark Spines
    apply_base_style(ax)

    # --- Grid styling ---
    ax.grid(True, axis='x', color=THEME['colors']['spine'], linestyle='-', linewidth=0.6, alpha=0.2)
    ax.grid(False, axis='y')

    plt.tight_layout()
    plt.show()

def plot_h6_stepwise(history, model_name="Seq2SeqGRU"):
    """
    Plot stepwise accuracy for H6 vs H1 reference using updated styles.
    """
    # 1. Pull colors and line weights from THEME
    color_h6 = THEME['colors'][6] 
    color_h1 = THEME['colors'][1]
    line_width = THEME['lines']['model_width']
    
    # --- Data prep ---
    final_acc_h6 = history["h6_step_acc"][-1]   
    final_acc_h1 = history["h1_acc"][-1]        
    trials = [f"c{i}" for i in range(4, 10)]

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(12, 8))

    # H6 line (Blue) - Stepwise progression
    ax.plot(trials, final_acc_h6, marker="o", linestyle="-",
            color=color_h6, label="H6", linewidth=3, markersize=6)

    # H1 reference line (Red) - Flat baseline
    ax.axhline(y=final_acc_h1, color=color_h1,
               linestyle="--", label="H1", linewidth=3)

    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    # --- Labels and Title ---
    ax.set_title(f"H6 trial progression", pad=30)
    ax.set_xlabel("Game Trial", labelpad=10)
    ax.set_ylabel("Accuracy", labelpad=15)

    # --- Legend ---
    # Inherits borderpad=0.8 and frameon=True from styles.py rcParams
    ax.legend(loc='lower right', bbox_to_anchor=(1.0, 1.05), ncol=2)

    # --- Apply Global Visual DNA ---
    # Styles the semibold title, large labels, and the boxed legend border/font
    apply_base_style(ax)

    # --- Final Polish ---
    ax.set_ylim(0.7, 0.85)
    
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.grid(True, axis="y", color=THEME['colors']['spine'], linestyle="-", linewidth=0.6, alpha=0.15)

    plt.tight_layout()
    plt.show()