# utils/styles.py
import matplotlib.pyplot as plt

# Your updated palette
COLORS = ["#4b6bea", "#d43628", "#4daea1",  "#ff7e73",  "#85dd6f", "#e6a95b", "#fc8cf2","#9498a0"]
THEME = {
    'colors': {
        1: COLORS[0],       # Horizon 1 (Bold Red)
        6: COLORS[1],       # Horizon 6 (Bold Blue)
        'neutral': COLORS[2],
        'spine': "#404040",  # Darker gray for defined axis lines
    },
    'fonts': {
        'title':  {'family': 'sans-serif', 'size': 22},
        'label':  {'family': 'sans-serif', 'size': 20},
        'legend': {'family': 'sans-serif', 'size': 16},
        'tick':   {'family': 'sans-serif', 'size': 16},
    },
    'lines': {
        'model_width': 3.5, 
        'spine_width': 1.5,  # Thicker lines for better definition
        'spine_alpha': 1.0   # Fully opaque for darkness
    }
}

def apply_base_style(ax):
    """
    Standardizes the visual style with dark bottom/left axes, 
    bold titles, and large labels.
    """
    # 1. Darker Spine styling (Bottom and Left only)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for side in ['left', 'bottom']:
        ax.spines[side].set_color(THEME['colors']['spine'])
        ax.spines[side].set_linewidth(THEME['lines']['spine_width'])
        ax.spines[side].set_alpha(THEME['lines']['spine_alpha'])
    
    # 2. Force Ticks (The numbers)
    for label in ax.get_yticklabels() + ax.get_xticklabels():
        label.set_fontfamily(THEME['fonts']['tick']['family'])
        label.set_fontsize(THEME['fonts']['tick']['size'])

    # 3. Force Axis Labels (p(high info), etc.)
    ax.xaxis.label.set_fontfamily(THEME['fonts']['label']['family'])
    ax.xaxis.label.set_fontsize(THEME['fonts']['label']['size'])
    ax.yaxis.label.set_fontfamily(THEME['fonts']['label']['family'])
    ax.yaxis.label.set_fontsize(THEME['fonts']['label']['size'])

    # 4. Bold Font styling for the Title
    ax.title.set_fontfamily(THEME['fonts']['title']['family'])
    ax.title.set_fontsize(THEME['fonts']['title']['size'])
    ax.title.set_weight(600)

    # 5. Legend Styling
    leg = ax.get_legend()
    if leg:
        for text in leg.get_texts():
            text.set_fontfamily(THEME['fonts']['legend']['family'])
            text.set_fontsize(THEME['fonts']['legend']['size'])
        # leg.get_frame().set_linewidth(0)
        # leg.get_frame().set_alpha(0)

        frame = leg.get_frame()
        frame.set_linewidth(0.5)                      # Visible border
        frame.set_edgecolor(THEME['colors']['spine']) # Match the dark axis color
        frame.set_facecolor('white')                  # Solid background
        frame.set_alpha(0.5)                          # Slightly translucent
        
    return ax