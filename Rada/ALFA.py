#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate: Baseline Model Performance Comparison Radar Chart (English Version)
- Using Okabe–Ito color palette (paper-friendly, colorblind-friendly)
- Non-main models with reduced fill alpha for clearer boundaries
- Main model MG-GAT-BiLSTM highlighted with percentage labels
- All text in English
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from pathlib import Path as PathLib
from datetime import datetime

# ============== Global Style ==============
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Get the directory where this script is located
SCRIPT_DIR = PathLib(__file__).parent
OUTPUT_DIR = SCRIPT_DIR
OUTPUT_DIR.mkdir(exist_ok=True)


# ============== Radar Projection (Custom) ==============
def radar_factory(num_vars, frame='polygon'):
    """Create radar chart projection"""
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = 'radar'

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # 0° points to top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
            return lines

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars, radius=.5, edgecolor="k")
            else:
                raise ValueError(f"Unknown frame: {frame}")

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                spine = Spine(axes=self, spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5) + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError(f"Unknown frame: {frame}")

    register_projection(RadarAxes)
    return theta


# ============== Generate Beautiful Radar Chart (English Version) ==============
def create_beautiful_radar():
    """Create beautiful radar chart (clear colors & low overlap) - English Version"""
    # Metrics (English)
    categories = [
        'Precision\n',
        'Recall\n',
        'F1-Score\n',
        'TNR\n',
        'ROC AUC\n'
    ]

    # Data from ALFA dataset performance comparison (flight_14)
    # Order: Precision, Recall, F1-Score, TNR, ROC AUC
    data = {
        'MG-GAT-BiLSTM': [0.9659, 0.9660, 0.9659, 0.9657, 0.9861],  # Best overall performance
        'MLP': [0.3500, 0.9545, 0.5122, 0.2941, 0.8229],
        'BiLSTM': [0.4895, 0.7955, 0.6061, 0.6697, 0.7887],
        'CNN-LSTM': [0.9241, 0.8295, 0.8743, 0.9155, 0.9515],
        'GRU': [0.5323, 0.7500, 0.6226, 0.7376, 0.7916],
        'Transformer': [0.9620, 0.8636, 0.9102, 0.9577, 0.9504],
    }

    # Okabe–Ito color palette (colorblind-friendly)
    colors = {
        'MG-GAT-BiLSTM': '#E63946',  # Highlight red: main model
        'MLP': '#0072B2',  # Blue
        'BiLSTM': '#F0E442',  # Yellow
        'CNN-LSTM': '#009E73',  # Blue-green
        'GRU': '#CC79A7',  # Magenta
        'Transformer': '#56B4E9',  # Sky blue
    }
    markers = {
        'MG-GAT-BiLSTM': 'o',
        'MLP': '^',
        'BiLSTM': 's',
        'CNN-LSTM': 'D',
        'GRU': 'p',
        'Transformer': '*',
    }

    N = len(categories)
    theta = radar_factory(N, frame='polygon')

    fig, ax = plt.subplots(figsize=(14, 12), subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(top=0.9, bottom=0.05, left=0.08, right=0.92)

    # Background & Title
    ax.set_facecolor('white')
    plt.title('',
              size=22, weight='bold', pad=40, color='#2C3E50')

    # Axis labels
    ax.set_varlabels(categories)
    for label in ax.get_xticklabels():
        label.set_fontsize(22)  # Increased font size for better visibility
        label.set_weight('bold')
        label.set_color('#34495E')

    # Radius and grid (adjusted for ALFA dataset range: 0.3 to 0.96)
    ax.set_ylim(0.0, 1.0)
    yticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{y * 100:.0f}%' for y in yticks],
                       fontsize=16, color='#7F8C8D', weight='bold')
    ax.grid(True, linestyle='--', alpha=0.35, linewidth=1.1, color='#B0BEC5')

    # Concentric light rings (hierarchy guide) - adjusted for 0-100% range
    theta_full = np.linspace(0, 2 * np.pi, 360)
    # Create alternating bands from 0% to 100%
    bands = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    for i, (r0, r1) in enumerate(bands):
        if i % 2 == 0:
            ax.fill_between(theta_full, r0, r1, color='#F5F7FA', alpha=0.6, zorder=0)

    # Plot order (weak to strong) - MG-GAT-BiLSTM last to highlight superiority
    plot_order = ['MLP', 'BiLSTM', 'GRU', 'CNN-LSTM', 'Transformer', 'MG-GAT-BiLSTM']
    for name in plot_order:
        vals = data[name]
        color = colors[name]
        marker = markers[name]

        is_main = (name == 'MG-GAT-BiLSTM')
        lw = 4.5 if is_main else 2.5  # Thicker line for main model
        a_line = 1.0 if is_main else 0.7  # More opaque for main model
        a_fill = 0.15 if is_main else 0.05  # More visible fill for main model
        msize = 140 if is_main else 80  # Larger markers for main model
        z = 15 if is_main else 5  # Higher z-order for main model

        ax.plot(theta, vals, linewidth=lw, color=color, label=name,
                alpha=a_line, zorder=z, solid_joinstyle='round', solid_capstyle='round')
        ax.fill(theta, vals, alpha=a_fill, color=color, zorder=z - 1)
        ax.scatter(theta, vals, s=msize, color=color, marker=marker,
                   zorder=z + 1, edgecolors='white', linewidth=2.2, alpha=a_line)

        if is_main:
            # Main model with percentage labels (adjusted for 98-100% range)
            # Category names for angle offset
            category_names = ['Precision', 'Recall', 'F1-Score', 'TNR', 'ROC AUC']
            for i, (ang, v, cat_name) in enumerate(zip(theta, vals, category_names)):
                # Apply angle offset for Recall, TNR, ROC AUC to avoid overlap with axis labels
                if cat_name in ['Recall', 'TNR', 'ROC AUC']:
                    # Small angular offset (±3 degrees)
                    if cat_name == 'Recall':
                        ang_offset = ang - np.radians(3)  # -3 degrees
                    elif cat_name == 'TNR':
                        ang_offset = ang + np.radians(3)  # +3 degrees
                    else:  # ROC AUC
                        ang_offset = ang - np.radians(3)  # -3 degrees
                else:
                    ang_offset = ang  # No offset for Precision and F1-Score

                # Label radius: v - 0.02 (move towards center, further from axis labels)
                label_y = max(v - 0.02, 0.0)  # Ensure it doesn't go below 0%

                # Use 4 decimal places for better precision display
                ax.text(ang_offset, label_y, f'{v:.4f}',
                        ha='center', va='center', fontsize=18, color=color, weight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color,
                                  linewidth=2.5, alpha=0.95),
                        zorder=25)  # Highest z-order to ensure labels are always on top

    # Legend: line + point style
    handles = []
    for n in ['MG-GAT-BiLSTM', 'MLP', 'BiLSTM', 'CNN-LSTM', 'GRU', 'Transformer']:
        h = plt.Line2D([0], [0],
                       color=colors[n], linewidth=3,
                       marker=markers[n], markersize=9,
                       markerfacecolor=colors[n],
                       markeredgecolor='white', markeredgewidth=1.6,
                       label=n)
        handles.append(h)

    leg = ax.legend(handles=handles, loc='upper right',
                    bbox_to_anchor=(1.28, 1.06), fontsize=16,
                    framealpha=0.95, edgecolor='#B0BEC5',
                    fancybox=True, shadow=True, title='', title_fontsize=17)

    for t in leg.get_texts():
        if t.get_text() == 'MG-GAT-BiLSTM':
            t.set_weight('bold')
            t.set_color(colors['MG-GAT-BiLSTM'])
            t.set_fontsize(17)
        else:
            t.set_fontsize(16)  # Ensure all legend text is larger
    leg.get_title().set_weight('bold')

    # Export (add timestamp to avoid overwriting)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_png = OUTPUT_DIR / f'baseline_model_performance_radar_chart_english_{timestamp}.png'
    plt.savefig(str(output_png), dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.3)
    print("[OK] Beautiful radar chart (English) generated:\n  -", output_png, "(300 DPI)")
    plt.close()


# ============== Entry Point ==============
if __name__ == '__main__':
    print("=" * 70)
    print("Generate Beautiful Radar Chart (English Version)")
    print("=" * 70)
    create_beautiful_radar()
    print("\n[Complete] Output location:", OUTPUT_DIR.absolute())
