#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate: Baseline Model Performance Comparison Radar Chart (English Version)
- Labels positioned TIGHTLY against vertices (minimal offset)
- Red text and Red borders for main model scores
- Directional adjustments tuned for closeness
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

    # Data
    data = {
        'MG-GAT-BiLSTM': [0.9974, 0.9867, 0.9920, 0.9995, 0.9998],  # Main model
        'MLP': [0.9912, 0.9878, 0.9894, 0.9971, 0.9996],
        'BiLSTM': [0.9886, 0.9855, 0.9868, 0.9964, 0.9993],
        'CNN-LSTM': [0.9881, 0.9851, 0.9861, 0.9965, 0.9991],
        'GRU': [0.9882, 0.9853, 0.9868, 0.9962, 0.9994],
        'Transformer': [0.9883, 0.9852, 0.9863, 0.9966, 0.9995],
    }

    # Colors
    colors = {
        'MG-GAT-BiLSTM': '#E63946',  # Red
        'MLP': '#0072B2',
        'BiLSTM': '#F0E442',
        'CNN-LSTM': '#009E73',
        'GRU': '#CC79A7',
        'Transformer': '#56B4E9',
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
        label.set_fontsize(22)
        label.set_weight('bold')
        label.set_color('#34495E')

    # Radius and grid (adjusted for dataset range: 98% to 100%)
    # Slightly higher upper limit to ensure top dot doesn't get clipped
    ax.set_ylim(0.98, 1.0002)
    yticks = [0.98, 0.99, 1.0]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{y * 100:.0f}%' for y in yticks],
                       fontsize=16, color='#7F8C8D', weight='bold')
    ax.grid(True, linestyle='--', alpha=0.35, linewidth=1.1, color='#B0BEC5')

    # Concentric light rings
    theta_full = np.linspace(0, 2 * np.pi, 360)
    bands = [(0.98, 0.99), (0.99, 1.0)]
    for i, (r0, r1) in enumerate(bands):
        if i % 2 == 0:
            ax.fill_between(theta_full, r0, r1, color='#F5F7FA', alpha=0.6, zorder=0)

    # Plot order
    plot_order = ['MLP', 'BiLSTM', 'GRU', 'CNN-LSTM', 'Transformer', 'MG-GAT-BiLSTM']

    for name in plot_order:
        vals = data[name]
        color = colors[name]
        marker = markers[name]

        is_main = (name == 'MG-GAT-BiLSTM')
        lw = 4.5 if is_main else 2.5
        a_line = 1.0 if is_main else 0.7
        a_fill = 0.15 if is_main else 0.05
        msize = 140 if is_main else 80
        z = 15 if is_main else 5

        ax.plot(theta, vals, linewidth=lw, color=color, label=name,
                alpha=a_line, zorder=z, solid_joinstyle='round', solid_capstyle='round')
        ax.fill(theta, vals, alpha=a_fill, color=color, zorder=z - 1)
        ax.scatter(theta, vals, s=msize, color=color, marker=marker,
                   zorder=z + 1, edgecolors='white', linewidth=2.2, alpha=a_line)

        # === Special Labeling for Main Model (Red Box, Red Text, TIGHT Position) ===
        if is_main:
            category_names = ['Precision', 'Recall', 'F1-Score', 'TNR', 'ROC AUC']

            for i, (ang, v, cat_name) in enumerate(zip(theta, vals, category_names)):

                # --- Revised Position Adjustment Logic ---
                # Goal: Tighter positions, closer to the vertices.
                # Since the Y-axis scale is tiny (0.98-1.0), radial additions must be tiny.

                ang_offset = ang
                base_radius = v  # The exact data point radius

                if cat_name == 'Precision':
                    # Top: Just barely above the point.
                    # Reduced offset significantly from 0.0015 to 0.0003
                    label_radius = base_radius + 0.0003

                elif cat_name == 'Recall':
                    # Left: Shift angle slightly Counter-Clockwise
                    ang_offset = ang + np.radians(5)
                    # Reduced radial offset to keep it tight
                    label_radius = base_radius + 0.0002

                elif cat_name == 'ROC AUC':
                    # Right: Shift angle slightly Clockwise
                    # Reduced angle shift to 5 degrees
                    ang_offset = ang - np.radians(5)
                    # Reduced radial offset significantly to keep it tight
                    label_radius = base_radius + 0.0002

                elif cat_name == 'F1-Score':
                    # Down-Left: Shift angle slightly towards bottom
                    ang_offset = ang + np.radians(5)
                    # Reduced radial offset significantly from 0.002 to 0.0005
                    label_radius = base_radius + 0.0005

                elif cat_name == 'TNR':
                    # Down-Right: Shift angle slightly towards bottom
                    # Reduced angle shift to 5 degrees
                    ang_offset = ang - np.radians(5)
                    # Reduced radial offset significantly from 0.002 to 0.0005
                    label_radius = base_radius + 0.0005

                # Draw the Label
                ax.text(ang_offset, label_radius, f'{v:.2%}',
                        ha='center', va='center',
                        fontsize=20,  # Large font
                        color='red',  # Red Text
                        weight='bold',
                        bbox=dict(
                            boxstyle='round,pad=0.2',
                            facecolor='white',
                            edgecolor='red',  # Red Border
                            linewidth=2.0,
                            alpha=0.95
                        ),
                        zorder=30)  # Highest z-order

    # Legend
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
            t.set_fontsize(16)
    leg.get_title().set_weight('bold')

    # Export
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