import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


title_text = "Performance comparison: DeepHyb vs HyDe"
subtitle_metrics = "Key classification metrics"
subtitle_types = "Accuracy across four-taxon combinations"
xlabel_types = "Four-taxon combinations"
ylabel_score = "Score"
legend_deephyd = "DeepHyb"
legend_hyde = "HyDe"


metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
deephyd_metrics = [0.8773, 0.5, 1.0, 0.6667]
hyde_metrics = [0.8501, 0.4501, 1.0, 0.6202]


types = [
    "O_S1_S1_S1", "O_S1_S1_S2", "O_S1_S1_H", "O_S1_S1_S3", "O_S1_S2_S2",
    "O_S1_S2_H", "O_S1_H_H", "O_S1_S2_S3", "O_S1_H_S3", "O_S1_S3_S3",
    "O_S2_S2_S2", "O_S2_S2_H", "O_S2_H_H", "O_H_H_H", "O_S2_S2_S3",
    "O_S2_H_S3", "O_H_H_S3", "O_S2_S3_S3", "O_H_S3_S3", "O_S3_S3_S3"
]
deephyd_types_acc = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
hyde_types_acc = [0.9920, 0.9649, 0.9667, 0.9667, 0.9660, 0.9633, 0.9638, 0.9626, 1.0, 0.9573, 0.9840, 0.9620, 0.9622, 0.9980, 0.9611, 0.0, 0.9622, 0.9602, 0.9684, 0.9880]


gap_scale = 0.9  #
font_scale = 1.0  
legend_y_pos = 0.95  


plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 8 * font_scale
plt.rcParams['axes.labelsize'] = 10 * font_scale
plt.rcParams['axes.titlesize'] = 12 * font_scale
plt.rcParams['legend.fontsize'] = 9 * font_scale
plt.rcParams['xtick.labelsize'] = 7 * font_scale
plt.rcParams['ytick.labelsize'] = 8 * font_scale


color_deephyd = '#2E86AB'
color_hyde = '#A23B72'
color_grid = '#E0E0E0'


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12 * gap_scale, 11 * gap_scale), gridspec_kw={'height_ratios': [1, 2.5]})


x_metrics = np.arange(len(metrics))
width = 0.35 * gap_scale
bars1 = ax1.bar(x_metrics - width/2, deephyd_metrics, width, label=legend_deephyd, color=color_deephyd, alpha=0.8, edgecolor='white', linewidth=1)
bars2 = ax1.bar(x_metrics + width/2, hyde_metrics, width, label=legend_hyde, color=color_hyde, alpha=0.8, edgecolor='white', linewidth=1)


ax1.set_title(subtitle_metrics, fontweight='bold', pad=10 * gap_scale)
ax1.set_ylabel(ylabel_score, fontweight='bold')
ax1.set_xticks(x_metrics)
ax1.set_xticklabels(metrics)
ax1.set_ylim(0, 1.1)
ax1.grid(axis='y', alpha=0.3, color=color_grid)
ax1.legend(loc=(1.02, legend_y_pos), frameon=True, fancybox=True, shadow=True, framealpha=0.8)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)


for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.4f}', ha='center', va='bottom', fontsize=7 * font_scale)
for bar in bars2:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.4f}', ha='center', va='bottom', fontsize=7 * font_scale)


x_types = np.arange(len(types))
line1 = ax2.plot(x_types, deephyd_types_acc, marker='o', linewidth=2.5 * gap_scale, markersize=4 * gap_scale, color=color_deephyd, label=legend_deephyd, alpha=0.9)
line2 = ax2.plot(x_types, hyde_types_acc, marker='s', linewidth=2.5 * gap_scale, markersize=4 * gap_scale, color=color_hyde, label=legend_hyde, alpha=0.9)


ax2.set_title(subtitle_types, fontweight='bold', pad=10 * gap_scale)
ax2.set_xlabel(xlabel_types, fontweight='bold', labelpad=15 * gap_scale)
ax2.set_ylabel(ylabel_score, fontweight='bold')
ax2.set_xticks(x_types)
ax2.set_xticklabels(types, rotation=45, ha='right', rotation_mode='anchor')
ax2.set_ylim(-0.05, 1.05)
ax2.grid(axis='y', alpha=0.3, color=color_grid)
ax2.legend(loc=(1.02, legend_y_pos), frameon=True, fancybox=True, shadow=True, framealpha=0.8)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)


fig.suptitle(title_text, fontweight='bold', fontsize=14 * font_scale, y=0.97)


plt.tight_layout()
plt.subplots_adjust(top=0.92, hspace=0.5 * gap_scale, bottom=0.12, right=0.85)


plt.savefig('Figure_3.svg', format='svg', dpi=300, bbox_inches='tight')