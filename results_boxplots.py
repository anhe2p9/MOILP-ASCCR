import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib
import ast

matplotlib.rcParams['font.family'] = 'Times New Roman'

# Leer el Excel
archivo = "C:/Users/X1502/Adriana/LAB334/Eventos/Posters_session_MOILP_4feb/LOC_and_CC_final.xlsx"

cols_mono = ["extractions", "CCdiff", "LOCdiff"]
cols_multi = ["avg_extractions_objective", "avg_cc_difference_objective", "avg_loc_difference_objective"]
names_general = [r"Extractions", r"CC$_{diff}$", r"LOC$_{diff}$"]
y_labels = [r"Extractions per instance", r"CC$_{diff}$ per instance", r"LOC$_{diff}$ per instance"]

# Colores
color_mono = "#88d8b0"
color_multi = "#ffe599"
color_mediana = "#d62828"
color_lex = "#cdb4db"

# Leer datos
df_mono = pd.read_excel(archivo, sheet_name=0)[cols_mono]
df_multi = pd.read_excel(archivo, sheet_name=1)


# Convertir la columna de string a tupla
lex_tuples = df_multi["ideal"].dropna().apply(ast.literal_eval)
num_obj = len(lex_tuples.iloc[0])
lex_opt_lists = [lex_tuples.apply(lambda x: x[i]).tolist() for i in range(num_obj)]


# Figura y ejes principales
fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')

for i, (c_mono, c_multi, name) in enumerate(zip(cols_mono, cols_multi, names_general)):
    ax = axes[i]

    d_mono = df_mono[c_mono].dropna()
    d_multi = df_multi[c_multi].dropna()

    if c_mono == "LOCdiff":
        limite_loc = 300
        d_mono = d_mono.clip(upper=limite_loc)
        d_multi = d_multi.clip(upper=limite_loc)

    data = [d_mono, d_multi, pd.Series(lex_opt_lists[i])]

    box = ax.boxplot(
        data,
        positions=[1.6, 2.5, 3.2],
        widths=[1.0, 0.5, 0.5],
        patch_artist=True,
        whis=1.5,
        showfliers=False,
        medianprops=dict(color=color_mediana, linewidth=3, linestyle="dotted"),
        boxprops=dict(linewidth=1),
        whiskerprops=dict(color='black', linewidth=1),
        capprops=dict(color='black', linewidth=1)
    )

    # Colorear cajas
    box['boxes'][0].set_facecolor(color_mono)
    box['boxes'][1].set_facecolor(color_multi)
    box['boxes'][2].set_facecolor(color_lex)

    ax.set_facecolor('#f9f9f9')
    ax.set_title(name, fontsize=34, pad=12)
    ax.set_xlim(1, 3.5)
    ax.set_xticks([1.6, 2.85])
    ax.set_xticklabels(["ILP", "MO-ILP"], fontsize=26)
    ax.set_ylabel(y_labels[i], fontsize=28)
    ax.tick_params(axis='y', labelsize=26)
    for spine in ax.spines.values():
        spine.set_linewidth(1)

# Crear un eje invisible para la leyenda
ax_legend = fig.add_axes([0, 0, 1, 0.1], frameon=False)  # x0, y0, ancho, alto
ax_legend.axis('off')  # eje invisible

legend_elements = [
    Line2D([0], [0], color=color_multi, lw=10, label='Average'),
    Line2D([0], [0], color=color_lex, lw=10, label='Lex-opt')
]
ax_legend.legend(handles=legend_elements, loc='center', ncol=3, fontsize=24, frameon=True, bbox_to_anchor=(0.8, 0.5))

# Ajustar espacio de los subplots principales
fig.subplots_adjust(left=0.07, right=0.95, top=0.9, bottom=0.19, wspace=0.35)



plt.savefig(
    "boxplots_MOILP.png",
    dpi=1200,
    bbox_inches="tight"
)

plt.show()