import json
import math

from pathlib import Path

import plotly.graph_objects as go
from pymoo.indicators.hv import HV
from matplotlib.patches import Rectangle

import csv
import re
import os.path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import os.path
import zipfile
import tempfile
import shutil

from ILP_CC_reducer.model.ILPmodel import GeneralILPmodel
model = GeneralILPmodel(active_objectives=["extractions", "cc", "loc"])



def generate_2d_pf_plot(results_path, output_pdf_path):
    df = pd.read_csv(results_path)

    # Validar que hay datos y al menos 3 columnas numéricas
    if df.shape[0] == 0:
        print(f"No solutions found in {results_path}. 2DPF plot not generated.")
    else:
        numeric_columns = df.select_dtypes(include=[np.number])
        if numeric_columns.shape[1] < 2:
            print("It is not possible to represent 2DPF because there is less than 2 numeric columns.")
        else:
            # Seleccionar primeras 3 columnas numéricas
            objetivos = numeric_columns.iloc[:, :2].values
            nombres_objetivos = numeric_columns.columns[:2]

            objetivo1 = df.iloc[:, 0]  # Todas las filas, columna 0
            objetivo2 = df.iloc[:, 1]

            if objetivos.size == 0:
                print("Without numeric values.")
            else:

                o1 = np.array(objetivo1)
                o2 = np.array(objetivo2)

                sorted_indices = np.argsort(o1)
                o1 = o1[sorted_indices]
                o2 = o2[sorted_indices]

                # Crear la figura
                plt.figure(figsize=(8, 6))
                fig, ax = plt.subplots()

                plot_colors = {
                    'lavanda': "#9B8FC6",
                    'naranja': "#E07B39",
                    'verde': "#C1FFC1"
                }

                for i in range(len(o1) - 1):
                    # Línea horizontal hacia el siguiente x
                    plt.plot([o1[i], o1[i + 1]], [o2[i], o2[i]], color=plot_colors['lavanda'],
                             linewidth=2, zorder=2)
                    # Línea vertical hasta el siguiente y
                    plt.plot([o1[i + 1], o1[i + 1]], [o2[i], o2[i + 1]], color=plot_colors['naranja'],
                             linewidth=2, zorder=2)

                for i in range(len(o1)):
                    width = max(o1) + 1 - o1[i]
                    height = max(o2) + 1 - o2[i]
                    rect = Rectangle((o1[i], o2[i]), width, height, facecolor=plot_colors['verde'])
                    ax.add_patch(rect)


                plt.plot([o1[0], o1[0]], [max(o2)+1, o2[0]], color=plot_colors['naranja'],
                         linewidth=2, zorder=2)
                plt.plot([o1[-1], max(o1)+1], [o2[-1], o2[-1]], color=plot_colors['lavanda'],
                         linewidth=2, zorder=2)

                # Dibujar puntos
                plt.scatter(o1, o2, color='black', s=100, zorder=3)

                x_min, x_max = plt.xlim()
                y_min, y_max = plt.ylim()

                dx = (x_max - x_min) * 0.04
                dy = (y_max - y_min) * 0.02

                for idx, (x, y) in enumerate(zip(o1, o2), start=1):
                    plt.text(x + dx, y + dy, f's{idx}', ha='center', fontsize=20, color='black', zorder=4)

                # Etiquetas y título
                objective_map = {
                    model.extractions_objective.__name__: r"$EXTRACTIONS$",
                    model.cc_difference_objective.__name__: r"$CC_{diff}$",
                    model.loc_difference_objective.__name__: r"$LOC_{diff}$"
                }

                x_label = objective_map.get(nombres_objetivos[0])
                y_label = objective_map.get(nombres_objetivos[1])

                plt.tick_params(axis='both', which='major', labelsize=12)

                plt.xlabel(x_label, fontsize=14)
                plt.ylabel(y_label, fontsize=14)
                plt.grid(True, zorder=1)

                # Guardar como PDF
                plt.savefig(output_pdf_path, format='pdf')
                plt.close()
                print(f"2D PF plot saved in {output_pdf_path}.")


def generate_parallel_coordinates_plot(results_path, output_pdf_path):
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', 'D', '^', 'v', '*', 'x', '+', 'p', 'h', '1', '2', '3', '4', '|', '_']
    colors = plt.cm.tab20.colors  # hasta 20 colores distintos

    # Cargar el CSV
    df = pd.read_csv(results_path)

    if df.empty:
        print(f"No solutions found in {results_path}. Parallel coordinates plot not generated.")
    else:
        # We suppose that all columns are objectives
        objetivos_cols = df.columns

        objective_map = {
            model.extractions_objective.__name__: r"EXTRACTIONS",
            model.cc_difference_objective.__name__: r"CC$_{diff}$",
            model.loc_difference_objective.__name__: r"LOC$_{diff}$"
        }

        # Crear nombres s1, s2, ..., sN
        df['id'] = [f's{i+1}' for i in range(len(df))]

        # DataFrame para graficar
        df_plot = df[['id'] + list(objetivos_cols)]

        # Asegúrate de que las columnas sean numéricas
        for col in objetivos_cols:
            df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')

        x = range(len(objetivos_cols))

        for i, row in df.iterrows():
            style = line_styles[i % len(line_styles)]
            marker = markers[i % len(markers)]
            color = colors[i % len(colors)]
            y = [row[col] for col in objetivos_cols]
            plt.plot(x, y, label=row['id'], linestyle=style, marker=marker, color=color, alpha=0.8, linewidth=3)

        # Mapeo de nombres de objetivos a nombres legibles
        x_labels = [objective_map.get(col, col) for col in objetivos_cols]

        # Dibujo
        plt.xticks(ticks=x, labels=x_labels, fontsize=18)
        plt.ylabel(r'Objectives values', fontsize=18)
        plt.xlabel(r'Objectives to minimize', fontsize=18)
        plt.legend([], [], frameon=False)  # Quita la leyenda si hay muchas soluciones
        plt.legend(title=r'Solution', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.grid(True)
        plt.tight_layout()

        # Guardar como PDF
        plt.savefig(output_pdf_path, format='pdf')
        plt.close()
        print(f"Parallel coordinates plot saved in {output_pdf_path}.")


def generate_3d_pf_plot(results_path, output_html_path):

    df = pd.read_csv(results_path)

    # Validar que hay datos y al menos 3 columnas numéricas
    if df.shape[0] == 0:
        print(f"No solutions found in {results_path}. 3DPF plot not generated.")
    else:
        numeric_columns = df.select_dtypes(include=[np.number])
        if numeric_columns.shape[1] < 3:
            print("It is not possible to represent 3DPF plot because there is less than 3 numeric columns.")
        else:
            # Seleccionar primeras 3 columnas numéricas
            objetivos = numeric_columns.iloc[:, :3].values
            nombres_objetivos = numeric_columns.columns[:3]

            if objetivos.size == 0:
                print("Without numeric values.")
            else:
                # Calculate nadir directly from Pareto front
                nadir = np.max(objetivos, axis=0)
                ref_point = nadir + 1

                fig = go.Figure()

                n1, n2, n3 = ref_point

                solutions = []
                with open(results_path, newline='') as csvfile:
                    lector = csv.reader(csvfile)
                    next(lector)  # Saltar la cabecera
                    for row in lector:
                        sol = tuple(map(int, row))  # Convierte los strings a enteros
                        solutions.append(sol)

                parallel_face_colors = {
                    'top_bottom': "#E6E6FA",
                    'front_back': "#FFDAB9",
                    'left_right': "#C1FFC1"
                }

                # Caras agrupadas por paralelismo (cada par son dos triángulos)
                parallel_faces = [
                    # Inferior (z=c) y superior (z=10)
                    {
                        'faces': [(0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7)],
                        'color': parallel_face_colors['top_bottom']
                    },
                    # Frontal (y=b) y trasera (y=15)
                    {
                        'faces': [(0, 1, 5), (0, 5, 4), (2, 3, 7), (2, 7, 6)],
                        'color': parallel_face_colors['front_back']
                    },
                    # Derecha (x=20) e izquierda (x=a)
                    {
                        'faces': [(1, 2, 6), (1, 6, 5), (0, 3, 7), (0, 7, 4)],
                        'color': parallel_face_colors['left_right']
                    }
                ]

                for sol in solutions:
                    a, b, c = sol[0], sol[1], sol[2]

                    # Coordenadas de los 8 vértices del cubo
                    x = [a, n1, n1, a, a, n1, n1, a]
                    y = [b, b, n2, n2, b, b, n2, n2]
                    z = [c, c, c, c, n3, n3, n3, n3]

                    # Añadir una traza Mesh 3d por cada par de caras con mismo color
                    for group in parallel_faces:
                        color = group['color']
                        face_tris = group['faces']

                        i_vals = [f[0] for f in face_tris]
                        j_vals = [f[1] for f in face_tris]
                        k_vals = [f[2] for f in face_tris]

                        fig.add_trace(go.Mesh3d(
                            x=x, y=y, z=z,
                            i=i_vals, j=j_vals, k=k_vals,
                            color=color,
                            opacity=1,
                            flatshading=True,
                            showscale=False
                        ))

                # Dibujar puntos
                f1, f2, f3 = zip(*solutions)
                fig.add_trace(go.Scatter3d(
                    x=f1, y=f2, z=f3,
                    mode='markers+text',
                    marker=dict(size=10, color='black'),
                    text=[f's{idx+1}' for idx in range(len(solutions))],
                    textposition='top center',
                    textfont=dict(color='black', size = 18),
                    name='Solutions'
                ))

                objective_map = {
                    model.extractions_objective.__name__: r"EXTRACTIONS",
                    model.cc_difference_objective.__name__: r"CC<sub>diff</sub>",
                    model.loc_difference_objective.__name__: r"LOC<sub>diff</sub>"
                }

                x_label = objective_map.get(nombres_objetivos[0])
                y_label = objective_map.get(nombres_objetivos[1])
                z_label = objective_map.get(nombres_objetivos[2])

                fig.update_layout(scene=dict(xaxis=dict(title=dict(text=x_label, font=dict(size=25))),
                                             yaxis=dict(title=dict(text=y_label, font=dict(size=25))),
                                             zaxis=dict(title=dict(text=z_label, font=dict(size=25))),
                                             aspectmode='data'))
                fig.write_html(output_html_path)
                print(f"3D PF saved in {output_html_path}.")


def traverse_and_pf_plot(input_path, output_path):
    carpeta_graficas = os.path.join(output_path, "PF3D")
    os.makedirs(carpeta_graficas, exist_ok=True)

    for proyecto in os.listdir(input_path):
        ruta_proyecto = os.path.join(input_path, proyecto)
        if not os.path.isdir(ruta_proyecto) or proyecto == "PF3D":
            continue

        # Carpeta del proyecto dentro de GRÁFICAS
        carpeta_salida_proyecto = os.path.join(carpeta_graficas, f"{proyecto}_PF_3D")
        os.makedirs(carpeta_salida_proyecto, exist_ok=True)

        for class_method_folder in os.listdir(ruta_proyecto):
            if class_method_folder.startswith('HybridMethodForThreeObj'):
                method_path = os.path.join(ruta_proyecto, class_method_folder)
                if not os.path.isdir(method_path):
                    continue

                for archivo in os.listdir(method_path):
                    if archivo.endswith("_results.csv"):
                        ruta_csv = os.path.join(method_path, archivo)
                        salida_html = os.path.join(carpeta_salida_proyecto, f"{class_method_folder}_3DPF.html")
                        print(f"Generating 3D PF for: {ruta_csv}")
                        generate_3d_pf_plot(ruta_csv, salida_html)

            if class_method_folder.startswith('EpsilonConstraintAlgorithm'):
                continue


def traverse_and_plot(input_path: str, output_path: str):
    carpeta_graficas = os.path.join(output_path, "plots")
    os.makedirs(carpeta_graficas, exist_ok=True)

    objetivos_validos = {"extractions", "cc", "loc"}

    for project in os.listdir(input_path):
        ruta_proyecto = os.path.join(input_path, project)
        print(f"Going through project {project}.")
        if not os.path.isdir(ruta_proyecto) or project == "plots":
            continue

        # Project folder inside parallel_coordinates_plots
        carpeta_salida_proyecto = os.path.join(carpeta_graficas, f"{project}_plots")
        os.makedirs(carpeta_salida_proyecto, exist_ok=True)

        for class_method_folder in os.listdir(ruta_proyecto):
            print(f"Going through solution {class_method_folder}.")
            method_path = os.path.join(ruta_proyecto, class_method_folder)
            if not os.path.isdir(method_path):
                continue

            # Extract objectives from folder name
            partes = class_method_folder.split("_")
            if len(partes) < 2:
                continue  # If folder name does not start at least with "Algorithm_objectives", continue

            objetivos = partes[1].split("-")
            # Make sure that they are valid objectives
            objetivos = [o for o in objetivos if o in objetivos_validos]

            for archivo in os.listdir(method_path):
                if archivo.endswith("_results.csv"):
                    ruta_csv = os.path.join(method_path, archivo)
                    if len(objetivos) == 2:
                        salida_pdf = os.path.join(
                            carpeta_salida_proyecto, f"{class_method_folder}_2DPF_plot.pdf"
                        )
                        print(f"Generating 2D PF plot for: {ruta_csv}")
                        generate_2d_pf_plot(ruta_csv, salida_pdf)
                    elif len(objetivos) == 3:
                        salida_pdf = os.path.join(
                            carpeta_salida_proyecto, f"{class_method_folder}_parallel_coordinates_plot.pdf"
                        )
                        print(f"Generating parallel coordinates plot for: {ruta_csv}")
                        generate_parallel_coordinates_plot(ruta_csv, salida_pdf)

                        salida_html = os.path.join(carpeta_salida_proyecto, f"{class_method_folder}_3DPF.html")
                        print(f"Generating 3D PF for: {ruta_csv}")
                        generate_3d_pf_plot(ruta_csv, salida_html)


def generate_relative_hypervolume_plot(input_path, output_path):
    input_file = pd.read_csv(input_path)
    if not input_file.empty:
        abs_hv = input_file["absoluteHypervolume"].values
        solution_times = input_file["solutionObtainingTime"].values

        hv_max = np.max(abs_hv, axis=0)
        hv_relative = abs_hv / hv_max

        plt.figure(figsize=(8, 5))
        plt.plot(solution_times, hv_relative, marker='o', linestyle='-')
        plt.xlabel("Time (s)")
        plt.ylabel("Relative hypervolume")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path)


def generate_statistics_obj(
    results_path: str,
    complete_data_path: str,
    output_path: str,
    proyecto: str,
    class_method: str,
    num_obj: int
):
    """
    Calculates statistics for a CSV file with num_obj targets.
    Returns a dictionary with the results or None if an error occurs.
    """
    try:
        results_file = pd.read_csv(results_path)
        execution_time_average = ""
        total_execution_time = ""
        if complete_data_path:
            complete_data_file = pd.read_csv(complete_data_path)
            if not complete_data_file.empty:
                execution_time_average = complete_data_file["executionTime"].mean()

        if output_path:
            with open(output_path, "r") as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()  # última línea
                    # Extraemos el número usando regex
                    match = re.search(r"Total execution time:\s*([0-9.]+)", last_line)
                    if match:
                        time_value = float(match.group(1))
                        total_execution_time = time_value

        if results_file.empty:
            raise ValueError(f"Empty file: {results_path}. It is not possible to generate any result.")

        numeric_columns = results_file.select_dtypes(include=[np.number])
        if numeric_columns.shape[1] < num_obj:
            raise ValueError(f"Less than {num_obj} numeric columns.")

        objetivos = numeric_columns.iloc[:, :num_obj].values
        nombres_objetivos = numeric_columns.columns[:num_obj]

        nadir = np.max(objetivos, axis=0)
        ref_point = nadir + 1

        hv = HV(ref_point=ref_point)
        hipervolumen = hv.do(objetivos)

        ideal = np.min(objetivos, axis=0)
        hv_max = hv.do(ideal)
        hv_normalized = hipervolumen / hv_max

        algorithm = "Unknown"

        if "_" in class_method:
            clase, method = class_method.rsplit("_", 1)
            parts = clase.split("_", 2)

            algorithm = parts[0]  # antes del primer "_"
            clase = parts[2]
        else:
            clase, method = class_method, ""

        medias = np.round(np.mean(objetivos, axis=0), 2)
        standard_dev = np.round(np.std(objetivos, axis=0), 2)
        medianas = np.median(objetivos, axis=0)
        iqr = np.percentile(objetivos, 75, axis=0) - np.percentile(objetivos, 25, axis=0)

        res = {
            "project": proyecto,
            "class": clase,
            "method": method,
            "algorithm": algorithm,
            "num_solutions": objetivos.shape[0],
            "ideal": f"({', '.join(str(int(x)) for x in ideal)})",
            "nadir": f"({', '.join(str(int(x)) for x in ref_point)})",
            "hypervolume": hipervolumen,
            "normalized_hypervolume": np.round(hv_normalized, 2),
            "execution_time_average": execution_time_average,
            "total_execution_time": total_execution_time
        }

        for i, nombre in enumerate(nombres_objetivos):
            res[f"avg_{nombre}"] = medias[i]
            res[f"std_{nombre}"] = standard_dev[i]
            res[f"median_{nombre}"] = medianas[i]
            res[f"iqr_{nombre}"] = iqr[i]

        return res

    except Exception as e:
        return {"error": str(e)}


def generate_statistics(input_path: str, output_path: str):
    resultados_2obj = []
    resultados_3obj = []
    invalid_files = []

    for proyecto in os.listdir(input_path):
        ruta_proyecto = os.path.join(input_path, proyecto)
        if not os.path.isdir(ruta_proyecto):
            continue

        for class_method in os.listdir(ruta_proyecto):
            ruta_clase = os.path.join(ruta_proyecto, class_method)
            if not os.path.isdir(ruta_clase):
                continue

            results_file = next(
                (f for f in os.listdir(ruta_clase) if f.endswith("_results.csv")),
                None
            )
            if results_file is None:
                continue

            complete_data_path, execution_time_path = None, None

            complete_data_file = next(
                (f for f in os.listdir(ruta_clase) if f.endswith("_complete_data.csv")),
                None
            )

            execution_time_file = next(
                (f for f in os.listdir(ruta_clase) if f.endswith("_output.txt")),
                None
            )

            results_path = os.path.join(ruta_clase, results_file)

            if complete_data_file:
                complete_data_path = os.path.join(ruta_clase, complete_data_file)
            if execution_time_file:
                execution_time_path = os.path.join(ruta_clase, execution_time_file)

            if ('_extractions-cc_' in class_method
                  or '_cc-extractions_' in class_method
                  or '_loc-cc_' in class_method
                  or '_cc-loc_' in class_method
                  or '_extractions-loc_' in class_method
                  or '_loc-extractions_' in class_method):
                resultado = generate_statistics_obj(
                    results_path, complete_data_path, execution_time_path, proyecto, class_method, num_obj=2
                )
                if resultado is None or 'error' in resultado:
                    invalid_files.append({
                        "project": proyecto,
                        "class_method": class_method,
                        "archivo": results_path,
                        "error": resultado.get('error', 'Error desconocido')
                    })
                else:
                    resultados_2obj.append(resultado)

            elif ('extractions-cc-loc' in class_method
                  or 'extractions-loc-cc' in class_method
                  or 'loc-extractions-cc' in class_method
                  or 'cc-extractions-loc' in class_method
                  or 'cc-loc-extractions' in class_method
                  or 'loc-cc-extractions' in class_method):
                resultado = generate_statistics_obj(
                    results_path, complete_data_path, execution_time_path, proyecto, class_method, num_obj=3
                )
                if resultado is None or 'error' in resultado:
                    invalid_files.append({
                        "project": proyecto,
                        "class_method": class_method,
                        "archivo": results_path,
                        "error": resultado.get('error', 'Error desconocido')
                    })
                else:
                    resultados_3obj.append(resultado)

    # Guardar resultados
    if resultados_2obj:
        df_2obj = pd.DataFrame(resultados_2obj)
        ruta_2obj = os.path.join(output_path, "hipervolumen_2objs_summary.csv")
        df_2obj.to_csv(ruta_2obj, index=False)
        print(f"\n✅ Resumen 2 objetivos guardado en: {ruta_2obj}")

    if resultados_3obj:
        df_3obj = pd.DataFrame(resultados_3obj)
        ruta_3obj = os.path.join(output_path, "hipervolumen_3objs_summary.csv")
        df_3obj.to_csv(ruta_3obj, index=False)
        print(f"\n✅ Resumen 3 objetivos guardado en: {ruta_3obj}")

    if invalid_files:
        df_invalid = pd.DataFrame(invalid_files)
        ruta_invalid = os.path.join(output_path, "invalid_files.csv")
        df_invalid.to_csv(ruta_invalid, index=False)
        print(f"⚠️  Archivos con errores guardados en: {ruta_invalid}")


def analyze_model_data(method_path: Path, objectives: tuple):
    # Dictionary to store the three DataFrames
    dataframes = {}
    row_counts = {}

    # Get all CSV files in the folder
    csv_files = list(method_path.glob("*.csv"))

    # Read each CSV file and store it in the dictionary
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        # Extract the part after the last underscore and before ".csv"
        suffix = csv_file.stem.split("_")[-1]
        dataframes[suffix] = df  # Use the suffix as key

        # Count the number of rows (excluding header)
        row_counts[suffix] = len(df)

    # Add 1 if 'cc' in objectives and another 1 if 'loc' in objectives
    extra = 2 * sum(obj in objectives for obj in ("cc", "loc"))

    # Sum rows from specific CSVs and add the extra value
    keys_to_sum = ["sequences", "nested"]
    variables = sum(row_counts[k] for k in keys_to_sum if k in row_counts) + extra

    # Compute factor: 1 normally, 2 if 'cc' or 'loc' in objectives, 3 if both
    factor = 1 + 2 * sum(obj in objectives for obj in ("cc", "loc"))

    # Sum all constraints, applying factor only to 'sequences'
    constraints = sum(
        row_counts[k] * (factor if k == "sequences" else 1)
        for k in ("sequences", "nested", "conflict")
        if k in row_counts
    ) + 1  # +1 for x_0 == 1 constraint

    return variables, constraints




def generate_global_relative_hv_vs_time(
    results_root: str,
    output_dir: str | None = None,
    interpolation_points: int = 200
):
    """
    Generates ONE graph per algorithm (EpsilonConstraint / HybridMethod)
    and per number of objectives (2 and 3), averaging HV at each relative time point.

    Stores all processed data in a CSV maestro to avoid re-processing.
    """

    matplotlib.rcParams['font.family'] = 'Times New Roman'

    original_input_path = results_root
    processed_dir = os.path.join(os.path.dirname(os.path.abspath(original_input_path)), "processed_data")
    os.makedirs(processed_dir, exist_ok=True)
    csv_master_path = os.path.join(processed_dir, "all_relative_hv_data.csv")

    # Si existe CSV maestro, cargarlo y saltar procesamiento pesado
    if os.path.exists(csv_master_path):
        print(f"CSV maestro encontrado en {csv_master_path}. Cargando datos directamente...")
        all_data = pd.read_csv(csv_master_path)
        # Convertir t_rel y hv_rel de string a listas

        def ensure_list(x):
            if isinstance(x, str):
                return json.loads(x)
            return x

        all_data["t_rel"] = all_data["t_rel"].apply(ensure_list)
        all_data["hv_rel"] = all_data["hv_rel"].apply(ensure_list)

        skip_processing = True
    else:
        all_data = pd.DataFrame(columns=["project", "class", "method", "algorithm", "num_obj", "t_rel", "hv_rel"])
        skip_processing = False

    curves = {}

    temp_dir = None

    # ----------------------------------------------------------
    # Normalizar results_root
    # ----------------------------------------------------------
    if not skip_processing:
        if results_root.endswith(".zip"):
            print(f"Extracting zip {results_root}.")
            temp_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(results_root, 'r') as z:
                z.extractall(temp_dir)
            results_root = temp_dir

        def find_results_folder(root_path: str) -> str | None:
            for current_root, dirs, files in os.walk(root_path):
                for d in dirs:
                    if d.lower() == "results":
                        return os.path.join(current_root, d)
            return None

        if os.path.basename(os.path.normpath(results_root)).lower() != "results":
            found = find_results_folder(results_root)
            if found:
                print(f"Found results folder at {found}")
                results_root = found
            else:
                print("No 'results' folder found. Using original folder.")
                results_root = temp_dir if temp_dir else results_root

    # Output directory
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(original_input_path))
    os.makedirs(output_dir, exist_ok=True)

    # ---------- Browse through all folders ----------
    if not skip_processing:
        for project in os.listdir(results_root):
            project_path = os.path.join(results_root, project)
            if not os.path.isdir(project_path):
                continue
            print(f"Project path: {project_path}.")

            for solution_folder in os.listdir(project_path):
                solution_path = os.path.join(project_path, solution_folder)
                if not os.path.isdir(solution_path):
                    continue
                print(f"Solution path: {solution_path}.")

                # Parse algorithm
                if solution_folder.startswith("EpsilonConstraintAlgorithm"):
                    algorithm = "EpsilonConstraintAlgorithm"
                elif solution_folder.startswith("HybridMethodAlgorithm"):
                    algorithm = "HybridMethodAlgorithm"
                else:
                    continue

                # Parse class, method, num_obj
                try:
                    parts = solution_folder.split("_", 2)
                    objectives_str = parts[1]  # e.g., "extractions-cc-loc"
                    class_method_str = parts[2]  # e.g., "src.main.java.eu.activage.datalake.query.DataLakeClient.java_execute"
                    class_name, method_name = class_method_str.rsplit("_", 1)
                    num_obj = len(objectives_str.split("-"))
                except Exception:
                    continue

                if num_obj not in (2, 3):
                    continue

                # Process CSV files
                for file in os.listdir(solution_path):
                    if not file.endswith("_complete_data.csv"):
                        continue

                    csv_path = os.path.join(solution_path, file)
                    try:
                        df = pd.read_csv(csv_path)
                        if df.empty:
                            continue
                        if not {"absoluteHypervolume", "solutionObtainingTime"} <= set(df.columns):
                            continue

                        times = df["solutionObtainingTime"].values
                        hv_abs = df["absoluteHypervolume"].values
                        hv_max = hv_abs[-1]
                        if hv_max <= 0:
                            continue

                        # Relative HV
                        hv_rel = hv_abs / hv_max

                        # Relative time
                        duration = times[-1] - times[0]
                        if duration == 0:
                            t_rel = np.array([0, 1])
                            hv_rel = np.array([hv_rel[0], hv_rel[0]])
                        else:
                            t_rel = (times - times[0]) / duration

                        # Interpolation
                        t_interp = np.linspace(0, 1, interpolation_points)
                        hv_interp = np.interp(t_interp, t_rel, hv_rel)

                        # Append to curves for plotting
                        curves.setdefault((algorithm, num_obj), []).append(hv_interp)

                        # Append to master DataFrame (una fila por método)
                        df_temp = pd.DataFrame({
                            "project": [project],
                            "class": [class_name],
                            "method": [method_name],
                            "algorithm": [algorithm],
                            "num_obj": [num_obj],
                            "t_rel": [json.dumps(t_interp.tolist())],
                            "hv_rel": [json.dumps(hv_interp.tolist())]
                        })
                        all_data = pd.concat([all_data, df_temp], ignore_index=True)

                    except Exception as e:
                        print(f"Error processing {csv_path}: {e}")
                print("-----------------------------------------------------------------------------------------------")

        # Guardar CSV maestro al final
        all_data.to_csv(csv_master_path, index=False)
        print(f"CSV maestro guardado en {csv_master_path}.")

    # ---------- Rebuild curves from all_data ----------
    if skip_processing:
        curves = {}
        for (alg, n_obj), group in all_data.groupby(["algorithm", "num_obj"]):

            def ensure_list(x):
                if isinstance(x, str):
                    import json
                    return json.loads(x)
                return x

            hv_list = [ensure_list(row) for row in group["hv_rel"]]

            curves[(alg, n_obj)] = hv_list

    colors = {
        "EpsilonConstraintAlgorithm": "#1f77b4",
        "HybridMethodAlgorithm": "#ff7f0e"
    }

    project_labels = {
        "bytecode-viewer": "Bytecode-Viewer",
        "cybercaptor-server": "Cybercaptor-Server",
        "fastjson": "FastJson",
        "fiware-commons": "Fiware-Commons",
        "iotbroker": "IOTBroker",
        "jedis": "Jedis",
        "jmetal": "JMetal",
        "knowage-core": "Knowage-core",
        "MOEAFramework": "MOEA-Framework",
        "queryexecution": "Query-Execution",
        "Ayesa_data": "Ayesa project"
    }

    algorithm_labels = {
        "HybridMethodAlgorithm": "Hybrid Method",
        "EpsilonConstraintAlgorithm": "AUGMECON"
    }

    linewidth = 2

    individual_plot_per_algorithm(curves, interpolation_points, colors, linewidth, output_dir)
    comparative_plot_per_algorithm(curves, interpolation_points, colors, linewidth,
                                   output_dir, algorithm_labels)
    generate_comparative_plots_per_project(all_data, output_dir, algorithm_labels, project_labels)
    generate_comparative_plot_Ayesa(all_data, output_dir, algorithm_labels)

    # Clean temporary folder if it was created
    if temp_dir is not None:
        shutil.rmtree(temp_dir)

    print(f"ALL relative HV plots correctly saved in {output_dir}.")


def individual_plot_per_algorithm(curves: dict, interpolation_points: int, colors: dict,
                                  linewidth: int, output_dir: str):
    # Individual plots per algorithm and number of objectives
    for (algorithm, num_obj), hv_list in curves.items():
        if not hv_list:
            continue
        plt.figure(figsize=(8, 5))
        hv_mean = np.mean(hv_list, axis=0)
        t_interp = np.linspace(0, 1, interpolation_points)

        ax = plt.gca()
        ax.set_facecolor('#f9f9f9')
        ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
        ax.plot(t_interp, hv_mean, color=colors.get(algorithm, 'black'), linewidth=linewidth, label=algorithm)
        ax.set_xlabel("Relative Time", fontsize=18)
        ax.set_ylabel("Average Relative HV", fontsize=18)
        ax.set_ylim(0, 1.05)

        legend = ax.legend(frameon=True, facecolor='white', edgecolor='gray', fontsize=12)
        legend.get_frame().set_alpha(0.8)
        plt.tight_layout()
        filename = f"{algorithm}_{num_obj}obj_relative_hv_vs_time.pdf"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    print(f"Individual relative HV plots correctly saved in {output_dir}.")


def comparative_plot_per_algorithm(curves: dict, interpolation_points: int, colors: dict,
                                   linewidth: int, output_dir: str, algorithm_labels: dict):
    # Comparative plots per number of objectives
    for num_obj in (2, 3):
        has_data = any(curves.get((alg, num_obj)) for alg in ["EpsilonConstraintAlgorithm", "HybridMethodAlgorithm"])
        if not has_data:
            continue

        plt.figure(figsize=(8, 5))
        ax = plt.gca()
        ax.set_facecolor('#f9f9f9')
        ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

        for algorithm in ["EpsilonConstraintAlgorithm", "HybridMethodAlgorithm"]:
            hv_list_raw = curves.get((algorithm, num_obj), [])

            # Filtrar listas cuyo primer elemento sea 1
            hv_list = [hv for hv in hv_list_raw if len(hv) > 0 and hv[0] != 1]
            if not hv_list:
                continue
            hv_mean = np.mean(hv_list, axis=0)
            t_interp = np.linspace(0, 1, interpolation_points)
            ax.plot(
                t_interp,
                hv_mean,
                label=f"{algorithm_labels.get(algorithm, algorithm)}",
                color=colors.get(algorithm, 'black'),
                linewidth=linewidth
            )

        ax.set_xlabel("Relative Time", fontsize=12)
        ax.set_ylabel("Average Relative HV", fontsize=12)
        ax.set_ylim(0, 1.05)

        legend = ax.legend(frameon=True, facecolor='white', edgecolor='gray', fontsize=12)
        frame = legend.get_frame()
        frame.set_alpha(0.8)  # transparencia
        frame.set_linewidth(0.5)

        plt.tight_layout()
        filename = f"Comparison_algorithms_{num_obj}obj_relative_hv_vs_time.pdf"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    print(f"Comparative relative HV plots correctly saved in {output_dir}.")


def generate_comparative_plots_per_project(all_data, output_dir: str, algorithm_labels: dict, project_labels):
    projects = all_data["project"].unique()
    num_projects = len(projects)
    rows = 2
    cols = 5
    rows = math.ceil(num_projects / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(20, 3.5 * rows))
    axes = axes.flatten()

    handles_dict = {}

    for i, project in enumerate(projects):
        ax = axes[i]
        project_data = all_data[all_data["project"] == project]

        for num_obj in (2, 3):
            for algorithm in ["EpsilonConstraintAlgorithm", "HybridMethodAlgorithm"]:
                # seleccionar las filas que cumplan condiciones
                filtered_data = project_data[
                    (project_data["algorithm"] == algorithm) &
                    (project_data["num_obj"] == num_obj)
                    ]

                # construir lista de hv_rel filtrando listas cuyo primer elemento sea 1
                hv_list = [row for row in filtered_data["hv_rel"] if len(row) > 0 and row[0] != 1]
                if not hv_list:
                    continue
                hv_mean = np.mean(hv_list, axis=0)
                t_interp = project_data.iloc[0]["t_rel"]
                line, = ax.plot(t_interp, hv_mean, label=f"{algorithm_labels.get(algorithm, algorithm)}", linewidth=2)

                if algorithm not in handles_dict:
                    handles_dict[algorithm] = line

        ax.set_title(project_labels.get(project, project), fontsize=24)
        ax.set_facecolor('#f9f9f9')
        ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
        ax.set_xlabel("Relative Time", fontsize=20)
        ax.set_ylabel("Average Relative HV", fontsize=20)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis='y', labelsize=16)
        ax.tick_params(axis='x', labelsize=16)
        ax.grid(True, linestyle='--', alpha=0.3)

    # eliminar ejes sobrantes si hay menos proyectos que subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.subplots_adjust(bottom=0.12)

    legend = fig.legend(
        handles=handles_dict.values(),
        labels=[algorithm_labels.get(a, a) for a in handles_dict.keys()],
        loc='lower center',
        bbox_to_anchor=(0.5, 0.02),
        bbox_transform=fig.transFigure,
        ncol=len(handles_dict),
        frameon=True,
        facecolor='white',
        edgecolor='gray',
        fontsize=12
    )

    legend.get_frame().set_alpha(0.8)
    legend.get_frame().set_linewidth(0.5)

    plt.tight_layout(rect=[0, 0.09, 0.9, 1.1])
    plt.savefig(os.path.join(output_dir, "all_projects_HV_comparison.pdf"), bbox_inches='tight')
    plt.close()

    print(f"Comparative relative HV plots PER PROJECT correctly saved in {output_dir}.")


def generate_comparative_plot_Ayesa(all_data, output_dir: str, algorithm_labels: dict):
    project_name = "Ayesa_data"
    proj_data = all_data[all_data["project"] == project_name]

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    for algorithm in ["EpsilonConstraintAlgorithm", "HybridMethodAlgorithm"]:
        hv_list = [row for row in proj_data[proj_data["algorithm"] == algorithm]["hv_rel"]]
        if not hv_list:
            continue
        hv_mean = np.mean(hv_list, axis=0)
        t_interp = proj_data.iloc[0]["t_rel"]
        plt.plot(t_interp, hv_mean, label=f"{algorithm_labels.get(algorithm, algorithm)}", linewidth=2)

    ax.set_facecolor('#f9f9f9')
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Relative Time")
    ax.set_ylabel("Average Relative HV")
    ax.set_ylim(0, 1.05)

    legend = plt.legend(frameon=True, facecolor='white', edgecolor='gray', fontsize=12)
    legend.get_frame().set_alpha(0.8)
    legend.get_frame().set_linewidth(0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{project_name}_comparison.pdf"))
    plt.close()

    print(f"Comparative relative HV plots for AYESA correctly saved in {output_dir}.")