import csv
import ast
import os
import re
import textwrap
import tree_sitter_java as tsjava
from tree_sitter import Language, Parser
from concurrent.futures import ThreadPoolExecutor
import webbrowser
from pathlib import Path

import os
# Forzar a Matplotlib a usar un backend que no sea GUI (antes de que se cargue)
os.environ['MPLBACKEND'] = 'Agg'

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass

# Configuración inicial
JAVA_LANGUAGE = Language(tsjava.language())
parser = Parser(JAVA_LANGUAGE)

import pandas as pd
import numpy as np
import plotly.graph_objects as go

from plotly.subplots import make_subplots

from ILP_CC_reducer.model.ILPmodel import GeneralILPmodel
model = GeneralILPmodel(active_objectives=["extractions", "cc", "loc"])

from openai import OpenAI

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o"   # o "gpt-5.1"

client = OpenAI(api_key=API_KEY)


def generate_3d_pf_and_parallel_coordinates_plot(objectives_list, complete_data_file, output_html_path,
                                                 refact_cache_file, original_class_file, show=False):
    df = pd.read_csv(complete_data_file)

    if df.shape[0] == 0:
        print(f"No solutions found in {complete_data_file}. 3DPF plot not generated.")
        return

    if 'solution' not in df.columns:
        print("Error: No se encontró la columna 'solution' en el archivo CSV.")
        return

    df['solution_tuple'] = df['solution'].apply(ast.literal_eval)
    objetivos = np.array(df['solution_tuple'].to_list())

    if objetivos.shape[1] < 3:
        print(
            "It is not possible to represent 3DPF plot because there is less than 3 objectives in the solution tuple.")
        return

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'xy'}]],
        subplot_titles=('3D Pareto front', 'Parallel coordinates plot')
    )

    solutions = [tuple(sol) for sol in objetivos]

    # Recibimos los enlaces y las explicaciones
    enlaces_ordenados, explicaciones_ordenadas = create_refactoring_files(
        complete_data_file, output_html_path, solutions, refact_cache_file, original_class_file, objectives_list
    )

    # Enviamos las explicaciones a customize_3d_pf_plot
    fig = customize_3d_pf_plot(objetivos, solutions, fig, enlaces_ordenados, explicaciones_ordenadas)
    fig = customize_parallel_coordinates(solutions, fig, enlaces_ordenados)
    fig = customize_plotly_figures(objectives_list, fig)
    html_contenido = make_plotly_interactive(fig, output_html_path)

    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(html_contenido)
    print("[OK] Script de redirección web inyectado con éxito en el frente de Pareto.")

    # Abrir el frente de Pareto interactivo en el navegador solo si se pide con --show
    if show:
        webbrowser.open(Path(output_html_path).resolve().as_uri())


def create_refactoring_files(complete_data_file, output_html_path, solutions, refact_cache_file,
                             original_class_file, orden_objetivos):
    # 1. Detectamos la carpeta del archivo input y creamos la subcarpeta allí dentro
    carpeta_input = os.path.dirname(complete_data_file)
    folder_refactorizaciones = os.path.join(carpeta_input, "refactorizaciones_soluciones")

    os.makedirs(folder_refactorizaciones, exist_ok=True)
    print(f"\nGenerating refactoring files for each solution in parallel...")

    # [NUEVO] Preparación única de código, AST y cálculo de desfase global
    with open(original_class_file, 'r', encoding='utf-8', newline='') as f:
        codigo_str = f.read()
        codigo_bytes = codigo_str.encode('utf-8')

    mapa_offsets = crear_mapa_char_a_byte(codigo_str)
    tree = parser.parse(codigo_bytes)
    root_node = tree.root_node

    desfase_global, nodo_metodo_orig = calcular_metodo_y_desfase(
        complete_data_file, refact_cache_file, root_node, codigo_bytes, mapa_offsets
    )

    # print(f"DESFASE GLOBAL: {desfase_global}.")

    # 2. Generamos los enlaces calculando la ruta relativa respecto al HTML del gráfico
    carpeta_output = os.path.dirname(output_html_path)
    enlaces_ordenados = []
    for idx in range(len(solutions)):
        ruta_real_solucion = os.path.join(folder_refactorizaciones, f"solucion_{idx + 1}.html")
        ruta_relativa_web = os.path.relpath(ruta_real_solucion, carpeta_output)
        enlaces_ordenados.append(ruta_relativa_web)

    # 3. Función auxiliar que ejecutará cada hilo (ahora recibe las variables compartidas)
    def procesar_hilo(idx):
        ruta_html_sol = os.path.join(folder_refactorizaciones, f"solucion_{idx + 1}.html")
        try:
            # Pasamos las estructuras compartidas ya calculadas
            return process_refactoring(complete_data_file, idx, ruta_html_sol, orden_objetivos,
                                       desfase_global, nodo_metodo_orig, root_node, codigo_bytes, mapa_offsets)
        except Exception as e:
            print(f"❌ [Error] Saltando solución #{idx + 1} debido a un fallo: {e}")
            return "Explicación no disponible debido a un error."

    # 4. Lanzamos la ejecución en paralelo
    with ThreadPoolExecutor(max_workers=10) as executor:
        explicaciones_ordenadas = list(executor.map(procesar_hilo, range(len(solutions))))

    return enlaces_ordenados, explicaciones_ordenadas


def generar_explicacion_metodo_openai(codigo_original, codigo_refactorizado, fila_dict, objectives_list):
    # Convertimos el diccionario a un string bonito para el prompt
    import json
    info_solucion_json = json.dumps(fila_dict, indent=4, ensure_ascii=False)

    prompt = f"""
    Actúa como un Arquitecto de Software experto y consultor técnico. Tu objetivo es ayudar a un desarrollador 
    a decidir si debe ELEGIR esta solución de refactorización específica de entre varias opciones de un Frente de Pareto.
    Las refactorizaciones tratan de reducir la COMPLEJIDAD COGNITIVA del código (medida a nivel de método).

    MÉTRICAS DETALLADAS DE ESTA SOLUCIÓN (Formato JSON):
    {info_solucion_json}

    ORDEN DE LOS OBJETIVOS OPTIMIZADOS (Corresponde a los valores de la tupla 'solution'):
    {objectives_list}

    CÓDIGO ORIGINAL:
    {codigo_original}

    CÓDIGO REFACTORIZADO:
    {codigo_refactorizado}

    GUÍA DE INTERPRETACIÓN DE MÉTRICAS PARA TU ANÁLISIS:
    - 'solution': Valores óptimos encontrados. El orden de los números se corresponde con '{objectives_list}', 
        donde "extractions" es el número de extracciones realizadas, CC_diff es la diferencia de complejidad cognitiva
         entre la máxima y mínima complejidad cognitiva de todas las extracciones y el método original refactorizado, 
         y LOC_diff es la diferencia de líneas de código entre las máximas y mínimas líneas de código de todas
          las extracciones y el método original refactorizado.
    - 'solution_info (index,CC,LOC)': Lista de tuplas donde cada elemento detalla el (índice de extracción,
       Complejidad Cognitiva (CC) interna de ese bloque, y Líneas de Código (LOC) de ese bloque).
    - 'reductionComplexity' y 'finalComplexity': Cuánta complejidad cognitiva se le quita al método original y cómo queda de simple.
    - 'not_nested_...' vs 'nested_...': Indica si los métodos extraídos están libres o si hay métodos extraídos dentro
       de otros métodos extraídos (anidamiento).
    - 'min/max/mean/totalExtractedLOC' y 'CC': Te dicen la distribución del tamaño y la carga de los métodos extraídos
       (¿son todos homogéneos o hay uno gigante y otros diminutos?). LOS se refiere a líneas de ´codigo y CC a complejidad cognitiva.

    TAREA:
    Escribe una justificación de ingeniería de máximo 2-3 líneas orientada a la SELECCIÓN de esta solución. 
    Debes realizar un análisis crítico combinando lo que observas en el código con lo que revelan las métricas
    cuantitativas anteriores.

    REGLAS CRÍTICAS DE CONTROL DE CALIDAD:
    1. PROHIBIDO usar clichés vagos como "mejora la legibilidad", "es más limpio", "sigue buenas prácticas" o
        "facilita el mantenimiento". Sé específico con la semántica del código (nombres de funciones/variables).
    2. Menciona qué responsabilidad o lógica concreta (ej. "el procesamiento de paquetes", "la validación de rutas")
        se está aislando según el código modificado.
    3. Cohesiona las métricas con el diseño: Si ves que hay alta reducción de complejidad ('reductionComplexity')
        o elementos anidados ('nested_extractions' > 0), explica al desarrollador el impacto de ese trade-off
        arquitectónico en el mundo real.
    4. Define el PERFIL DE PREFERENCIA ideal: Termina la última frase indicando taxativamente para qué tipo
        de usuario es idónea esta solución (ej. "Ideal para usuarios que priorizan reducir la complejidad
        del método principal sin importarles generar anidamiento secundario" o "Ideal para un refactorizado
        conservador, plano y de bajo impacto").

    RESPUESTA:
    Devuelve ÚNICAMENTE el párrafo de justificación, sin introducciones, sin saludos y sin formato markdown pesado.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # O el modelo que utilices
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # Temperatura baja para un análisis técnico y consistente
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # print(f"[Error OpenAI] No se pudo obtener la justificación: {e}")
        return "Solución estructurada basada en los objetivos de optimización del Frente de Pareto."


def customize_3d_pf_plot(objetivos, solutions, fig, enlaces_ordenados, explicaciones_ordenadas):
    nadir = np.max(objetivos, axis=0)
    ref_point = nadir + 1
    n1, n2, n3 = ref_point

    parallel_face_colors = {
        'top_bottom': "#E6E6FA",
        'front_back': "#FFDAB9",
        'left_right': "#C1FFC1"
    }

    parallel_faces = [
        {'faces': [(0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7)], 'color': parallel_face_colors['top_bottom']},
        {'faces': [(0, 1, 5), (0, 5, 4), (2, 3, 7), (2, 7, 6)], 'color': parallel_face_colors['front_back']},
        {'faces': [(1, 2, 6), (1, 6, 5), (0, 3, 7), (0, 7, 4)], 'color': parallel_face_colors['left_right']}
    ]

    for sol in solutions:
        a, b, c = sol[0], sol[1], sol[2]
        x = [a, n1, n1, a, a, n1, n1, a]
        y = [b, b, n2, n2, b, b, n2, n2]
        z = [c, c, c, c, n3, n3, n3, n3]

        for group in parallel_faces:
            color = group['color']
            face_tris = group['faces']
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z,
                i=[f[0] for f in face_tris], j=[f[1] for f in face_tris], k=[f[2] for f in face_tris],
                color=color, opacity=1, flatshading=True, showscale=False,
                hoverinfo='skip'
            ), row=1, col=1)

    f1, f2, f3 = zip(*solutions)

    # =========================================================
    # Formatear las explicaciones para que no se salgan
    # =========================================================
    explicaciones_formateadas = []
    for exp in explicaciones_ordenadas:
        # Divide el texto en líneas de máximo 60 caracteres y las une con saltos de línea HTML
        texto_roto = "<br>".join(textwrap.wrap(exp, width=60))
        explicaciones_formateadas.append(texto_roto)

    customdata_combinado = list(zip(enlaces_ordenados, explicaciones_formateadas))

    fig.add_trace(go.Scatter3d(
        x=f1, y=f2, z=f3,
        mode='markers+text',
        marker=dict(size=10, color='black'),
        text=[f's{idx + 1}' for idx in range(len(solutions))],
        textposition='top center',
        textfont=dict(color='black', size=18),
        name='Solutions',
        customdata=customdata_combinado,
        hoverlabel=dict(align="left"),  # <-- NUEVO: Alinea el texto a la izquierda
        hovertemplate=(
            "<b>Solution: %{text}</b><br><br>"
            "EXTRACTIONS: %{x}<br>"
            "CC<sub>diff</sub>: %{y}<br>"
            "LOC<sub>diff</sub>: %{z}<br><br>"
            "<b>Explanation:</b><br>%{customdata[1]}"
            "<extra></extra>"
        )
    ), row=1, col=1)

    return fig


def customize_parallel_coordinates(solutions, fig, enlaces_ordenados):
    pasos = 30  # Densidad de puntos invisibles para asegurar el clic

    for idx, sol in enumerate(solutions):
        x_dense = []
        y_dense = []
        marker_sizes = []

        # Tramo 1: de EXTRACTIONS (x=0) a CCdiff (x=1)
        for i in range(pasos):
            t = i / pasos
            x_dense.append(t)
            y_dense.append(sol[0] + t * (sol[1] - sol[0]))
            # Solo mostramos el marcador principal en el extremo
            marker_sizes.append(8 if i == 0 else 0)

        # Tramo 2: de CCdiff (x=1) a LOCdiff (x=2)
        for i in range(pasos):
            t = i / pasos
            x_dense.append(1 + t)
            y_dense.append(sol[1] + t * (sol[2] - sol[1]))
            marker_sizes.append(8 if i == 0 else 0)

        # Punto final: LOCdiff (x=2)
        x_dense.append(2)
        y_dense.append(sol[2])
        marker_sizes.append(8)

        fig.add_trace(go.Scatter(
            x=x_dense,
            y=y_dense,
            mode='lines+markers',
            name=f's{idx + 1}',
            marker=dict(size=marker_sizes),  # Puntos intermedios ocultos
            line=dict(width=3),
            customdata=[enlaces_ordenados[idx]] * len(x_dense),
            # Tooltip limpio para que no muestre decimales raros en medio de la línea
            hovertemplate="<b>%{fullData.name}</b><extra></extra>"
        ), row=1, col=2)

    return fig


def customize_plotly_figures(nombres_objetivos, fig):
    # 1. Mapeo de nombres para que usen formato HTML válido en Plotly
    objective_map = {
        "extractions": "EXTRACTIONS",
        "cc": "CC<sub>diff</sub>",
        "loc": "LOC<sub>diff</sub>"
    }

    # Asignación de variables dinámicas
    x_label = objective_map.get(nombres_objetivos[0], nombres_objetivos[0])
    y_label = objective_map.get(nombres_objetivos[1], nombres_objetivos[1])
    z_label = objective_map.get(nombres_objetivos[2], nombres_objetivos[2])

    # 2. Configuramos las etiquetas del Eje X para el gráfico 2D (coordenadas paralelas)
    fig.update_xaxes(
        tickmode='array',
        tickvals=[0, 1, 2],
        ticktext=[x_label, y_label, z_label],
        row=1, col=2
    )

    # 3. Configuración original para el 3D y el layout general
    fig.update_layout(
        scene=dict(
            xaxis=dict(title=dict(text=x_label, font=dict(size=25))),
            yaxis=dict(title=dict(text=y_label, font=dict(size=25))),
            zaxis=dict(title=dict(text=z_label, font=dict(size=25))),
            aspectmode='data'
        ),
        scene_camera=dict(eye=dict(x=-1.25, y=-1.25, z=1.25)),
        hoverdistance=-1
    )

    texto_ayuda = (
        "<b>Navegación 3D:</b><br>"
        "🖱️ Clic Izquierdo: Girar<br>"
        "🔘 Rueda: Zoom<br>"
        "🖱️ Clic Derecho: Desplazar"
    )

    fig.add_annotation(
        text=texto_ayuda,
        xref="paper", yref="paper",
        x=0.01, y=0.98,          # Ubicación en la esquina superior izquierda (Subplot 1)
        showarrow=False,
        align="left",
        bgcolor="rgba(255, 255, 255, 0.85)",  # Fondo blanco semitransparente
        bordercolor="#d3d3d3",
        borderwidth=1,
        borderpad=8,
        font=dict(size=12, color="#333333", family="Arial")
    )

    return fig

def make_plotly_interactive(fig, output_html_path):
    fig.write_html(output_html_path, include_mathjax='cdn')
    print(f"3D PF saved in {output_html_path}.")

    script_interactivo = """
        <script>
        document.addEventListener('DOMContentLoaded', function(){
            var plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
            if (plotDiv) {

                // Único evento personalizado: Clic en las gráficas para abrir la URL
                plotDiv.on('plotly_click', function(data){
                    if(data.points && data.points[0] && data.points[0].customdata) {
                        var custom = data.points[0].customdata;
                        // Extraemos la URL dependiendo de cómo la empaquete Plotly (array o string)
                        var url = Array.isArray(custom) ? custom[0] : custom;

                        // Verificamos que realmente hay una URL válida antes de intentar abrirla
                        if (url && typeof url === 'string') {
                            window.open(url, '_blank');
                        }
                    }
                });

            }
        });
        </script>
        """
    with open(output_html_path, 'r', encoding='utf-8') as f:
        html_contenido = f.read()

    html_contenido = html_contenido.replace("</body>", f"{script_interactivo}\n</body>")

    return html_contenido


def crear_mapa_char_a_byte(texto):
    mapa = [0] * (len(texto) + 1)
    byte_offset = 0
    for i, char in enumerate(texto):
        mapa[i] = byte_offset
        byte_offset += len(char.encode('utf-8'))
    mapa[len(texto)] = byte_offset
    return mapa


def encontrar_tipo_real_ast(nodo_metodo, nombre_var, codigo_bytes):
    tipo_encontrado = ["Object"]

    def walk(n):
        if tipo_encontrado[0] != "Object": return
        if n.type == 'formal_parameter':
            name_node = n.child_by_field_name('name')
            if name_node and codigo_bytes[name_node.start_byte:name_node.end_byte].decode('utf-8') == nombre_var:
                type_node = n.child_by_field_name('type')
                if type_node: tipo_encontrado[0] = codigo_bytes[type_node.start_byte:type_node.end_byte].decode('utf-8')
        elif n.type == 'local_variable_declaration':
            type_node = n.child_by_field_name('type')
            for child in n.children:
                if child.type == 'variable_declarator':
                    name_node = child.child_by_field_name('name')
                    if name_node and codigo_bytes[name_node.start_byte:name_node.end_byte].decode(
                            'utf-8') == nombre_var:
                        if type_node: tipo_encontrado[0] = codigo_bytes[type_node.start_byte:type_node.end_byte].decode(
                            'utf-8')
        elif n.type == 'enhanced_for_statement':
            name_node = n.child_by_field_name('name')
            if name_node and codigo_bytes[name_node.start_byte:name_node.end_byte].decode('utf-8') == nombre_var:
                type_node = n.child_by_field_name('type')
                if type_node: tipo_encontrado[0] = codigo_bytes[type_node.start_byte:type_node.end_byte].decode('utf-8')
        for child in n.children: walk(child)

    walk(nodo_metodo)
    return tipo_encontrado[0]


def obtener_locales_metodo(nodo_metodo, codigo_bytes):
    locales = set()
    def buscar(n):
        if n.type in ['variable_declarator', 'formal_parameter']:
            name_node = n.child_by_field_name('name')
            if name_node: locales.add(codigo_bytes[name_node.start_byte:name_node.end_byte].decode('utf8'))
        elif n.type == 'enhanced_for_statement':
            name_node = n.child_by_field_name('name')
            if name_node: locales.add(codigo_bytes[name_node.start_byte:name_node.end_byte].decode('utf8'))
        for child in n.children: buscar(child)
    buscar(nodo_metodo)
    return locales


def obtener_parametros_limpios(nodo_metodo_orig, lista_nodos, codigo_bytes, start, end):
    locales_metodo = obtener_locales_metodo(nodo_metodo_orig, codigo_bytes)  # <-- AÑADIR AQUÍ
    declaradas_dentro = set()
    usos_externos = {}

    def registrar_declaracion(n_id):
        if n_id and start <= (n_id.start_byte + n_id.end_byte) / 2 <= end:
            name = codigo_bytes[n_id.start_byte:n_id.end_byte].decode('utf8')
            declaradas_dentro.add(name)

    def buscar_decls(n):
        if n.type == 'variable_declarator':
            registrar_declaracion(n.child_by_field_name('name'))
        elif n.type == 'enhanced_for_statement':
            name_node = n.child_by_field_name('name')
            if name_node:
                registrar_declaracion(name_node)
            else:
                for child in n.children:
                    if child.type == 'identifier':
                        registrar_declaracion(child)
                        break
        for child in n.children: buscar_decls(child)

    for nodo in lista_nodos: buscar_decls(nodo)

    def buscar_usos(n):
        if n.type == 'identifier' and start <= (n.start_byte + n.end_byte) / 2 <= end:
            nombre = codigo_bytes[n.start_byte:n.end_byte].decode('utf8')
            parent = n.parent
            es_valido = True

            if parent.type == 'method_invocation' and parent.child_by_field_name('name') == n: es_valido = False
            if parent.type == 'field_access' and parent.child_by_field_name('field') == n: es_valido = False
            if nombre in declaradas_dentro: es_valido = False
            if nombre in ['this', 'super', 'null', 'true', 'false'] or nombre[0].isupper(): es_valido = False
            if nombre not in locales_metodo: es_valido = False # <-- AÑADIR ESTA LÍNEA AQUÍ

            if es_valido:
                if nombre not in usos_externos:
                    usos_externos[nombre] = encontrar_tipo_real_ast(nodo_metodo_orig, nombre, codigo_bytes)

        for child in n.children: buscar_usos(child)

    for nodo in lista_nodos: buscar_usos(nodo)
    return usos_externos


# Lógica para detectar si hay variables que deben ser retornadas (parámetros de salida)
def obtener_parametros_salida(nodo_metodo_orig, codigo_bytes, start, end):
    locales_metodo = obtener_locales_metodo(nodo_metodo_orig, codigo_bytes)
    modificadas_dentro = set()
    declaradas_dentro = set()
    usadas_despues = set()

    # 1. Buscar modificaciones o declaraciones dentro del bloque
    def buscar_modificaciones(n):
        if n.type == 'assignment_expression':
            left = n.child_by_field_name('left')
            # Si el lado izquierdo es un identificador puro (reasignación de referencia o primitivo)
            if left and left.type == 'identifier' and start <= n.start_byte <= end:
                modificadas_dentro.add(codigo_bytes[left.start_byte:left.end_byte].decode('utf8'))
        elif n.type == 'update_expression':
            for child in n.children:
                if child.type == 'identifier' and start <= n.start_byte <= end:
                    modificadas_dentro.add(codigo_bytes[child.start_byte:child.end_byte].decode('utf8'))
        elif n.type == 'variable_declarator':
            name_node = n.child_by_field_name('name')
            if name_node and start <= name_node.start_byte <= end:
                # Modificado: Las declaradas dentro nacen y mueren aquí, no son salidas globales
                declaradas_dentro.add(codigo_bytes[name_node.start_byte:name_node.end_byte].decode('utf8'))
        elif n.type == 'enhanced_for_statement':
            name_node = n.child_by_field_name('name')
            if name_node and start <= name_node.start_byte <= end:
                declaradas_dentro.add(codigo_bytes[name_node.start_byte:name_node.end_byte].decode('utf8'))

        for child in n.children:
            buscar_modificaciones(child)

    # 2. Buscar usos de variables DESPUÉS del bloque (offset de inicio mayor que 'end')
    def buscar_usos_despues(n):
        if n.type == 'identifier' and (n.start_byte + n.end_byte) / 2 > end:
            nombre = codigo_bytes[n.start_byte:n.end_byte].decode('utf8')
            parent = n.parent
            es_valido = True

            if parent.type == 'method_invocation' and parent.child_by_field_name('name') == n: es_valido = False
            if parent.type == 'field_access' and parent.child_by_field_name('field') == n: es_valido = False
            if nombre in ['this', 'super', 'null', 'true', 'false'] or nombre[0].isupper(): es_valido = False

            if es_valido:
                usadas_despues.add(nombre)

        for child in n.children:
            buscar_usos_despues(child)

    # Ejecutamos una sola vez cada búsqueda sobre el AST
    buscar_modificaciones(nodo_metodo_orig)
    buscar_usos_despues(nodo_metodo_orig)

    # 3. Intersección: Se modificó dentro, se usa después, es local... y RESTAMOS las declaradas dentro
    salidas = modificadas_dentro.intersection(usadas_despues).intersection(locales_metodo) - declaradas_dentro

    resultado_salidas = {}
    for var in salidas:
        resultado_salidas[var] = encontrar_tipo_real_ast(nodo_metodo_orig, var, codigo_bytes)

    return resultado_salidas


def validar_y_encadenar_nodos(root_node, start, end):
    padre = root_node.descendant_for_byte_range(start, end)
    if padre.type != 'block':
        while padre and padre.parent and padre.type != 'block':
            if padre.type.endswith('statement') or padre.type.endswith('declaration'): break
            padre = padre.parent
        return [padre], padre.start_byte, padre.end_byte

    nodos_encadenados = []
    for child in padre.children:
        if child.type in ['{', '}']: continue
        overlap_start = max(start, child.start_byte)
        overlap_end = min(end, child.end_byte)
        overlap_len = max(0, overlap_end - overlap_start)
        child_len = child.end_byte - child.start_byte

        if child_len > 0 and (overlap_len / child_len) > 0.5:
            nodos_encadenados.append(child)

    if not nodos_encadenados: return [padre], padre.start_byte, padre.end_byte
    return nodos_encadenados, nodos_encadenados[0].start_byte, nodos_encadenados[-1].end_byte


def calcular_metodo_y_desfase(ruta_csv_principal, ruta_csv_extra, root_node, codigo_bytes, mapa_offsets):
    nombre_archivo = os.path.basename(ruta_csv_principal)
    nombre_metodo = nombre_archivo.split('-')[0]
    match_linea = re.search(r'-(\d+)_', nombre_archivo)
    if not match_linea:
        raise ValueError(f"No se pudo extraer el número de línea del archivo: {nombre_archivo}")

    linea_referencia = int(match_linea.group(1))
    metodos_candidatos = []

    def buscar_metodos(n):
        if n.type == 'method_declaration':
            name_node = n.child_by_field_name('name')
            if name_node and codigo_bytes[name_node.start_byte:name_node.end_byte].decode('utf-8') == nombre_metodo:
                metodos_candidatos.append(n)
        for child in n.children:
            buscar_metodos(child)

    buscar_metodos(root_node)

    if not metodos_candidatos:
        raise ValueError(f"No se encontró ningún método llamado '{nombre_metodo}' en el código.")

    nodo_metodo_orig = min(metodos_candidatos, key=lambda n: abs((n.start_point[0] + 1) - linea_referencia))
    body_node = nodo_metodo_orig.child_by_field_name('body')
    if not body_node or body_node.type != 'block':
        raise ValueError("El método encontrado no tiene un bloque de código válido.")

    sentencias_reales = [c for c in body_node.children if c.type not in ['{', '}', 'block_comment', 'line_comment']]
    if not sentencias_reales:
        return 0, nodo_metodo_orig

    real_body_start_byte = sentencias_reales[0].start_byte
    max_diferencia = -1
    csv_body_start_char = 0

    with open(ruta_csv_extra, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                start_c = int(row[0])
                end_c = int(row[1])
                diferencia = end_c - start_c
                if diferencia > max_diferencia:
                    max_diferencia = diferencia
                    csv_body_start_char = start_c
            except (ValueError, IndexError):
                continue

    if max_diferencia == -1:
        raise ValueError("No se encontraron rangos válidos en el CSV extra.")

    csv_body_start_byte = mapa_offsets[csv_body_start_char]
    desfase_global = real_body_start_byte - csv_body_start_byte

    return desfase_global, nodo_metodo_orig


def generar_nombre_metodo_openai(codigo_metodo, nombres_usados):

    # Formateamos la lista de nombres para el prompt
    nombres_evitar = ", ".join(nombres_usados) if nombres_usados else "Ninguno"

    prompt = f"""
    Analiza el siguiente fragmento de código Java extraído de un método y propón un único nombre representativo
    en formato camelCase (por ejemplo, 'calcularTotal', 'validarUsuario'). 
    Devuelve ÚNICAMENTE el nombre del método, sin introducciones, sin explicaciones, 
    sin punto y final y sin comillas.

    IMPORTANTE: NO uses ninguno de los siguientes nombres,
     ya que han sido utilizados previamente en esta clase: [{nombres_evitar}]

    Código:
    {codigo_metodo}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=15
        )
        nombre = response.choices[0].message.content.strip()
        nombre_limpio = re.sub(r'[^a-zA-Z0-9_]', '', nombre)
        return nombre_limpio if nombre_limpio else "metodoExtraido"
    except Exception as e:
        # print(f"[Error OpenAI] No se pudo obtener el nombre: {e}")
        return "metodoExtraido"


def process_refactoring(ruta_csv_principal, n_solucion, ruta_salida_html, objectives_list,
                        desfase_global, nodo_metodo_orig, root_node, codigo_bytes, mapa_offsets):
    original_method_name = os.path.basename(ruta_csv_principal).split('-')[0]

    with open(ruta_csv_principal, mode='r', encoding='utf-8') as f:
        reader = list(csv.DictReader(f))
        fila = reader[n_solucion]
        lista_offsets = ast.literal_eval(fila['offsets'])

    extracciones_procesadas = []
    nombres_usados = []

    for idx, (char_start, char_end) in enumerate(lista_offsets):
        byte_start_bruto = mapa_offsets[char_start]
        byte_end_bruto = mapa_offsets[char_end]

        byte_start_real = byte_start_bruto + desfase_global
        byte_end_real = byte_end_bruto + desfase_global

        lista_nodos, start_final, end_final = validar_y_encadenar_nodos(root_node, byte_start_real, byte_end_real)

        params = obtener_parametros_limpios(nodo_metodo_orig, lista_nodos, codigo_bytes, start_final, end_final)
        params_salida = obtener_parametros_salida(nodo_metodo_orig, codigo_bytes, start_final, end_final)

        args_firma = [f"{tipo} {nombre}" for nombre, tipo in params.items()]
        args_llamada = list(params.keys())

        texto_original_extraido = codigo_bytes[start_final:end_final].decode('utf-8')

        nombre_base = generar_nombre_metodo_openai(texto_original_extraido, nombres_usados)
        nombres_usados.append(nombre_base)

        nombre_metodo_nuevo = f"{nombre_base}"

        tipo_retorno = "void"
        var_retorno = None
        asignacion_llamada = ""

        if len(params_salida) > 0:
            var_retorno, tipo_retorno = list(params_salida.items())[0]
            asignacion_llamada = f"{var_retorno} = "
            if len(params_salida) > 1:
                vars_criticas = ", ".join(params_salida.keys())
                alerta_comentario = (
                    f"\n    // ⚠️ ¡ALERTA DE COMPILACIÓN! — LIMITACIÓN DE JAVA ⚠️\n"
                    f"    // Este bloque requería devolver: [{vars_criticas}].\n"
                    f"    // Solo se está retornando '{var_retorno}'. El código requerirá adaptación manual.\n"
                )
                texto_original_extraido = alerta_comentario + texto_original_extraido
                print(
                    f"[Aviso] Extracción {idx} en solución {n_solucion + 1} tiene múltiples salidas. "
                    f"Solo se retornará '{var_retorno}' por limitaciones de Java.")

        firma = f"private {tipo_retorno} {nombre_metodo_nuevo}({', '.join(args_firma)})"

        if "throws Exception" in codigo_bytes[nodo_metodo_orig.start_byte:nodo_metodo_orig.end_byte].decode('utf-8'):
            if "throw " in codigo_bytes[start_final:end_final].decode('utf-8'):
                firma += " throws Exception"

        # Aplicamos la reasignación en la llamada (si corresponde)
        llamada = f"{asignacion_llamada}{nombre_metodo_nuevo}({', '.join(args_llamada)});"

        extracciones_procesadas.append({
            'idx': idx, 'start': start_final, 'end': end_final,
            'firma': firma, 'llamada': llamada, 'texto_original': texto_original_extraido,
            'var_retorno': var_retorno  # Guardamos la variable que necesita un 'return'
        })

    def obtener_reemplazos_directos(c_start, c_end):
        sublist = []
        for e in extracciones_procesadas:
            if c_start <= e['start'] and e['end'] <= c_end:
                if e['start'] != c_start or e['end'] != c_end:
                    sublist.append(e)
        directos = []
        for e1 in sublist:
            es_anidado = False
            for e2 in sublist:
                if e1 != e2 and e2['start'] <= e1['start'] and e1['end'] <= e2['end']:
                    es_anidado = True
                    break
            if not es_anidado: directos.append(e1)
        return directos

    bytes_metodo_modificado = codigo_bytes[nodo_metodo_orig.start_byte:nodo_metodo_orig.end_byte]
    reemplazos_orig = obtener_reemplazos_directos(nodo_metodo_orig.start_byte, nodo_metodo_orig.end_byte)
    reemplazos_orig_ordenados = sorted(reemplazos_orig, key=lambda x: x['start'], reverse=True)

    for r in reemplazos_orig_ordenados:
        rel_start = r['start'] - nodo_metodo_orig.start_byte
        rel_end = r['end'] - nodo_metodo_orig.start_byte
        # MARCADOR: Envolvemos la llamada al método
        llamada_marcada = f"[[START_CALL_{r['idx']}]]" + r['llamada'] + f"[[END_CALL_{r['idx']}]]"
        bytes_metodo_modificado = bytes_metodo_modificado[:rel_start] + llamada_marcada.encode(
            'utf-8') + bytes_metodo_modificado[rel_end:]

    texto_metodo_original_refactorizado = bytes_metodo_modificado.decode('utf-8')

    nuevos_metodos_codigo = []
    for ext in extracciones_procesadas:
        bytes_cuerpo = ext['texto_original'].encode('utf-8')
        reemplazos_internos = obtener_reemplazos_directos(ext['start'], ext['end'])
        reemplazos_internos_ordenados = sorted(reemplazos_internos, key=lambda x: x['start'], reverse=True)

        for r in reemplazos_internos_ordenados:
            rel_start = r['start'] - ext['start']
            rel_end = r['end'] - ext['start']
            # MARCADOR: Envolvemos las llamadas anidadas dentro de otros métodos extraídos
            llamada_marcada = f"[[START_CALL_{r['idx']}]]" + r['llamada'] + f"[[END_CALL_{r['idx']}]]"
            bytes_cuerpo = bytes_cuerpo[:rel_start] + llamada_marcada.encode('utf-8') + bytes_cuerpo[rel_end:]

        cuerpo_final_texto = bytes_cuerpo.decode('utf-8')

        if ext['var_retorno']:
            cuerpo_final_texto = cuerpo_final_texto.rstrip() + f"\n    return {ext['var_retorno']};\n"

        # MARCADOR: Envolvemos el método extraído al completo (firma, llaves y cuerpo)
        codigo_completo_metodo = f"[[START_METHOD_{ext['idx']}]]{ext['firma']} {{\n    {cuerpo_final_texto}\n}}[[END_METHOD_{ext['idx']}]]"
        nuevos_metodos_codigo.append(codigo_completo_metodo)

    bloque_final = []
    bloque_final.append("// ========================================================")
    bloque_final.append(f"// SUSTITUIR EL MÉTODO ORIGINAL '{original_method_name}' POR ESTE:")
    bloque_final.append("// ========================================================")
    bloque_final.append(texto_metodo_original_refactorizado.strip())
    bloque_final.append("\n// ========================================================")
    bloque_final.append("// PEGAR ESTOS NUEVOS MÉTODOS DEBAJO DEL ANTERIOR:")
    bloque_final.append("// ========================================================")
    for m_codigo in nuevos_metodos_codigo:
        bloque_final.append(m_codigo)
        bloque_final.append("")

    texto_codigo_limpio = "\n".join(bloque_final)

    # 1. Copiar los bytes del método original directamente
    bytes_original_marcado = codigo_bytes[nodo_metodo_orig.start_byte:nodo_metodo_orig.end_byte]

    # 2. Recopilamos todos los puntos de inserción (START y END) operando 100% en BYTES
    eventos = []
    for ext in extracciones_procesadas:
        local_start = ext['start'] - nodo_metodo_orig.start_byte
        local_end = ext['end'] - nodo_metodo_orig.start_byte
        idx = ext['idx']
        longitud = local_end - local_start

        eventos.append(
            {'pos': local_start, 'is_start': True, 'len': longitud, 'tag': f"[[START_BOX_{idx}]]".encode('utf-8')})
        eventos.append(
            {'pos': local_end, 'is_start': False, 'len': longitud, 'tag': f"[[END_BOX_{idx}]]".encode('utf-8')})

    def sort_key(e):
        length_factor = -e['len'] if e['is_start'] else e['len']
        return (e['pos'], e['is_start'], length_factor)

    eventos.sort(key=sort_key, reverse=True)

    # 3. Insertamos las etiquetas en la secuencia de bytes de atrás hacia adelante
    for e in eventos:
        pos = e['pos']
        bytes_original_marcado = bytes_original_marcado[:pos] + e['tag'] + bytes_original_marcado[pos:]

    # 4. Finalmente, decodificamos a string con los marcadores ya en su sitio perfecto
    codigo_marcado = bytes_original_marcado.decode('utf-8')

    # 5. Escapar caracteres de Java para que sean HTML seguros
    import html
    codigo_original_html = html.escape(codigo_marcado)
    codigo_refactorizado_html = html.escape(texto_codigo_limpio)

    # 6. Reemplazar los placeholders por las etiquetas SPAN reales en AMBOS lados
    for idx in range(len(lista_offsets)):
        # Generar un tono de color único para cada extracción y el número visible
        hue = (idx * 137) % 360  # Truco para que colores consecutivos sean muy distintos
        estilo_dinamico = f'style="--tema-hue: {hue};" data-num="{idx + 1}"'

        # Lado izquierdo (código original)
        codigo_original_html = codigo_original_html.replace(
            f"[[START_BOX_{idx}]]", f'<span class="recuadro-extraccion" {estilo_dinamico}>'
        ).replace(f"[[END_BOX_{idx}]]", '</span>')

        # Lado derecho (1): Llamadas a los métodos
        codigo_refactorizado_html = codigo_refactorizado_html.replace(
            f"[[START_CALL_{idx}]]", f'<span class="recuadro-llamada" {estilo_dinamico}>'
        ).replace(f"[[END_CALL_{idx}]]", '</span>')

        # Lado derecho (2): Métodos extraídos completos
        codigo_refactorizado_html = codigo_refactorizado_html.replace(
            f"[[START_METHOD_{idx}]]", f'<span class="recuadro-metodo" {estilo_dinamico}>'
        ).replace(f"[[END_METHOD_{idx}]]", '</span>')

    html_plantilla = f"""<!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>Refactorización - Solución {n_solucion + 1}</title>
        <style>
            /* Contenedor en paralelo (Izquierda vs Derecha) */
            .comparador-container {{
                display: flex;
                gap: 20px;
                width: 100%;
                font-family: 'Courier New', Courier, monospace;
            }}
            .columna-codigo {{
                flex: 1;
                background: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 6px;
                padding: 15px;
                overflow-x: auto;
            }}
            .columna-codigo h3 {{
                margin-top: 0;
                font-family: sans-serif;
                color: #495057;
                border-bottom: 2px solid #dee2e6;
                padding-bottom: 8px;
            }}
            /* Mantiene el contenedor seguro y con scroll si es necesario */
            .columna-codigo code {{
                display: block;
                width: max-content;
                min-width: 100%;
            }}
            
            /* 1. FORZAR AL CONTENEDOR 'PRE' A NO ROMPER LÍNEAS NUNCA */
            pre, pre * {{
                white-space: pre !important;
                word-break: normal !important;
                overflow-wrap: normal !important;
            }}
            
            pre {{
                margin: 0;
                display: block !important;
                width: max-content !important; /* Se estira obligatoriamente hasta el final de la línea más larga */
                min-width: 100% !important;
                box-sizing: border-box !important;
            }}
            
            /* 2. ASEGURAR QUE LOS RECUADROS GRANDES SE ADAPTEN (MÉTODOS Y EXTRACCIONES) */
            .recuadro-extraccion, .recuadro-metodo {{
                border: 2px dashed hsl(var(--tema-hue), 75%, 50%);
                background-color: hsla(var(--tema-hue), 75%, 50%, 0.1);
                border-radius: 4px;
                padding: 4px 8px;
                
                display: block !important;
                width: 100% !important; /* Rellena el 100% del max-content del 'pre' padre */
                box-sizing: border-box !important;
                
                position: relative;
                margin-top: 4px;
                margin-bottom: 4px;
                
                /* Doble seguridad anti-salto de línea aquí dentro */
                white-space: pre !important;
                word-break: normal !important;
                overflow-wrap: normal !important;
            }}

            /* 3. RECUADRO CORTO PARA LAS LLAMADAS A MÉTODOS */
            .recuadro-llamada {{
                border: 2px dashed hsl(var(--tema-hue), 75%, 50%);
                background-color: hsla(var(--tema-hue), 75%, 50%, 0.1);
                border-radius: 4px;
                
                /* Dejamos 32px de padding a la derecha para que entre el número holgadamente */
                padding: 2px 32px 2px 6px; 
                
                /* inline-block hace que la caja se ajuste al tamaño del texto */
                display: inline-block !important; 
                box-sizing: border-box !important;
                
                position: relative;
                white-space: pre !important;
            }}

            /* 4. FORMATO UNIFICADO PARA LOS NÚMEROS (CIRCULITOS) */
            .recuadro-extraccion::after, 
            .recuadro-metodo::after, 
            .recuadro-llamada::after {{
                content: attr(data-num); /* Lee el número que inyectas en Python */
                position: absolute;
                background-color: hsl(var(--tema-hue), 75%, 50%);
                color: white;
                border-radius: 50%;
                width: 20px;
                height: 20px;
                
                /* Centrado del texto dentro del círculo */
                display: inline-flex;
                align-items: center;
                justify-content: center;
                font-family: sans-serif;
                font-size: 11px;
                font-weight: bold;
                line-height: 1;
            }}
            
            /* Posicionamiento del número en recuadros grandes (arriba a la derecha) */
            .recuadro-extraccion::after, 
            .recuadro-metodo::after {{
                top: 6px;
                right: 8px;
            }}

            /* Posicionamiento del número en llamadas (centrado verticalmente a la derecha) */
            .recuadro-llamada::after {{
                top: 50%;
                right: 6px;
                transform: translateY(-50%);
            }}
        </style>
    </head>
    <body>
        <h2>Análisis de la Solución</h2>
        
        <div class="comparador-container">
            <div class="columna-codigo">
                <h3>Código Original</h3>
                <pre><code>{codigo_original_html}</code></pre>
            </div>
            
            <div class="columna-codigo">
                <h3>Código Refactorizado</h3>
                <pre><code>{codigo_refactorizado_html}</code></pre>
            </div>
        </div>
    </body>
    </html>
    """

    with open(ruta_salida_html, 'w', encoding='utf-8') as out_f:
        out_f.write(html_plantilla)

    # 1. Obtenemos los strings limpios
    codigo_original_limpio = codigo_bytes[nodo_metodo_orig.start_byte:nodo_metodo_orig.end_byte].decode('utf-8')
    texto_codigo_limpio = "\n".join(bloque_final)

    # 2. Llamada a OpenAI enviando las métricas para que la explicación sea ultra-específica
    explicacion = generar_explicacion_metodo_openai(codigo_original_limpio, texto_codigo_limpio, fila, objectives_list)
    return explicacion