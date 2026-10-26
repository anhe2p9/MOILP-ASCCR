import os
import os.path


import subprocess
import pandas as pd

import sys
import csv
import argparse
import configparser
from pathlib import Path

import utils.results_utils as results_utils

from ILP_CC_reducer.operations.ILP_engine import ILPEngine
from ILP_CC_reducer.algorithms import __all__ as ALGORITHMS_NAMES

model_engine = ILPEngine()

def main_one_obj(alg_name: str, instance_path: Path=None, tau: int=15, objective: str=None,
                 obtain_model: bool=False, solve_model: bool=False, time_limit: int=3600):

    csv_data = ["project", "class", "method", "missingFile", "emptyFile",
         "numberOfVariables", "numberOfConstraints", "numberOfExtractions", "numberOfUsedVariables",
         "initialComplexity", "solution", "offsets", "extractions",
         "NOT_nestedSolution", "NOT_nestedExtractions",
         "NESTED_solution", "NESTED_extractions",
         "reductionComplexity", "finalComplexity",
         "minExtractedLOC", "maxExtractedLOC", "meanExtractedLOC", "totalExtractedLOC", "nestedLOC", 
         "minReductionOfCC", "maxReductionOfCC", "meanReductionOfCC", "totalReductionOfCC", "nestedCC",
         "minExtractedParams", "maxExtractedParams", "meanExtractedParams", "totalExtractedParams",
         "modelStatus", "terminationCondition", "executionTime"]

    # Crear el archivo desde cero (sobrescribir si existe)
    csv_path = f"{instance_path}_{objective}_results.csv"
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_data)
        writer.writeheader()

    for project_folder in sorted(os.listdir(instance_path)):
        project_folder = Path(project_folder)
        print(f"Project folder: {project_folder}")
        project_path = instance_path / project_folder
        for class_folder in sorted(os.listdir(project_path)):
            class_folder = Path(class_folder)
            print(f"Class folder: {class_folder}")
            class_path = project_path / class_folder
            for method_folder in sorted(os.listdir(class_path)):
                method_folder = Path(method_folder)
                print(f"Method folder: {method_folder}")
                method_path = class_path / method_folder
                print(f"Total path: {method_path}")
                if os.path.isdir(method_path):
                    project_folder_name = project_folder.name
                    print(f"Processing project: {project_folder_name}, class: {class_folder}, method: {method_folder}")

                    # Check threshold
                    check_threshold(method_path, tau)

                    # Process algorithm
                    algorithm = model_engine.get_algorithm_from_name(alg_name)
                    
                    # Process instance
                    instance = model_engine.load_concrete(method_path)
                    
                    folders_data = {
                        "project": str(project_folder_name),
                        "class": str(class_folder),
                        "method": str(method_folder)
                                    }

                    variables, constraints = results_utils.analyze_model_data(method_path, (objective,))
                    print(f"There are {variables} variables.")
                    print(f"There are {constraints} constraints.")

                    # Complete info to ensure code structure
                    info_dict = {
                        "variables": variables,
                        "constraints": constraints,
                        "folders_data": folders_data,
                        "objective": objective,
                        "obtain_model": obtain_model,
                        "solve_model": solve_model,
                        "time_limit": time_limit
                    }

                    results_csv = model_engine.apply_algorithm(algorithm, instance, tau, info_dict)

                    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(results_csv)
                    print(f"Added line successfully to {csv_path}.")

    if solve_model:
        print(f"CSV file with results for one objective correctly saved in {csv_path}.")

    print(
        "============================================================================================================")





def main_multiobjective(num_of_objectives: int, alg_name: str, instance_folder: Path, general_path: str,
                        tau: int=15, subdivisions: tuple=None, weights: tuple=None, objectives: tuple=None,
                        time_limit: int=3600):


    # Process algorithm
    algorithm = model_engine.get_algorithm_from_name(alg_name)
    
    # Process instance
    instance = model_engine.load_concrete(instance_folder)
    instance["instance_folder"] = general_path

    variables, constraints = results_utils.analyze_model_data(instance_folder, objectives)
    print(f"There are {variables} variables.")
    print(f"There are {constraints} constraints.")

    # Complete info to ensure code structure
    info_dict = {
        "num_of_objectives": num_of_objectives,
        "objectives_list": objectives,
        "subdivisions": subdivisions,
        "weights": weights,
        "time_limit": time_limit
    }

    if alg_name == 'WeightedSumAlgorithm':
        csv_data, concrete_model, output_data = model_engine.apply_algorithm(algorithm, instance, tau, info_dict)
        write_output_to_files(general_path, csv_data, output_data)
    elif (alg_name == 'HybridMethodAlgorithm'
          or alg_name == 'EpsilonConstraintAlgorithm'):
        # Create parent directory explicitly
        model_engine.apply_algorithm(algorithm, instance, tau, info_dict)
        write_output_to_files(general_path)
    else:
        sys.exit(f"Unknown algorithm '{alg_name}'. Algorithms for more than one objective must be:"
                 f" WeightedSumAlgorithm, EpsilonConstraintAlgorithm, or HybridMethodAlgorithm.")


def get_all_path_names(instance_folder: Path):
    method_name = os.path.basename(instance_folder)
    class_name = os.path.basename(instance_folder.parent)
    project_name = os.path.basename(instance_folder.parent.parent)
    return method_name, class_name, project_name


def write_output_to_files(general_path: str, csv_info: list = None,
                          output_data: list = None, complete_data: list = None):

    if not os.path.exists(Path(general_path).parent):
        os.makedirs(Path(general_path).parent)

    if csv_info:
        # Save data in a CSV file
        filename = f"{general_path}_results.csv"

        if os.path.exists(filename):
            os.remove(filename)

        with open(filename, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerows(csv_info)
            print(f"CSV file correctly created in {filename}.")

    if output_data:
        # Save output in a TXT file
        output_filename = f"{general_path}_output.txt"

        if os.path.exists(output_filename):
            os.remove(output_filename)

        with open(output_filename, "w") as f:
            for linea in output_data:
                f.write(linea + "\n")
            print(f"Output correctly saved in {output_filename}.")

    if complete_data:
        # Save complete data in a TXT file
        complete_data_filename = f"{general_path}_complete_data.csv"

        if os.path.exists(complete_data_filename):
            os.remove(complete_data_filename)

        with open(complete_data_filename,
                  mode="w", newline="", encoding="utf-8") as complete_csv:
            writer = csv.writer(complete_csv)
            writer.writerows(complete_data)
            print(f"Complete CSV file correctly created in {complete_data_filename}.")




PROPERTIES_FILE = "properties.ini"

def delete_ini(path):
    if os.path.exists(path):
        os.remove(path)

def load_config(file=PROPERTIES_FILE):
    """Loads configuration from a file .ini if it exists."""

    config = configparser.ConfigParser()
    config.read(file)

    parameters = {}

    if "Properties" in config:
        section = config["Properties"]
        parameters["model_instance"] = section["model_instance"]
        if "ilp_algorithm" in section:
            parameters["ilp_algorithm"] = section["ilp_algorithm"]
        if "threshold" in section:
            parameters["threshold"] = section.getint("threshold")
        if "subdivisions" in section:
            parameters["subdivisions"] = section.getint("subdivisions")
        if "weights" in section:
            parameters["weights"] = section["weights"]
        if "second_obj" in section:
            parameters["second_obj"] = section["second_obj"]
        if "results_dir" in section:
            parameters["results_dir"] = section["results_dir"]

    return parameters





def save_config(parameters, file=PROPERTIES_FILE):
    """Saves properties in a .ini file"""
    
    
    config = configparser.ConfigParser()
    # Convert lists/tuples into string before saving them
    config["Properties"] = {
        key: ",".join(map(str, value)) if isinstance(value, (list, tuple)) else str(value)
        for key, value in parameters.items()
    }

    with open(file, "w") as f:
        config.write(f)

    print(f"Properties saved in {file}")


def check_threshold(model_instance, threshold):
    model_instance = Path(model_instance)
    print(f"INSTANCE PATH: {model_instance}")

    sequences_file = next((f for f in model_instance.iterdir() if f.name.endswith('_sequences.csv')), None)
    if sequences_file:
        with sequences_file.open(newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            filas = list(reader)
            if len(filas) > 1 and len(filas[1]) > 2:
                x0_cc_value = int(filas[1][2])
                print(f"Actual CC: {x0_cc_value}.")
        if x0_cc_value <= threshold:
            sys.exit(f'Objective threshold must be lower than actual CC.')

    if not model_instance.is_dir():
        sys.exit(f'The model instance must be a folder with three CSV files (multiobjective)'
                 f' or the base path with all projects (one objective).')


def classify_solution_files(solutions_folder_path):
    solutions_folder = Path(solutions_folder_path)

    original_class = None
    complete_data_path = None
    refact_cache = None

    if not solutions_folder.is_dir():
        raise FileNotFoundError(f"Path '{solutions_folder_path}' is not a valid folder.")

    # Buscamos los archivos dentro de la carpeta
    for archivo in solutions_folder.iterdir():
        if archivo.is_file():
            nombre = archivo.name
            if nombre.endswith('.java'):
                original_class = archivo
            elif nombre.endswith('_complete_data.csv'):
                complete_data_path = archivo
            elif nombre.endswith('.csv'):
                refact_cache = archivo

    # --- Validación de archivos faltantes ---
    faltan = []
    if original_class is None:
        faltan.append("Clase original (.java)")
    if complete_data_path is None:
        faltan.append("Datos completos (_complete_data.csv)")
    if refact_cache is None:
        faltan.append("Cache de refactorización (.csv)")

    # Si la lista "faltan" tiene elementos, lanzamos el error
    if faltan:
        # Unimos los elementos con comas para un mensaje limpio
        mensaje_error = f"Error en '{solutions_folder.name}': Falta por definir: {', '.join(faltan)}"
        raise FileNotFoundError(mensaje_error)

    return original_class, complete_data_path, refact_cache


def obtain_arguments():
    """Defines arguments from command line and parse them."""

    parser = argparse.ArgumentParser(
        description='ILP model engine. Given an abstract model m, a model instance a, an algorithm a and optionally '
                    'a threshold t, '
                    'a determined number of subdivisions s or three weights w,'
                    'an objectives order o, and other additional argumentes,'
                    ' it applies the correspondent algorithm to find the optimal solutions of the model instance. '
                    'One can also give as input a properties file path.')
    parser.add_argument('-f', '--file', dest='properties_file', type=str, default=None,
                        help=f'Properties file name in case one want to give every parameter from a .ini file.')
    parser.add_argument('-n', '--num_of_objectives', dest='num_of_objectives', type=str, default=None,
                        help=f'Number of objectives to minimize.')
    parser.add_argument('-m', '--model_path', dest='model_path', type=str, default=None,
                        help='Path to the model to be analyzed (obtain number of variables and constraints.')
    parser.add_argument('-i', '--instance', dest='model_instance', type=str, default=None,
                        help='Model instance to be optimized. '
                             'It can be the folder path with the three data files in CSV format for multiobjective'
                             'or the general folder path with all instances for one objective.')
    parser.add_argument('-a', '--algorithm', dest='ilp_algorithm', type=str, default=None,
                        help=f'Algorithm to be applied to the model instance in the case of multiobjective ILP:'
                             f' {[a for a in ALGORITHMS_NAMES]}.')
    parser.add_argument('-t', '--tau', dest='threshold', type=int, default=None,
                        help=f'Threshold (tau) to be reached by the optimization model.')
    parser.add_argument('-s', '--subdivisions', dest='subdivisions', type=int,
                        default=None, help=f'Number of subdivisions to generate different weights.')
    parser.add_argument('-w', '--weights', dest='weights', type=str, default=None,
                        help=f'Weights assigned for weighted sum in the case of a specific combination of weights.'
                             f' Three weights w1,w2,w3 separated by comma (",").')
    parser.add_argument('-o', '--objectives', dest='objectives', type=str, default=None,
                        help=f'List of objectives to minimize. '
                             f'In case of two or three objectives, write them separated by comma (","):'
                             f' "obj1", "obj1,obj2" or "obj1,obj2,ob3".')
    parser.add_argument('--model', action='store_true',
                        help=f'For one objective, it tries to just obtain the model.')
    parser.add_argument('--solve', action='store_true',
                        help=f'For one objective, it tries to solve the model.')
    parser.add_argument('--plot', action='store_true',
                        help=f'Plots the result of the given result. It gives just one plot.')
    parser.add_argument('--3dPF', action='store_true',
                        help=f'Plots the 3D PF of the given result. It gives just one PF plot.')
    parser.add_argument('--relHV', action='store_true',
                        help=f'Plots the relative HV with respect time of the given result. It gives just one PF plot.')
    parser.add_argument( '--all_plots', action='store_true',
                        help=f'Plots all results in a given directory. More than one plot will be created.')
    parser.add_argument('--statistics', action='store_true',
                        help=f'Creates a CSV file with the statistics of all the results found in a given directory.'
                             f'The statistics are: hypervolume, median, iqr, average and std for each objective.')
    parser.add_argument('--all_3dPF', action='store_true',
                        help=f'Plots all 3D PFs in a given directory. More than one PF plot will be created.')
    parser.add_argument('--all_relHV', action='store_true',
                        help=f'Plots all relative HVs with respect time in a given directory.'
                             f' More than one PF plot will be created.')
    parser.add_argument('--input', dest='input_dir', type=str, default=None,
                        help=f'The input path for plots and/or statistics can be specified,'
                             f' and if there is no input path, the output path will be the general "output/results" '
                             f'folder for all results.')
    parser.add_argument('--output', dest='output_dir', type=str, default=None,
                        help=f'GLOBAL output folder: where --all_plots, --statistics, --all_3dPF and --all_relHV'
                             f' save their results. It does not affect a single execution (see --results_dir).')
    parser.add_argument('--results_dir', dest='results_dir', type=str, default=None,
                        help=f'Folder for a SINGLE execution: optimization results (CSV, TXT) and its individual'
                             f' plots (--plot, --3dPF, --relHV). If not given, "output/results/<project>" is used.')
    parser.add_argument('--save', action='store_true', help='Save properties in a .ini file')
    parser.add_argument('-tl', '--timelimit', dest='time_limit', type=int, default=3600,
                        help=f'Maximum desired time for problem resolution.')
    parser.add_argument('-oc', '--original_class', dest='original_class', type=str, default=None,
                        help=f'Path to the original class where the method original method is.')
    parser.add_argument('-rc', '--refactoring_cache', dest='refactoring_cache', type=str, default=None,
                        help=f'Path to the refactoring cache of the method.')
    parser.add_argument('-sfp', '--solution_path', dest='solution_path', type=str,
                        default=None, help=f'Path to the folder with files about solution needed'
                                           f' if the instance is already solved and '
                                           f'one just wants to represent the solution.')
    parser.add_argument('--show', action='store_true',
                        help='Open the interactive 3D PF HTML in the web browser once generated.')
    
    args = parser.parse_args()
    parameters = vars(args)
    

    return parameters


# ==========================================
# 1. PARAMETER LOADING AND CONFIGURATION
# ==========================================

def setup_configuration(args):
    """Loads properties, merges command-line arguments, and returns the final config."""
    config = {}
    if args['properties_file']:
        properties_file_path = Path(args['properties_file'])
        print(f"PROPERTIES FILE PATH: {properties_file_path}")
        if not properties_file_path.is_file():
            sys.exit('The model instance must be a .ini file.')
        config = load_config(properties_file_path)

    # Overwrite values with command-line arguments
    for key, value in args.items():
        if value is not None:
            config[key] = value

    if args["save"]:
        save_config(config)

    # Show final configuration
    print("Final configuration:")
    for key, val in config.items():
        print(f"   · {key} = {val}")

    return config


def parse_objectives_and_weights(config):
    """Parses objectives and weights from the configuration."""
    num_objs = int(config.get('num_of_objectives', 0)) if config.get('num_of_objectives') else None

    weights = config.get('weights')
    if weights and isinstance(weights, str):
        weights = tuple(map(float, weights.split(",")))

    objectives = config.get('objectives')
    if objectives:
        if isinstance(objectives, str):
            objectives = tuple(map(str, objectives.split(",")))
        if len(objectives) != num_objs:
            sys.exit("The length of the objectives list must be the same as the number of objectives specified.")
    else:
        all_objectives = ('extractions', 'cc', 'loc')
        objectives = all_objectives[:num_objs] if num_objs else ()

    return objectives, weights


# ==========================================
# 2. ALGORITHM EXECUTION
# ==========================================

def run_optimization(config, objectives, weights):
    """Executes the single-objective or multi-objective optimization pipeline."""
    num_objs = int(config.get('num_of_objectives', 0))
    model_instance = config.get('model_instance')
    threshold = int(config.get('threshold', 0)) if config.get('threshold') else 0
    ilp_algorithm = config.get('ilp_algorithm')
    time_limit = config.get('time_limit')

    if not num_objs:
        return None

    if model_instance and not num_objs:
        sys.exit('No number of objectives specified to minimize.')

    instance_path = Path(model_instance) if model_instance else None

    # --- SINGLE-OBJECTIVE ---
    if num_objs == 1:
        if model_instance:
            check_threshold(model_instance, threshold)
            algo = ilp_algorithm if ilp_algorithm else 'ObtainResultsAlgorithm'
            obtain_model = bool(config.get("model"))
            solve_model = bool(config.get("solve"))

            main_one_obj(algo, model_instance, threshold, objectives[0], obtain_model, solve_model, time_limit)
        else:
            sys.exit('General instance folder required.')
        return None

    # --- MULTI-OBJECTIVE ---
    elif num_objs > 1 and model_instance:
        check_threshold(model_instance, threshold)
        method_name, class_name, project_name = get_all_path_names(instance_path)

        results_dir = config.get('results_dir')
        if results_dir:
            # Carpeta elegida por el usuario: sin subcarpeta de proyecto
            base_output = results_dir
        else:
            # Comportamiento original
            base_output = os.path.join("output/results", project_name)

        general_path = os.path.join(
            base_output,
            f"{ilp_algorithm}_{'-'.join(objectives)}_{class_name}_{method_name}",
            method_name,
        )

        main_multiobjective(num_objs, ilp_algorithm, instance_path, general_path,
                            threshold, config.get('subdivisions'), weights, objectives, time_limit)
        return general_path

    return None


# ==========================================
# 3. PLOT AND STATISTICS GENERATION
# ==========================================

def generate_plots_and_stats(config, objectives, general_path, args):
    """Generates all plots and statistics configured by the user."""
    # Input/Output paths
    input_dir = config.get('input_dir') or config.get('results_dir') or "output/results"
    output_dir = config.get('output_dir')
    if not output_dir and input_dir:
        output_dir = str(Path(input_dir).parent)
    elif not output_dir and not input_dir:
        output_dir = "output/plots_and_statistics"

    num_objs = int(config.get('num_of_objectives', 0))
    refact_cache = Path(config['refactoring_cache']) if config.get('refactoring_cache') else None
    original_class = Path(config['original_class']) if config.get('original_class') else None

    # Individual plots (Only if there is a multi-objective results path)
    if general_path and num_objs > 1:
        results_csv_path = f"{general_path}_results.csv"
        complete_data_path = f"{general_path}_complete_data.csv"
        output_html_path = f"{general_path}_interactive_3dPF.html"

        if args["plot"]:
            if num_objs == 2:
                results_utils.generate_2d_pf_plot(results_csv_path, f"{general_path}_2DPF_plot.pdf")
            elif num_objs == 3:
                results_utils.generate_parallel_coordinates_plot(results_csv_path,
                                                                 f"{general_path}_parallel_coordinates_plot.pdf")

        if args["3dPF"]:
            if refact_cache and original_class:
                import utils.interactive_3DPF_with_solutions as interactive_3dpf
                interactive_3dpf.generate_3d_pf_and_parallel_coordinates_plot(objectives, complete_data_path,
                                                                              output_html_path, refact_cache,
                                                                              original_class,
                                                                              show=args["show"])
            else:
                results_utils.generate_3d_pf_plot(results_csv_path, f"{general_path}_3DPF.html")

        if args["relHV"]:
            results_utils.generate_relative_hypervolume_plot(complete_data_path,
                                                             f"{general_path}_relative_hv_with_time.pdf")

    # Global plots
    if args["all_plots"] and general_path:
        complete_data_path = f"{general_path}_complete_data.csv"
        results_utils.traverse_and_plot(input_dir, output_dir, complete_data_path, refact_cache, original_class)
    if args["statistics"]:
        results_utils.generate_statistics(input_dir, output_dir)
    if args["all_3dPF"]:
        results_utils.traverse_and_pf_plot(input_dir, output_dir)
    if args["all_relHV"]:
        results_utils.generate_global_relative_hv_vs_time(input_dir, output_dir)


# ==========================================
# 4. AUTOMATIC REFACTORING (JavaParser)
# ==========================================

def apply_pareto_refactoring(general_path, original_class_path):
    """Finds the lexicographical optimum in the Pareto CSV and calls JavaParser."""
    if not general_path or not original_class_path:
        print("⚠️ Cannot refactor: Missing results path or original class.")
        return

    csv_results_path = f"{general_path}_results.csv"
    jar_path = "refactor_java/target/refactor-app-1.0-SNAPSHOT-jar-with-dependencies.jar"

    if not os.path.exists(csv_results_path):
        print(f"⚠️ Results CSV not found at: {csv_results_path}")
        return

    try:
        # 1. Read and find lexicographical optimum (Prioritizes lower LOC_diff, then lower extractions)
        df = pd.read_csv(csv_results_path)

        # Ascending order by LOC_diff and then by extractions (lower is better)
        df_sorted = df.sort_values(by=['LOC_diff', 'extractions'], ascending=[True, True])
        best_sol = df_sorted.iloc[0]

        offsets_str = str(best_sol['offsets'])  # Expected format: "100-200,300-400"

        print(f"\n🚀 Lexicographical Optimum Found:")
        print(f"   · LOC_diff: {best_sol['LOC_diff']} | Extractions: {best_sol['extractions']}")
        print(f"   · Applying offsets: {offsets_str}")

        # 2. Invoke JavaParser
        cmd = ['java', '-jar', jar_path, str(original_class_path)] + offsets_str.split(',')
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✅ Refactoring successfully applied to {original_class_path.name}!")
        else:
            print(f"❌ Error executing JavaParser:\n{result.stderr}")

    except Exception as e:
        print(f"💥 Refactoring process failed: {e}")


# ==========================================
# 5. MAIN FUNCTION
# ==========================================

def main():
    # Obtain command-line arguments
    args = obtain_arguments()

    # 1. Initialize unified configuration
    config = setup_configuration(args)
    objectives, weights = parse_objectives_and_weights(config)

    # 2. Execute optimization
    general_path = run_optimization(config, objectives, weights)

    # 3. Generate plots and reports
    generate_plots_and_stats(config, objectives, general_path, args)

    # 4. Handle "solution_path" if explicitly requested by the user
    if args.get('solution_path'):
        import utils.interactive_3DPF_with_solutions as interactive_3dpf
        solution_path = Path(args['solution_path'])
        try:
            orig_class, complete, cache = classify_solution_files(solution_path)
            print("All files successfully found!")
            output_html = f"{solution_path}/{solution_path.name}_interactive_3dPF.html"
            interactive_3dpf.generate_3d_pf_and_parallel_coordinates_plot(objectives, complete, output_html, cache,
                                                                          orig_class, show=args["show"])
        except Exception as e:
            print(f"❌ Error handling solution_path: {e}")

    # # 5. Automatically refactor using the lexicographical optimum
    # if general_path and config.get('original_class'):
    #     original_class_path = Path(config['original_class'])
    #     apply_pareto_refactoring(general_path, original_class_path)


if __name__ == '__main__':
    main()