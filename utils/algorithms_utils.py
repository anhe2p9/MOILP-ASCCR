import math
import sys

import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización
import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización

import matplotlib.pyplot as plt
from ILP_CC_reducer.model import GeneralILPmodel

import numpy as np
import pandas as pd
import os
import csv
from pathlib import Path
from pymoo.indicators.hv import HV

plt.rcParams['text.usetex'] = True
model = GeneralILPmodel()

def initialize_output_data(general_path: str):
    if not os.path.exists(Path(general_path).parent):
        os.makedirs(Path(general_path).parent)

    # Save data for each solution in a CSV file
    output_filename = f"{general_path}_output.txt"

    if os.path.exists(output_filename):
        os.remove(output_filename)

    writer_output = open(output_filename, 'w', newline='')

    return writer_output

def initialize_results_file(general_path: str, objectives_list: list):
    # Save data in a CSV file
    results_filename = f"{general_path}_results.csv"

    if os.path.exists(results_filename):
        os.remove(results_filename)

    results_file = open(results_filename, 'a', newline='')
    results_writer = csv.writer(results_file)

    if Path(results_filename).stat().st_size == 0:
        results_writer.writerow([obj.__name__ for obj in objectives_list])
        results_file.flush()

    return results_writer, results_file


def add_result_to_output_data_file(concrete: pyo.ConcreteModel, objectives_list: list,
                                   new_row: tuple, output_data_writer, result):
    output_data_writer.write('===============================================================================\n')
    if result.solver.status == 'ok':
        for i, obj in enumerate(objectives_list):
            output_data_writer.write(f'Objective {obj.__name__}: {new_row[i]}\n')
        output_data_writer.write('Sequences selected:\n')
        for s in concrete.S:
            output_data_writer.write(f"x[{s}] = {concrete.x[s].value}\n")


def obtain_reference_point(concrete: pyo.ConcreteModel, objectives_list: list):
    reference_dict = {model.extractions_objective: len(concrete.S) + 1,
                      model.cc_difference_objective: concrete.nmcc[0] + 1,
                      model.loc_difference_objective: concrete.loc[0] + 1}

    reference_point = []
    for obj in objectives_list:
        reference_point.append(reference_dict[obj])

    return reference_point

def dominates(a: tuple, b: tuple) -> bool:
    """
    Returns True if point a dominates b.
    """
    return all(a[i] <= b[i] for i in range(len(a))) and any(a[i] < b[i] for i in range(len(a)))

def organize_objectives(specific_model: pyo.AbstractModel, objectives_names: list):

    if objectives_names:
        print(f"The objectives are: {objectives_names}")

        objective_map = {
            'EXTRACTIONS': specific_model.extractions_objective,
            'CC': specific_model.cc_difference_objective,
            'LOC': specific_model.loc_difference_objective
        }

        try:
            objectives_list = [objective_map[obj.upper()] for obj in objectives_names]
        except KeyError as e:
            sys.exit(f"Unknown objective '{e.args[0]}'. Objectives must be: EXTRACTIONS, CC or LOC.")
    else:
        objectives_list = None

    return objectives_list


def modify_component(mobj_model: pyo.AbstractModel, component: str, new_value: pyo.Any) -> None:
    """ Modify a given component of a model to avoid construct warnings """
    
    if hasattr(mobj_model, component):
        mobj_model.del_component(component)
    mobj_model.add_component(component, new_value)

def concrete_and_solve_model(mobj_model: pyo.AbstractModel, instance: dp.DataPortal, remaining_time: int=3600):
    """ Generates a Concrete Model for a given model instance and solves it using CPLEX solver """
    concrete = mobj_model.create_instance(instance)
    print(f"Remaining time: {remaining_time:.2f} seconds.")

    solver = pyo.SolverFactory('cplex')
    solver.options["threads"] = 1  # just 1 thread
    solver.options["timelimit"] = remaining_time

    result = solver.solve(concrete)
    return concrete, result



def print_result_and_sequences(concrete: pyo.ConcreteModel, solver_status: str, newrow: list, obj2: str=None):
    """ Print results and a vertical list of sequences selected """

    print('===============================================================================')
    if (solver_status == 'ok'):
        if obj2: # TODO: poner un for cada objetivo porque tiene que ser lo más general posible
            print(f'Objective SEQUENCES: {newrow[0]}')
            print(f'Objective {obj2}: {newrow[1]}')
        else:
            print(f'Objective SEQUENCES: {newrow[0]}')
            print(f'Objective CC_diff: {newrow[1]}')
            print(f'Objective LOC_diff: {newrow[2]}')
        print('Sequences selected:')
        for s in concrete.S:
            print(f"x[{s}] = {concrete.x[s].value}")
    print('===============================================================================')




def add_info_to_list(concrete: pyo.ConcreteModel, output_data: list, solver_status: str, obj1: str, obj2: str, newrow: list):
    """ Write results and a vertical list of selected sequences in a given file """
    
    
    if (solver_status == 'ok'):
        output_data.append(f'{obj1.__name__}: {newrow[0]}')
        output_data.append(f'{obj2.__name__}: {newrow[1]}')
        output_data.append('Sequences selected:')
        for s in concrete.S:
            output_data.append(f"x[{s}] = {concrete.x[s].value}")
    output_data.append('===============================================================================')








def generate_three_weights(n_divisions=6, theta_index=0, phi_index=0) -> tuple[float, float, float]:
    """
    Generates subdivisions in spherical coordinates for an octant.
        
    Args:
        n_divisions (int): Number of divisions in each plane (XY, XZ, YZ).
        
    Returns:
        dict: Dictionary with subdivisions in spherical coordinates.
    """
    # Crear ángulos según las divisiones
    angles = np.linspace(0.1, np.pi/2, n_divisions + 1)  # divisiones del plano
    subdivisions = {i: angles[i] for i in range(n_divisions+1)}
    
    w1, w2, w3 =  [math.sin(subdivisions[theta_index])*math.cos(subdivisions[phi_index]),
                  math.sin(subdivisions[theta_index])*math.sin(subdivisions[phi_index]),
                  math.cos(subdivisions[theta_index])]
    
    return w1, w2, w3


def generate_two_weights(n_divisions=6, theta_index=0) -> tuple[float, float]:
    """
    Generates subdivisions in polar coordinates for a quadrant.
        
    Args:
        n_divisions (int): Number of divisions in each axe (X, Y).
        
    Returns:
        dict: Dictionary with subdivisions in spherical coordinates.
    """
    # Crear ángulos según las divisiones
    angles = np.linspace(0.1, np.pi/2, n_divisions + 1)  # divisiones del plano
    subdivisions = {i: angles[i] for i in range(n_divisions+1)}
    
    w1, w2 =  [math.sin(subdivisions[theta_index]), math.cos(subdivisions[theta_index])]
    
    return w1, w2

def initialize_complete_data(general_path: str):
    if not os.path.exists(Path(general_path).parent):
        os.makedirs(Path(general_path).parent)

    # Save data for each solution in a CSV file
    complete_data_filename = f"{general_path}_complete_data.csv"

    if os.path.exists(complete_data_filename):
        os.remove(complete_data_filename)

    complete_data_file = open(complete_data_filename, 'a', newline='')
    writer_complete_data = csv.writer(complete_data_file)

    if Path(complete_data_filename).stat().st_size == 0:
        writer_complete_data.writerow(["numberOfSequences", "numberOfVariables", "numberOfConstraints",
                      "initialComplexity", "solution", "solution_info (index,CC,LOC)", "offsets", "extractions",
                      "not_nested_solution", "not_nested_extractions",
                      "nested_solution", "nested_extractions",
                      "reductionComplexity", "finalComplexity",
                      "minExtractedLOC", "maxExtractedLOC", "meanExtractedLOC", "totalExtractedLOC", "nestedLOC",
                      "minReductionOfCC", "maxReductionOfCC", "meanReductionOfCC", "totalReductionOfCC", "nestedCC",
                      "minExtractedParams", "maxExtractedParams", "meanExtractedParams", "totalExtractedParams",
                      "terminationCondition", "absoluteHypervolume", "solutionObtainingTime", "executionTime"])
        complete_data_file.flush()

    return writer_complete_data, complete_data_file


def writerow_complete_data_info(concrete: pyo.ConcreteModel, results, data,
                                solution, solution_time, hypervolume,
                                writer, file):
    """ Completes a csv with all solution data """

    complete_data_row = []

    objective_map = {
        'extractions': model.extractions_objective,
        'cc': model.cc_difference_objective,
        'loc': model.loc_difference_objective
    }

    """ Number of extractions """
    num_extractions = len([s for s in concrete.S])
    complete_data_row.append(num_extractions)

    """ Number of variables """
    num_vars_utilizadas = results.Problem[0].number_of_variables
    complete_data_row.append(num_vars_utilizadas)

    """ Number of constraints """
    num_constraints = sum(len(constraint) for constraint in concrete.component_objects(pyo.Constraint, active=True))
    complete_data_row.append(num_constraints)

    """ Initial complexity """
    initial_complexity = concrete.nmcc[0]
    complete_data_row.append(initial_complexity)

    if (results.solver.status == 'ok') and (results.solver.termination_condition == 'optimal'):
        """ Solution """
        complete_data_row.append(solution)

        """ Information about the solution """
        solution_info = [concrete.x[s].index() for s in concrete.S if concrete.x[s].value == 1 and s != 0]
        complete_data_row.append([(concrete.x[s].index(),
                                   round(pyo.value(concrete.nmcc[s] - sum(concrete.ccr[j, s] * concrete.z[j, s]
                                                                          for j,k in concrete.N if k == s))),
                                   round(pyo.value(concrete.loc[s] - sum((concrete.loc[j] - 1) * concrete.z[j, k]
                                                                         for j,k in concrete.N if k == s))))
                                  for s in concrete.S if concrete.x[s].value == 1])

        """ Offsets """
        df_csv = pd.read_csv(data["offsets"], header=None, names=["index", "start", "end"])

        # Filter by index in solution str list and obtain values
        solution_str = [str(i) for i in solution_info]
        offsets_list = df_csv[df_csv["index"].isin(solution_str)][["start", "end"]].values.tolist()

        offsets_list = [[int(start), int(end)] for start, end in offsets_list]
        complete_data_row.append(offsets_list)

        """ Extractions """
        extractions = round(pyo.value(objective_map['extractions'](concrete)))
        complete_data_row.append(extractions)

        """ Not nested solution """
        not_nested_solution = [concrete.x[s].index() for s, k in concrete.N if k == 0 and concrete.z[s, k].value != 0]
        complete_data_row.append(not_nested_solution)

        """ Not nested extractions """
        not_nested_extractions = len(not_nested_solution)
        complete_data_row.append(not_nested_extractions)

        """ Nested solution """
        nested_solution = {}

        for s, k in concrete.N:
            if concrete.z[s, k].value != 0 and k in solution_info:
                if k not in nested_solution:
                    nested_solution[k] = []  # Crear una nueva lista para cada k
                nested_solution[k].append(concrete.x[s].index())

        if len(nested_solution) != 0:
            complete_data_row.append(nested_solution)
        else:
            complete_data_row.append("")

        """ Nested extractions """
        nested_extractions = sum(len(v) for v in nested_solution.values())
        complete_data_row.append(nested_extractions)

        """ Reduction of complexity """
        cc_reduction = [(concrete.ccr[j, k] * concrete.z[j, k].value) for j, k in concrete.N if
                        k == 0 and concrete.z[j, k].value != 0]

        reduction_complexity = sum(cc_reduction)
        complete_data_row.append(reduction_complexity)

        """ Final complexity """
        final_complexity = initial_complexity - reduction_complexity
        complete_data_row.append(final_complexity)

        """ Minimum extracted LOC, Maximum extracted LOC, Mean extracted LOC, Total extracted LOC, Nested LOC """
        loc_for_each_sequence = [(concrete.loc[j] * concrete.z[j, k].value) for j, k in concrete.N if
                                 k == 0 and concrete.z[j, k].value != 0]
        if len(loc_for_each_sequence) > 0:
            min_extracted_loc = min(loc_for_each_sequence)
            complete_data_row.append(min_extracted_loc)
            max_extracted_loc = max(loc_for_each_sequence)
            complete_data_row.append(max_extracted_loc)
            mean_extracted_loc = round(float(np.mean(loc_for_each_sequence)))
            complete_data_row.append(mean_extracted_loc)
            total_extracted_loc = sum(loc_for_each_sequence)
            complete_data_row.append(total_extracted_loc)
            # NESTED LOC
            nested_loc = {}
            for v in nested_solution.values():
                for n in v:
                    nested_loc[n] = concrete.loc[n]
            if len(nested_loc) > 0:
                complete_data_row.append(nested_loc)
            else:
                complete_data_row.append("")
        else:
            for _ in range(5):
                complete_data_row.append("")

        """ Min reduction of CC, Max reduction of CC, Mean reduction of CC, Total reduction of CC, Nested CC """
        if len(cc_reduction) > 0:
            min_extracted_cc = min(cc_reduction)
            complete_data_row.append(min_extracted_cc)
            max_extracted_cc = max(cc_reduction)
            complete_data_row.append(max_extracted_cc)
            mean_extracted_cc = round(float(np.mean(cc_reduction)))
            complete_data_row.append(mean_extracted_cc)
            total_extracted_cc = initial_complexity - final_complexity
            complete_data_row.append(total_extracted_cc)
            # NESTED CC
            nested_cc = {}
            for v in nested_solution.values():
                for n in v:
                    nested_cc[n] = concrete.nmcc[n]
            if len(nested_cc) > 0:
                complete_data_row.append(nested_cc)
            else:
                complete_data_row.append("")
        else:
            for _ in range(5):
                complete_data_row.append("")

        """ Min extracted Params, Max extracted Params, Mean extracted Params, Total extracted Params """
        params_for_each_sequence = [(concrete.params[j] * concrete.z[j, k].value) for j, k in concrete.N if
                                    concrete.z[j, k].value != 0]
        if len(params_for_each_sequence) > 0:
            min_extracted_params = min(params_for_each_sequence)
            complete_data_row.append(min_extracted_params)
            max_extracted_params = max(params_for_each_sequence)
            complete_data_row.append(max_extracted_params)
            mean_extracted_params = round(float(np.mean(params_for_each_sequence)))
            complete_data_row.append(mean_extracted_params)
            total_extracted_params = sum(params_for_each_sequence)
            complete_data_row.append(total_extracted_params)
        else:
            for _ in range(4):
                complete_data_row.append("")
    else:
        for _ in range(25):
            complete_data_row.append("")

    """ Termination condition """
    complete_data_row.append(str(results.solver.termination_condition))

    """ Absolute Hypervolume """
    complete_data_row.append(hypervolume)

    """ Time for finding solution """
    complete_data_row.append(solution_time)

    """ Execution time """
    complete_data_row.append(results.solver.time)

    writer.writerow(complete_data_row)
    file.flush()

    return complete_data_row


def hypervolume_from_solutions_set(solutions_set):
    """
    Calculate the hypervolume from a set of non-dominated solutions.

    solutions_set: set of tuples with objective values
    ref_point: reference point for the hypervolume. If None, it is automatically calculated as max+1
    """
    if not solutions_set:
        return 0.0

    solutions = np.array(list(solutions_set))

    ref_point = np.max(solutions, axis=0) + 1

    hv = HV(ref_point=ref_point)
    return hv.do(solutions)