import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización
import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización

from typing import List, Tuple, Optional

import time
import sys
from math import prod

from fontTools.merge.util import current_time

import utils.algorithms_utils as algorithms_utils

from ILP_CC_reducer.algorithm.algorithm import Algorithm
from ILP_CC_reducer.model import GeneralILPmodel


PointND = Tuple[float, ...]
BoxND = Tuple[float, ...]

class HybridMethodAlgorithm(Algorithm):

    @staticmethod
    def get_name() -> str:
        return 'Hybrid Method algorithm'
    
    @staticmethod
    def get_description() -> str:
        return "It obtains supported and non-supported ILP solutions."

    @staticmethod
    def execute(data_dict: dict, tau: int, info_dict: dict):
        """
        Executes the Hybrid Method algorithm for solving multi-objective
        Integer Linear Programming (ILP) problems with an arbitrary number
        of objective functions.

        The hybrid method combines mathematical programming with a systematic
        decomposition of the objective space into axis-aligned boxes. Starting
        from an initial box defined by a reference point, the algorithm iteratively
        explores the search space by selecting the box with the largest volume.
        For each selected box, a single-objective ILP is solved by minimizing the
        sum of all objective functions subject to box-specific upper-bound
        constraints.

        If a feasible solution is found, it is added to the set of non-dominated
        solutions, recorded in the output files, and used to split the current box
        via a full p-split strategy. This splitting generates smaller sub-boxes
        that exclude the dominated region induced by the new solution. Boxes that
        are empty, redundant, or dominated by other boxes are filtered out to
        reduce the search space.

        If no feasible solution exists for a selected box, the box is discarded.
        The process continues until no boxes remain or the global time limit
        is reached.

        Throughout the execution, the algorithm enforces a global time limit,
        tracks solver performance, and stores detailed solution and execution
        information in output files.

        Parameters
        ----------
        data_dict : dict
            Dictionary containing the instance data and file paths required to
            build and solve the ILP model.
        tau : int
            Threshold parameter used in the ILP model constraints.
        info_dict : dict
            Dictionary containing execution parameters such as the number of
            objectives, objective ordering, and the global time limit.

        Returns
        -------
        None
            The total execution time and all generated non-dominated solutions
            are written to output files.
        """
        num_of_objectives = info_dict.get("num_of_objectives")
        objectives_names = info_dict.get("objectives_list")
        model = GeneralILPmodel(active_objectives=objectives_names)
        objectives_list = algorithms_utils.organize_objectives(model, objectives_names)

        general_path = data_dict["instance_folder"]

        if num_of_objectives == 2:
            if not objectives_list:  # if there is no order, the order will be [EXTRACTIONS,CC]
                objectives_list = [model.extractions_objective,
                                   model.cc_difference_objective]
        elif num_of_objectives == 3:
            if not objectives_list:  # if there is no order, the order will be [EXTRACTIONS,CC,LOC]
                objectives_list = [model.extractions_objective,
                                   model.cc_difference_objective,
                                   model.loc_difference_objective]
        else:
            sys.exit("Number of objectives for hybrid method algorithm must be 2 or 3.")

        time_limit = info_dict["time_limit"]

        output_data_writer = algorithms_utils.initialize_output_data(general_path)

        total_time = initialize_hybrid_method(model, objectives_list,
                                                         tau, data_dict,
                                                         output_data_writer, time_limit)

        output_data_writer.write('===============================================================================\n')
        output_data_writer.write(f"Total execution time: {total_time:.2f}\n")
        output_data_writer.flush()
        output_data_writer.close()

        print(f"Output correctly saved in {general_path}_output.txt.")
        print(
            "-------------------------------------------------------"
            "-------------------------------------------------------")



def initialize_hybrid_method(model: pyo.AbstractModel, objectives_list: list, tau: int,
                             data_dict: dict, output_data_writer, time_limit):
    start_total = time.time()

    data = data_dict['data']
    model.tau = pyo.Param(within=pyo.NonNegativeReals, initialize=tau, mutable=False)  # Threshold MO
    concrete = model.create_instance(data)
    reference_point = algorithms_utils.obtain_reference_point(concrete, objectives_list)
    initial_box = tuple(reference_point)
    hybrid_method_with_full_p_split(model, data_dict, objectives_list, output_data_writer,
                                    initial_box, start_total, time_limit)

    end_total = time.time()
    total_time = end_total - start_total

    return total_time


def hybrid_method_with_full_p_split(model: pyo.AbstractModel, data_dict, objectives_list, output_data_writer,
                                    initial_box: tuple, start_total, time_limit):
    general_path = data_dict["instance_folder"]
    complete_data_writer, complete_data_file = algorithms_utils.initialize_complete_data(general_path)
    results_writer, results_file = algorithms_utils.initialize_results_file(general_path, objectives_list)
    solutions_set = set()  # Non-dominated solutions set
    boxes = [initial_box]  # tuple list (u_1, ..., u_n)
    remaining_time = time_limit - (time.time() - start_total)

    print(
        "======================================================="
        "=======================================================")

    while boxes and remaining_time >= 0:

        print(f"Processing hybrid method with boxes: {boxes}.")

        def volume(box: tuple):
            return prod(box)

        idx = max(range(len(boxes)), key=lambda i: volume(boxes[i]))
        actual_box = boxes.pop(idx)
        print(f" * Selected box: {actual_box}.")

        (solution, concrete, result,
         cplex_time, solution_time) = solve_hybrid_method(model, data_dict['data'], objectives_list,
                                                              actual_box, output_data_writer,
                                                              start_total, remaining_time)

        if solution:
            solutions_set.add(solution)  # Add solution to solutions set
            print(f"New solution found: {solution}.")

            results_writer.writerow(solution)
            results_file.flush()

            output_data_writer.write(f"CPLEX time: {cplex_time}.\n")
            output_data_writer.flush()

            hypervolume = algorithms_utils.hypervolume_from_solutions_set(solutions_set)

            algorithms_utils.writerow_complete_data_info(concrete, result, data_dict, solution,
                                                         solution_time, hypervolume,
                                                         complete_data_writer, complete_data_file)

            # Split the box with the real solution found
            boxes = full_p_split(actual_box, solution, boxes)

            for i,box in enumerate(boxes):
                if inside(solution,box):
                    boxes.pop(i)
                    boxes = full_p_split(box, solution, boxes)

            print(f"GENERAL BOXES LIST: {boxes}.")
            non_dominated_boxes = filter_contained_boxes(boxes)
            print(f"NON DOMINATED BOXES: {non_dominated_boxes}.")
            boxes = non_dominated_boxes

            remaining_time = time_limit - (time.time() - start_total)

            print(
                "======================================================="
                "=======================================================")

        else:
            # No solution found -> discard box
            output_data_writer.write('======================================='
                                     '========================================\n')
            output_data_writer.write(f"SOLUTION NOT FOUND, CPLEX TIME: {cplex_time}.\n")
            output_data_writer.write('======================================='
                                     '========================================\n')
            output_data_writer.flush()

            remaining_time = time_limit - (time.time() - start_total)

            print(
                "======================================================="
                "=======================================================")

    print(
        "-------------------------------------------------------"
        "-------------------------------------------------------")
    print(f"Solutions: {solutions_set}.")
    print(
        "-------------------------------------------------------"
        "-------------------------------------------------------")

    results_file.close()
    print(f"Results correctly saved in {results_file.name}.")
    complete_data_file.close()
    print(f"Complete data correctly saved in {complete_data_file.name}.")


def solve_hybrid_method(model: pyo.AbstractModel, data: dp.DataPortal, objectives_list: list,
                        box: tuple, output_data_writer, start_total, remaining_time):

    algorithms_utils.modify_component(model, 'obj', pyo.Objective(
        rule=lambda m: sum(obj(m) for obj in objectives_list)))

    add_boxes_constraints(model, box, objectives_list)

    concrete, result = algorithms_utils.concrete_and_solve_model(model, data, remaining_time)

    if (result.solver.status == 'ok') and (result.solver.termination_condition == 'optimal') :
        new_row = tuple(round(pyo.value(obj(concrete))) for obj in objectives_list)  # Results for CSV file
        solution_time = time.time() - start_total
        algorithms_utils.add_result_to_output_data_file(concrete, objectives_list, new_row,
                                                        output_data_writer, result)

    else:
        new_row = None
        solution_time = None

    cplex_time = result.solver.time

    return new_row, concrete, result, cplex_time, solution_time


def add_boxes_constraints(model: pyo. AbstractModel, box: tuple, objectives_list: list):
    for i in range(len(objectives_list)):
        if hasattr(model, f'box_u{i+1}_constraint'):
            model.del_component(f'box_u{i+1}_constraint')

    for i, obj_func in enumerate(objectives_list):
        attr_name = f'box_u{i + 1}_constraint'
        rhs = box[i] - 1

        def make_rule(obj, r):
            return lambda m: obj(m) <= r

        setattr(
            model,
            attr_name,
            pyo.Constraint(rule=make_rule(obj_func, rhs))
        )


def full_p_split(box: BoxND, z: tuple, boxes: list) -> List[Optional[BoxND]]:
    dimensions = len(box)

    l = (0,) * dimensions
    u = box  # original box: l = (l1, ..., l_n), u = (u1, ..., u_n)
    new_boxes = []

    for i in range(dimensions):  # i = 0 (x), 1 (y), 2 (z)

        new_u = list(u)
        for j in range(i):
            new_u[j] = u[j]
        new_u[i] = max(z[i], l[i])  # cut by x_i < z_i

        is_empty = any(u[k] < z[k] for k in range(i+1, dimensions)) or new_u[i] <= l[i]

        new_boxes.append(None if is_empty else tuple(new_u))

    for i, box in enumerate(new_boxes):
        if box is not None:
            boxes.append(box)

    return boxes


def inside(a: tuple, b: tuple) -> bool:
    """
    Returns True if point 'a' is completely inside 'b'.
    """
    return all(a[i] < b[i] for i in range(len(a)))


def filter_contained_boxes(boxes: List[PointND]) -> List[PointND]:
    """
    Elimina las cajas cuyo punto superior está contenido (componente a componente)
    en otra caja de la lista.
    """
    non_dominated = []

    for i, box_i in enumerate(boxes):
        dominated = False
        for j, box_j in enumerate(boxes):
            if i == j:
                continue
            # If box_i dominates box_j
            if algorithms_utils.dominates(box_i, box_j):
                dominated = True
                print(f"Discarded box: {box_i} because it is inside box {box_j}.")
                break
        if not dominated:
            non_dominated.append(box_i)

    return non_dominated


