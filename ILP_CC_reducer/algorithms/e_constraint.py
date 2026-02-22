import pyomo.environ as pyo  # ayuda a definir y resolver problemas de optimización

import utils.algorithms_utils as algorithms_utils
import sys
import time

import pandas as pd
import ast

from ILP_CC_reducer.algorithm.algorithm import Algorithm
from ILP_CC_reducer.model import GeneralILPmodel


class EpsilonConstraintAlgorithm(Algorithm):

    @staticmethod
    def get_name() -> str:
        return 'Epsilon Constraint algorithm'

    @staticmethod
    def get_description() -> str:
        return "It obtains supported and non-supported ILP solutions for two objectives using e-constraint algorithm."

    @staticmethod
    def execute(data_dict: dict, tau: int, info_dict: dict):
        """
        Executes the Augmented Epsilon-Constraint (AUGMECON) algorithm for solving
        multi-objective Integer Linear Programming (ILP) problems with two or three
        objective functions.

        For the bi-objective case (p = 2), the method applies the classical
        augmented epsilon-constraint approach. It first optimizes the secondary
        objective to obtain an initial bound, then iteratively minimizes the primary
        objective while tightening an epsilon constraint on it. At each iteration,
        non-dominated solutions are generated and added to the Pareto front until
        no further feasible solutions exist or the global time limit is reached.

        For the tri-objective case (p = 3), the method follows a grid-based
        augmented epsilon-constraint strategy. It first computes a lexicographic
        payoff table to determine lower and upper bounds for the objective functions.
        Using these bounds, a grid is constructed over the ranges of the constrained
        objectives. For each grid point, an augmented epsilon-constraint problem is
        solved, generating candidate solutions that are checked for dominance and
        used to incrementally update the Pareto front.

        Throughout the execution, the method enforces a global time limit, records
        all feasible and non-dominated solutions, and writes detailed solver and
        solution information to output files.

        Parameters
        ----------
        data_dict : dict
            Dictionary containing the instance data and file paths required to build
            and solve the ILP model.
        tau : int
            Threshold parameter used in the ILP model constraints.
        info_dict : dict
            Dictionary containing execution parameters such as the number of
            objectives, objective ordering, and the global time limit.

        Returns
        -------
        None
            The total execution time and all generated solutions are written to
            output files.
        """
        num_of_objectives = info_dict.get("num_of_objectives")
        objectives_names = info_dict.get("objectives_list")
        model = GeneralILPmodel(active_objectives=objectives_names)
        objectives_list = algorithms_utils.organize_objectives(model, objectives_names)

        general_path = data_dict["instance_folder"]
        time_limit = info_dict["time_limit"]
        start_total = time.time()

        output_data_writer = algorithms_utils.initialize_output_data(general_path)

        if num_of_objectives == 2:
            if not objectives_list:  # if there is no order, the order will be [EXTRACTIONS,CC]
                objectives_list = [model.extractions_objective,
                                   model.cc_difference_objective]
            total_time = e_constraint_2objs(data_dict, tau, objectives_list, model,
                                            output_data_writer, start_total, time_limit)
        elif num_of_objectives == 3:
            if not objectives_list:  # if there is no order, the order will be [EXTRACTIONS,CC,LOC]
                objectives_list = [model.extractions_objective,
                                   model.cc_difference_objective,
                                   model.loc_difference_objective]
            total_time = e_constraint_3objs(data_dict, tau, objectives_list, model,
                                            output_data_writer, start_total, time_limit)
        else:
            sys.exit("Number of objectives for augmented e-constraint algorithm must be 2 or 3.")

        output_data_writer.write('===============================================================================\n')
        output_data_writer.write(f"Total execution time: {total_time:.2f}\n")
        output_data_writer.flush()
        output_data_writer.close()

        print(f"Output correctly saved in {general_path}_output.txt.")
        print(
            "-------------------------------------------------------"
            "-------------------------------------------------------")


def e_constraint_2objs(data_dict: dict, tau: int, objectives_list: list, model: pyo.AbstractModel,
                       output_data_writer, start_total, time_limit):
    data = data_dict['data']
    general_path = data_dict["instance_folder"]

    complete_data_writer, complete_data_file = algorithms_utils.initialize_complete_data(general_path)
    results_writer, results_file = algorithms_utils.initialize_results_file(general_path, objectives_list)

    if not objectives_list:  # if there is no order, the order will be [EXTRACTIONS,CC]
        objectives_list = [model.extractions_objective, model.cc_difference_objective]

    obj1, obj2 = objectives_list

    solutions_set = set()

    model.tau = pyo.Param(within=pyo.NonNegativeReals, initialize=tau, mutable=False)  # Threshold

    """ z <- Solve {min f2(x) subject to x in X} """
    remaining_time = time_limit - (time.time() - start_total)
    model.obj = pyo.Objective(rule=lambda m: obj2(m))  # Objective {min f2}
    concrete, result = algorithms_utils.concrete_and_solve_model(model, data, remaining_time)

    if (result.solver.status == 'ok') and (result.solver.termination_condition == 'optimal'):
        """
        z <- Solve {min f1(x) subject to f2(x) <= f2(z)}
        """
        f2z = round(pyo.value(obj2(concrete)))  # f2(z) := f2z

        print(
            "==================================================================================================")
        print(f"min f2(x), {obj2.__name__}, subject to x in X. Result obtained: f2(z) = {round(f2z)}.")

        model.f2z = pyo.Param(
            initialize=f2z)  # new parameter f2(z) to implement new constraint f2(x) <= f2(z)
        model.f2Constraint = pyo.Constraint(
            rule=lambda m: model.second_obj_diff_constraint(m, obj2))  # new constraint: f2(x) <= f2(z)
        algorithms_utils.modify_component(model, 'obj',
                                          pyo.Objective(rule=lambda m: obj1(m)))  # new objective: min f1(x)

        """ Solve {min f1(x) subject to f2(x) <= f2(z)} """
        concrete, result = algorithms_utils.concrete_and_solve_model(model, data, remaining_time)
        model.del_component('f2Constraint')  # delete f2(x) <= f2(z) constraint

        """ FP <- {z} (add z to Pareto front) """
        new_row = tuple(round(pyo.value(obj(concrete))) for obj in objectives_list)  # Results for CSV file
        solutions_set.add(new_row)

        solution_time = time.time() - start_total

        print("=====================================")
        print(f"New solution: {new_row}.")
        print("=====================================")

        hypervolume = algorithms_utils.hypervolume_from_solutions_set(solutions_set)

        algorithms_utils.writerow_complete_data_info(concrete, result, data_dict, new_row, solution_time, hypervolume,
                                                     complete_data_writer, complete_data_file)


        algorithms_utils.add_result_to_output_data_file(concrete, objectives_list, new_row,
                                                        output_data_writer, result)

        f1z = new_row[0]

        """ epsilon <- f1(z) - 1 """
        model.epsilon = pyo.Param(initialize=f1z - 1, mutable=True)  # Epsilon parameter
        model.s = pyo.Var(within=pyo.NonNegativeReals)  # s = epsilon - f1(x)
        l1 = f1z - 1  # lower bound for f1(x)
        model.lambda_value = pyo.Param(initialize=(1 / ((f1z - l1) * 10 ** 3)),
                                                      mutable=True)  # Lambda parameter

        solution_found = (result.solver.status == 'ok') and (
                result.solver.termination_condition == 'optimal')  # while loop control

        remaining_time = time_limit - (time.time() - start_total)

        """ While exists x in X that makes f1(x) <= epsilon do """
        while solution_found and remaining_time >= 0:
            """ estimate a lambda value > 0 """
            algorithms_utils.modify_component(model, 'lambda_value',
                                              pyo.Param(initialize=(1 / ((f1z - l1) * 10 ** 3)), mutable=True))

            """ Solve epsilon constraint problem """
            """ z <- solve {min f2(x) - lambda * l, subject to f1(x) + l = epsilon} """
            algorithms_utils.modify_component(model, 'obj', pyo.Objective(
                rule=lambda m: model.epsilon_objective_2obj(m, obj2)))  # min f2(x) - lambda * l
            algorithms_utils.modify_component(model, 'epsilonConstraint', pyo.Constraint(
                rule=lambda m: model.epsilon_constraint_2obj(m, obj1)))  # f1(x) + l = epsilon

            concrete, result = algorithms_utils.concrete_and_solve_model(model, data, remaining_time)

            """ Checks if exists x in X that makes f1(x) <= epsilon (if exists solution) """
            if (result.solver.status == 'ok') and (result.solver.termination_condition == 'optimal'):
                """ PF = PF U {z} """
                f1z = pyo.value(obj1(concrete))
                new_row = tuple(round(pyo.value(obj(concrete))) for obj in objectives_list)
                solutions_set.add(new_row)

                solution_time = time.time() - start_total

                results_writer.writerow(new_row)
                results_file.flush()

                print(f"New solution: {new_row}.")
                print("=====================================")

                hypervolume = algorithms_utils.hypervolume_from_solutions_set(solutions_set)

                algorithms_utils.writerow_complete_data_info(concrete, result, data_dict, new_row,
                                                             solution_time, hypervolume,
                                                             complete_data_writer, complete_data_file)

                """ epsilon = f1(z) - 1 """
                algorithms_utils.modify_component(model, 'epsilon',
                                                  pyo.Param(initialize=f1z - 1, mutable=True))

                # lower bound for f1(x) (it has to decrease with f1z)
                l1 = f1z - 1
                algorithms_utils.add_result_to_output_data_file(concrete, objectives_list, new_row,
                                                                output_data_writer, result)

            solution_found = (result.solver.status == 'ok') and (result.solver.termination_condition == 'optimal')
            remaining_time = time_limit - (time.time() - start_total)

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

    end_total = time.time()
    total_time = end_total - start_total

    return total_time



def e_constraint_3objs(data_dict: dict, tau: int, objectives_list: list, model: pyo.AbstractModel,
                       output_data_writer, start_total, time_limit):
    data = data_dict['data']
    general_path = data_dict["instance_folder"]

    complete_data_writer, complete_data_file = algorithms_utils.initialize_complete_data(general_path)
    results_writer, results_file = algorithms_utils.initialize_results_file(general_path, objectives_list)

    p = len(objectives_list)

    model.tau = pyo.Param(within=pyo.NonNegativeReals, initialize=tau, mutable=False)  # Threshold

    """ Obtain payoff table by the lexicographic optimization of the objective functions """
    opt_lex_table_min, concrete = compute_lexicographic_table(objectives_list, model, data, time_limit, start_total)

    """ Set lower bounds ul_k for k=2...p """
    ul_point = tuple(min(col) for col in zip(*opt_lex_table_min)) # (min(f1), min(f2), min(f3))
    print(f"Lower bounds: {ul_point}.")

    """ Set upper bounds ub_k for k=2...p """
    ub_dict = {model.extractions_objective: len(concrete.S),
               model.cc_difference_objective: concrete.nmcc[0],
               model.loc_difference_objective: concrete.loc[0]}

    ub_point = []
    for obj in objectives_list:
        ub_point.append(ub_dict[obj])

    ub = tuple(ub_point)

    print(f"Upper bounds: {ub}.")

    """ Calculate ranges (r_1, ..., r_p) """
    ranges = tuple(u - l for u,l in zip(ub_point,ul_point))
    print(f"Ranges: {ranges}.")

    """ Set number of gridpoints g_k (k=2...p) for the p-1 obj.functions' ranges """
    grid_points = ranges[1:]
    print(f"Grid points: {grid_points}.")

    model.s = pyo.Var(range(p), domain=pyo.NonNegativeReals)

    solutions_set = set()

    remaining_time = time_limit - (time.time() - start_total)

    if ranges and grid_points:
        for i in reversed(range(ul_point[1], ub_point[1] + 1)):  # gridpoints f2
            j = ub_point[2]

            while j >= ul_point[2] and remaining_time >= 0:

                e_const = [i,j]
                print("=====================================")
                print(f"Iteration = {e_const}")

                concrete, result, feasible = solve_e_constraint(objectives_list, model, e_const,
                                                                data, ranges, remaining_time)
                cplex_time = result.solver.time

                if feasible:
                    new_sol = [round(pyo.value(obj(concrete))) for obj in objectives_list]

                    desired_order_for_objectives = ['extractions_objective', 'cc_difference_objective',
                                                    'loc_difference_objective']
                    objectives_dict = {obj.__name__: obj for obj in objectives_list}
                    ordered_objectives = [objectives_dict[name] for name in desired_order_for_objectives if
                                          name in objectives_dict]
                    ordered_newrow = tuple(round(pyo.value(obj(concrete))) for obj in ordered_objectives)

                    new_sol_tuple = tuple(new_sol)

                    dominated = False
                    for sol in solutions_set:
                        if algorithms_utils.dominates(sol, new_sol_tuple):
                            dominated = True

                    j = new_sol_tuple[-1] - 1

                    if dominated:
                        print(f"Dominated solution.")
                        continue

                    if new_sol_tuple not in solutions_set:
                        print(f"New solution found: {tuple(ordered_newrow)}.")

                        solution_time = time.time() - start_total

                        solutions_set = update_pareto_front_replace(solutions_set, new_sol_tuple)
                        solutions_set.add(new_sol_tuple)

                        hypervolume = algorithms_utils.hypervolume_from_solutions_set(solutions_set)

                        algorithms_utils.writerow_complete_data_info(concrete, result, data_dict, new_sol_tuple,
                                                                     solution_time, hypervolume,
                                                                     complete_data_writer, complete_data_file)

                        algorithms_utils.add_result_to_output_data_file(concrete, objectives_list, new_sol_tuple,
                                                                        output_data_writer, result)

                        results_writer.writerow(new_sol_tuple)
                        results_file.flush()

                    else:
                        print(f"Repeated solution: {tuple(ordered_newrow)}.")

                        output_data_writer.write(
                            "======================================================================================")
                        output_data_writer.write(f"Repeated solution, CPLEX TIME: {cplex_time}")
                        output_data_writer.write(
                            "======================================================================================")

                else:
                    print(f"Infeasible.")

                    output_data_writer.write(
                        "======================================================================================")
                    output_data_writer.write(f"SOLUTION NOT FOUND, CPLEX TIME: {cplex_time}")
                    output_data_writer.write(
                        "======================================================================================")

                    j = ul_point[2] - 1

                remaining_time = time_limit - (time.time() - start_total)

            if remaining_time <= 0:
                break

        print("=====================================")

        print(
            "-------------------------------------------------------"
            "-------------------------------------------------------")
        print(f"Solutions: {solutions_set}.")
        print(
            "-------------------------------------------------------"
            "-------------------------------------------------------")
    else:
        print(
            "-------------------------------------------------------"
            "-------------------------------------------------------")
        print(f"No solution found because it was not possible to create lexicographic table.")
        print(
            "-------------------------------------------------------"
            "-------------------------------------------------------")

    end_total = time.time()
    total_time = end_total - start_total

    complete_data_file.close()
    results_file.close()

    """ Rewrite results file to avoid dominated solutions when algorithm ends """
    results_writer, results_file = algorithms_utils.initialize_results_file(general_path, objectives_list)
    for sol in solutions_set:
        results_writer.writerow(sol)
        results_file.flush()
    results_file.close()
    print(f"Results correctly saved in {results_file.name}.")

    """ Rewrite complete_data file to avoid dominated solutions when algorithm ends """
    filter_csv_by_solution_set(
        csv_path=f"{general_path}_complete_data.csv",
        solution_set=solutions_set,
        solution_column="solution"
    )
    print(f"Complete data correctly saved in {complete_data_file.name}.")

    return total_time

def compute_lexicographic_table(objectives_list: list, model: pyo.AbstractModel, data, time_limit, start_total):
    opt_lex_table = []
    concrete = None

    model.obj = pyo.Objective(rule=lambda m: objectives_list[0](m))

    # For each objective fas main (f1, f2, f3)
    for main_index, main_objective in enumerate(objectives_list):

        print("-------------------------------------------------------------------------------")
        print(f"Lexicographic row for minimizing {objectives_list[main_index].__name__}")

        # List to save f1(z), f2(z), f3(z) for this row
        row_values = []

        # Lexicographic order: first the main one, then the others
        lex_order = [main_objective] + [obj for obj in objectives_list if obj != main_objective]

        # For each lexicographic level
        for level, objective in enumerate(lex_order):
            remaining_time = time_limit - (time.time() - start_total)

            # 1. Change the model objective
            def make_obj_rule(obj):
                return lambda m: obj(m)

            algorithms_utils.modify_component(model, 'obj', pyo.Objective(rule=make_obj_rule(objective)))

            # 2. Solve the model
            concrete, result = algorithms_utils.concrete_and_solve_model(model, data, remaining_time)

            if (result.solver.status == 'ok') and (result.solver.termination_condition == 'optimal'):

                # 3. Save the values vector (f1(z), f2(z), f3(z))
                current_values = [round(pyo.value(f(concrete))) for f in objectives_list]
                row_values = current_values  # always updated with actual solution

                f_nz = round(pyo.value(objective(concrete)))  # valor f_n(z)
                print(f"  Level {level + 1}: min {objective.__name__}(x) = {f_nz}")

                # 4. Add lexicographic constraint for the next level
                attr_name = f'lex_fix_{objective.__name__}_{main_index}_{level}'

                def make_rule(obj, r):
                    return lambda m: obj(m) == r

                setattr(
                    model,
                    attr_name,
                    pyo.Constraint(rule=make_rule(objective, f_nz))
                )

            else:
                print(f"  ❌ Optimization failed at level {level + 1}")
                break

        # Save this row in the payoff table
        print(f"  → Lexicographic row obtained: {tuple(row_values)}")
        opt_lex_table.append(tuple(row_values))

        # Remove new constraints after calculating each row
        for c in list(model.component_objects(pyo.Constraint, active=True)):
            if c.name.startswith("lex_fix_"):
                model.del_component(c)

    print("-------------------------------------------------------------------------------")
    print("Complete lexicographic table:")
    for row in opt_lex_table:
        print(row)
    print("-------------------------------------------------------------------------------")

    return opt_lex_table, concrete


def solve_e_constraint(objectives_list: list, model:pyo.AbstractModel, e, data, ranges, remaining_time):
    eps = 1 / (10 ** 3)

    def make_objective(obj):
        return lambda m: obj(m) - eps * sum((m.s[i]/ranges[i]) for i in range(1, len(objectives_list)))

    algorithms_utils.modify_component(model, 'obj',
                                      pyo.Objective(rule=make_objective(objectives_list[0])))

    for k, objective in enumerate(objectives_list[1:]):
        attr_name = f'f{k + 1}z_constraint_eps_problem'

        def make_rule(itr, obj, ep):
            return lambda m: obj(m) + m.s[itr + 1] == ep

        algorithms_utils.modify_component(model, attr_name, pyo.Constraint(
            rule=make_rule(k, objective, e[k])))

    concrete, result = algorithms_utils.concrete_and_solve_model(model, data, remaining_time)

    # print(f"Objective: {concrete.obj.pprint()}.")
    # print(f"Constraint1: {concrete.f1z_constraint_eps_problem.pprint()}.")
    # print(f"Constraint2: {concrete.f2z_constraint_eps_problem.pprint()}.")

    solution_found = (result.solver.status == 'ok') and (result.solver.termination_condition == 'optimal')

    return concrete, result, solution_found

def update_pareto_front_replace(pareto_front: set, new_solution: tuple):
    """
    Adds the new_solution to the pareto_front and removes any existing
    solution that is dominated by the new_solution.
    """
    # Temporary list to store solutions not dominated by the new solution
    updated_front = set()

    for sol in pareto_front:
        if not algorithms_utils.dominates(new_solution, sol):
            # Keep existing solutions that are NOT dominated by new_solution
            updated_front.add(sol)
        # Solutions dominated by new_solution are automatically removed

    # Always add the new solution
    updated_front.add(new_solution)

    return updated_front


def filter_csv_by_solution_set(csv_path, solution_set, solution_column="solution"):
    """
    Keeps only rows whose 'solution' value is in solution_set
    and overwrites the CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    solution_set : set[tuple]
        Set of valid solutions (e.g. non-dominated).
    solution_column : str
        Column name containing the solution representation.
    """
    df = pd.read_csv(csv_path)

    if df.empty:
        return

    def parse_solution(value):
        if isinstance(value, tuple):
            return value
        if isinstance(value, str):
            return tuple(ast.literal_eval(value))
        raise ValueError(f"Unsupported solution format: {value}")

    parsed_solutions = df[solution_column].apply(parse_solution)

    mask = parsed_solutions.isin(solution_set)

    df_filtered = df[mask]

    df_filtered.to_csv(csv_path, index=False)