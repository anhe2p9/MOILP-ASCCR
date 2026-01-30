# Abstract
Clear and concise code is necessary to ensure maintainability, so it is crucial that the software is as simple as possible to understand, to avoid bugs and, above all, vulnerabilities. There are many ways to enhance software without changing its functionality, considering the extract method refactoring the primary process to reduce the effort required for code comprehension. The cognitive complexity measure employed in this work is the one defined by SonarSource, which is a company that develops well-known applications for static code analysis. This extraction problem can be modeled as a combinatorial optimization problem. The main difficulty arises from the existence of different criteria for evaluating the solutions obtained, requiring the formulation of the code extraction problem as a multi-objective optimization problem using alternative methods. We propose a multi-objective integer linear programming model to obtain a set of solutions that reduce the cognitive complexity of a given piece of code, such as balancing the number of lines of code and its cognitive complexity. In addition, several algorithms have been developed to validate the model. These algorithms have been integrated into a tool that enables the parameterised resolution of the problem of reducing software cognitive complexity.

# User Manual
Single and multi-objective Integer Linear Programming approach for Automatic Software Cognitive Complexity Reduction.

![Overview of the proposed multiobjective ILP CC reducer tool](ILP_CC_reducer_tool_readme.png)




# Table of Contents
- [Table of Contents](#table-of-contents)
- [ILP Model](#ilp-model-engine)
  - [Requirements](#-requirements)
  - [Download and Installation](#%EF%B8%8F-download-and-installation)
  - [Overview](#-overview)
  - [Problem Context](#-problem-context)
  - [Getting Started](#-getting-started)
    - [Arguments](#-arguments)
    - [Output Types](#-output-types)
  - [Objectives (Cognitive Complexity Metrics)](#-objectives-cognitive-complexity-metrics)
  - [Additional Script](#%EF%B8%8F-additional-script)
  - [Examples](#-examples)
  - [Project Structure](#-project-structure)



  
# ILP Model Engine

This project provides a command-line engine for solving ILP (Integer Linear Programming) problems related to reducing cognitive complexity in Java code refactorings at the method level. The tool supports solving both single-objective and multi-objective ILP instances using various algorithms and configurations.


## 📦 Requirements
- [Python 3.9+](https://www.python.org/)
- [CPLEX](https://www.ibm.com/es-es/products/ilog-cplex-optimization-studio)

The library has been tested in Linux (Mint and Ubuntu) and Windows 11.


## ⬇️ Download and Installation
1. Install [Python 3.9+](https://www.python.org/)
2. Download/Clone this repository and enter into the main directory.
3. Create a virtual environment: `python -m venv env`
4. Activate the environment: 
   
   In Linux: `source env/bin/activate`

   In Windows: `.\env\Scripts\Activate`

   ** In case that you are running Ubuntu, please install the package python3-dev with the command `sudo apt update && sudo apt install python3-dev` and update wheel and setuptools with the command `pip  install --upgrade pip wheel setuptools` right after step 4.
   
5. Install the dependencies: `pip install -r requirements.txt`



## 💡 Overview

The main purpose of this application is to automate the generation and resolution of ILP models designed to **optimize code refactorings**. The models aim to **minimize**:

1. The number of extractions (refactorings).
2. The difference between the maximum and minimum **cognitive complexity** across all resulting sequences.
3. The difference between the maximum and minimum **lines of code** across all resulting sequences.

## 🧠 Problem Context

Given a Java method and its corresponding **refactoring cache**, the system builds ILP model instances to explore different refactoring strategies. These models are then solved with one of several available algorithms.

The results may vary:
- A **.csv file** and a corresponding **.lp file** for **single-objective** ILP.
- For **multi-objective** problems (2 or 3 objectives), a set of solutions using:
  - **Weighted sum** (single or multiple combinations).
  - **Augmented ε-constraint** (for two objectives).
  - A **hybrid algorithm** (for three objectives), exploring the entire objective space to generate an approximate Pareto front.

---

## 🚀 Getting Started

To run the main engine:

```bash
python main.py [OPTIONS]
```


## 🔧 Arguments

| Argument                    | Description                                                                                                                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `-f`, `--file`              | Path to a `.ini` file containing all parameters.                                                                                                                                                  |
| `-n`, `--num_of_objectives` | Number of objectives to be considered: 1, 2 or 3.                                                                                                                                                 |
| `-i`, `--instance`          | Path to the model instance. It can be the folder path with the three data files in CSV format for multiobjective or the general folder path with all instances for one objective.                 |
| `-a`, `--algorithm`         | Algorithm to use for solving single and multiobjective ILP problems. Must be one of: `['ObtainResultsAlgorithm', 'WeightedSumAlgorithm', 'EpsilonConstraintAlgorithm', 'HybridMethodAlgorithm']`. |
| `-t`, `--tau`               | Threshold (τ) used in optimization (e.g., for ε-constraint).                                                                                                                                      |
| `-s`, `--subdivisions`      | (Optional) Number of subdivisions for generating weighted combinations.                                                                                                                           |
| `-w`, `--weights`           | (Optional) Specific weights for weighted sum: `w1,w2` or `w1,w2,w3`.                                                                                                                              |
| `-o`, `--objectives`        | (Optional) Objectives to minimize: `obj1,obj2` or `obj1,obj2,obj3`.                                                                                                                               |
| `--model`                   | (Optional) FOR ONE OBJECTIVE: it gives back the ILP model.                                                                                                                                        |
| `--solve`                   | (Optional) FOR ONE OBJECTIVE: it tries to solve the model.                                                                                                                                        |
| `--plot`                    | (Optional) Plot the result of a specific experiment.                                                                                                                                              |
| `--3dPF`                    | (Optional) Plot the 3D PF of the given result.                                                                                                                                                    |
| `--relHV`                   | (Optional) Plots the relative HV with respect time of the given result.                                                                                                                           |
| `--all_plots`               | (Optional) Plot all results in the given directory.                                                                                                                                               |
| `--all_3dPF`                | (Optional) Plot all 3D PFs in a given directory.                                                                                                                                                  |
| `--all_relHV`               | (Optional) Plots all relative HVs with respect time in a given directory.                                                                                                                         |
| `--statistics`              | (Optional) Generate a CSV file with statistics for all results in the given directory.                                                                                                            |
| `--input`                   | (Optional) Input directory for results (used for plotting/statistics). Defaults to `output/results`.                                                                                              |
| `--output`                  | (Optional) Output directory for plots/statistics. Defaults to `output/plots_and_statistics`.                                                                                                      |
| `--save`                    | (Optional) Save current configuration to a `.ini` file.                                                                                                                                           |



---

## 🧪 Output Types

Depending on the model and input configuration, the application can generate:

- For **single-objective ILP**:
  - A `.csv` file containing the solution.
  - An `.lp` file representing the ILP model for each instance.

- For **multi-objective ILP** (2 or 3 objectives):
  - A set of non-dominated solutions (Pareto front), either:
    - via **weighted sum**:
      - using a sweep over weight combinations (`--subdivisions`), or
      - with a specific combination (`--weights`).
    - via **augmented ε-constraint** algorithm (`--algorithm`):
      - CSV file with results.
      - Concrete model.
      - Output data with the solution for each Java method.
      - Plot in case of requested.
    - via a **hybrid objective-space exploration algorithm**:
      - CSV file with results.
      - Concrete model.
      - Output data with the solution for each Java method.
      - Nadir point.
      - Plot in case of requested.

---

## 🧠 Objectives (Cognitive Complexity Metrics)

The optimization process focuses on refactoring Java methods by minimizing the following cognitive complexity metrics:

1. **Number of Extractions**:  
   The total number of code extractions performed.  
   _Goal_: Minimize to keep changes limited.

2. **Complexity Range**:  
   Difference between the highest and lowest cognitive complexity values among extracted sequences.  
   _Goal_: Minimize to ensure balanced complexity across parts.

3. **Lines of Code Range**:  
   Difference between the largest and smallest number of lines of code among extracted parts.  
   _Goal_: Minimize to obtain balanced code lengths.

---

## 🗂️ Additional Scripts

- **`input_files_main.py`** (located in **ILP_data_from_refactoring_cache**)  
  - Generates ILP input files from a refactoring cache.  
  - Arguments:  
    - `input_folder`  
    - `output_folder` (optional, to save results in a different folder than the input)

- **`new_refactoring_names_assignation.py`** *(new)*  
  - Automatically assigns names to refactored code.  
  - Arguments:  
    - `input_folder`  
    - `output_folder` (optional, to save results in a different folder than the input)  
  - ⚠️ The automatic creation of refactored code from solutions is still **under development** and will be fully automated soon.

- **`run_all_instances.ps1`** *(new)*  
  - Executes `main.py` for **all problem instances contained in a folder hierarchy** with the following structure:  
    ```
    root_folder → projects → classes → methods
    ```
    Each **method folder** is treated as an independent instance and passed as the `-i` argument to `main.py`.  
  - This script is designed for **Windows environments** using **PowerShell**.  
  - Arguments:  
    - `root_folder` (path to the root directory containing all projects)  
    - `Algorithm` (algorithm desired, see [Arguments](#-arguments))
  - Internally executes the following command for each method folder:
    ```
    python main.py -n 3 -i <method_folder> -a DesiredAlgorithm -t 15 -tl 120 --plot --3dPF --relHV
    ```
  - Useful for batch execution of experiments across multiple projects, classes, and instances without manual intervention.
---

## 📘 Examples

This command generates the input needed for the main module of the tool:
```bash
python input_files_main.py ./input_folder ./output_folder
```


This command generates the solution for two-objectives ILP problem with weighted sum algorithm for two objectives:
```bash
python main.py -n 2 -i ./instances/my_instance -a WeightedSumAlgorithm -t 2 -s 6 -o extractions,cc
```


This command generates the solution for three-objectives ILP problem with hybrid method algorithm for three objectives, and it also generates the parallel coordinates plot and the complete Pareto front in three dimensions:
```bash
python main.py -n 3 -i ./instances/my_instance -a HybirdMethodAlgorithm -t 15 -o extractions,cc,loc --plot --3dPF
```





## 📂 Project Structure
    📁 M2I-TFM-Adriana  
    ├── 📁 ILP_CC_reducer  
    │   ├── 📁 algorithm  
    │   │   ├── __init__.py  
    │   │   └── algorithm.py  
    │   ├── 📁 algorithms  
    │   │   ├── __init__.py
    │   │   ├── obtain_results.py  
    │   │   ├── weighted_sum.py 
    │   │   ├── e_constraint.py  
    │   │   ├── hybrid_method.py
    │   ├── 📁 model  
    │   │   ├── __init__.py  
    │   │   ├── ILPmodel.py
    │   └── 📁 operations  
    │       ├── __init__.py  
    │       └── ILP_engine.py  
    ├── 📁 ILP_data_from_refactoring_cache  
    │   ├── 📁 dataset_refactoring_caches  
    │   ├── 📁 utils  
    │   │   ├── dataset.py  
    │   │   ├── offsets.py  
    │   │   └── refactoring_cache.py  
    │   ├── __init__.py  
    │   ├── input_files_main.py  
    │   └── README.md  
    ├── general_utils.py  
    ├── main.py  
    ├── README.md  
    └── requirements.txt  
