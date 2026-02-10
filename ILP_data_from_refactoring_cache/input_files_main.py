import sys
import os

# Add base directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the required libraries
from ILP_data_from_refactoring_cache.utils import dataset as dataset, refactoring_cache as rc
import argparse
import zipfile
from pathlib import Path
import re

# Main function
def main(path_to_refactoring_cache: str, output_folder: str, files_n: str):
    # Filter feasible extractions and assign id to extractions
    df = rc.set_extractions_id(dataset.dataframe_from_csv_file(path_to_refactoring_cache))

    # Save the mapping between feasible extractions and offsets into a CSV file
    dataset.dataframe_into_csv_file(
        rc.get_extractions_including_given_columns(df, ["A", "B"]), output_folder + f"/{files_n}_feasible_extractions_offsets.csv")

    # Save the extractions in conflict into a CSV file
    dataset.dataframe_into_csv_file(rc.get_conflicts(df), output_folder + f"/{files_n}_conflict.csv")

    # Save the lines of code, cognitive complexity, and number of parameters of the extractions into a CSV file
    dataset.dataframe_into_csv_file(rc.get_extractions_including_given_columns(df, ["extractedLOC", "extractedMethodCC", "parameters"]),
                                    output_folder + f"/{files_n}_sequences.csv")

    # Save the nested extractions into a CSV file
    dataset.dataframe_into_csv_file(rc.get_nested_extraction_for_each_extraction_computing_ccr(df),
                                    output_folder + f"/{files_n}_nested.csv")
    
    
    
def extract_class_method(file_name):
    """
       Extracts:
         - class_name: folder-package-class
         - method_name: method-line
       from the format:
         project@folder-package-clasa-method-line
       """
    match = re.search(
        r'@(.+?)-([^-]+)-(\d+)\.csv$',
        file_name
    )

    if not match:
        return None, None

    class_name = match.group(1)  # folder-package-class
    method_name = f"{match.group(2)}-{match.group(3)}"  # method-line

    return class_name, method_name




# Call the main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process refactoring cache and output results.')
    parser.add_argument('--input', dest='input_path', type=str,
                        help='Path to input folder or zip file')
    parser.add_argument('--output', dest='output_folder', type=str,
                        help='Folder to save the output CSV files')
    args = parser.parse_args()

    input_path = Path(args.input_path)

    # Base directory of the project
    base_dir = Path(__file__).resolve().parent.parent

    # -------- CASE 1: INPUT IS A FOLDER --------
    if input_path.is_dir():
        files_iter = (f for f in input_path.iterdir() if f.is_file())
        zip_file = None

    # -------- CASE 2: INPUT IS A ZIP --------
    elif zipfile.is_zipfile(input_path):
        zip_file = zipfile.ZipFile(input_path, 'r')
        files_iter = (
            f for f in zip_file.namelist()
            if not f.endswith('/')
               and not f.startswith('__MACOSX/')
        )

    else:
        raise ValueError(f"Input path is neither a folder nor a zip file: {input_path}")

    # -------- COMMON PROCESSING --------
    for file in files_iter:

        # Folder → Path object
        if zip_file is None:
            file_name = file.name
            file_ref = file  # Path, as before

        # Zip → string inside the zip
        else:
            if file.endswith('/'):
                continue  # skip directories inside zip
            file_name = Path(file).name
            file_ref = file  # name inside zip

        file_class, file_method = extract_class_method(file_name)

        # Determine contents folder for results
        if args.output_folder:
            output_base = Path(args.output_folder)
        else:
            # Folder for input + subfolder 'results'
            if input_path.is_file():
                output_base = input_path.parent / "results"
            else:
                output_base = input_path / "results"

        # Make sure that it exists
        output_base.mkdir(parents=True, exist_ok=True)

        print(f"All results will be saved in: {output_base}")

        # -------- COMMON PROCESSING --------
        for file in files_iter:

            # Folder → Path object
            if zip_file is None:
                file_name = file.name
                file_ref = file  # Path, as before

            # Zip → string inside the zip
            else:
                if file.endswith('/'):
                    continue  # skip directories inside zip
                file_name = Path(file).name
                file_ref = file  # name inside zip

            file_class, file_method = extract_class_method(file_name)

            if file_class is None or file_method is None:
                print(f"Skipping file (pattern mismatch): {file_name}")
                continue

            # Build output directory inside the container folder
            output_dir = output_base / file_class / file_method
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"Processing {file_class} class, and {file_method} method.")

            # Folder vs zip handling for main()
            if zip_file is None:
                main(file_ref, str(output_dir), file_method)
            else:
                with zip_file.open(file_ref) as f:
                    main(f, str(output_dir), file_method)

            print(f"New data is available in: {output_dir}")
            print("--------------------------------------------------------------------------------------------")