import os

DATASET_DIRECTORY = os.path.join(os.path.curdir, "datasets")


def students_experiments(
    name_output_dir="output",
    compute_results=[        
        "table_1",
        "figure_1",
        ],
    show_figures=True,
):
    
    import numpy as np
    import pandas as pd
    import os
    from pathlib import Path

    from divexplorer.FP_DivergenceExplorer import FP_DivergenceExplorer
    from divexplorer.FP_Divergence import FP_Divergence, abbreviateDict

    from import_datasets import import_process_compas, discretize

    from utils_print import printable

    # Check input
    mex = []

    supported_values = [
        "table_1",
        "figure_1",
    ]

    for compute_result in compute_results:

        if compute_result not in supported_values:
            mex.append(compute_result)
        if mex != []:
            mex = f"{' '.join(mex)} are not possible results, select one or more among {supported_values}"
            raise ValueError(mex)

    main_output_dir = os.path.join(os.path.curdir, name_output_dir)

    print(f"Output results in directory {main_output_dir}")

    # # Dataset
    
    abbreviations = {
        "race/ethnicity": "race/eth",
        "parental level of education": "parent edu",
        "test preparation course": "test prep",
        "math score": "math",
        "reading score": "reading",
        "writing score": "writing"
    }

    dataset_name = "compas"

    # Student Performance dataset
    # Found on Kaggle: https://www.kaggle.com/datasets/sadiajavedd/students-academic-performance-dataset








import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name_output_dir",
        default="output",
        help="specify the name of the output folder",
    )
    parser.add_argument(
        "--compute_results",
        nargs="*",
        type=str,
        default=[
            "table_1",
            "figure_1",
        ],
        help='specify the figures and tables to compute, specify one or more among ["table_1", "figure_1"]',
    )
    parser.add_argument(
        "--dataset_dir",
        default=DATASET_DIRECTORY,
        help="specify the dataset directory",
    )

    parser.add_argument(
        "--no_show_figs",
        action="store_false",
        help="specify not_show_figures to vizualize the plots. The results are stored into the specified outpur dir.",
    )

    args = parser.parse_args()

    students_experiments(
        name_output_dir=args.name_output_dir,
        compute_results=args.compute_results,
        dataset_dir=args.dataset_dir,
        show_figures=args.no_show_figs,
    )