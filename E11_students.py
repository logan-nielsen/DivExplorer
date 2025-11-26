import os

DATASET_DIRECTORY = os.path.join(os.path.curdir, "datasets")


def student_performance_experiments(
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