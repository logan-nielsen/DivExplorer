import os

DATASET_DIRECTORY = os.path.join(os.path.curdir, "datasets")


def student_performance_experiments(
    name_output_dir="output",
    compute_results=["?doweneedthis"],
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