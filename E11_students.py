import os
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from import_datasets import (
    import_process_students,
    train_predict,
    discretize,
)
from divexplorer.FP_DivergenceExplorer import FP_DivergenceExplorer
from divexplorer.FP_Divergence import FP_Divergence, abbreviateDict
from utils_print import (
    printable,
    printableAll,
    printableCorrective,
)

DATASET_DIRECTORY = os.path.join(os.path.curdir, "datasets")


def students_experiments(
    name_output_dir="output",
    compute_results=[
        "students_table_1",
        "students_figure_1",
        "students_table_2",
        "students_figure_2",
        "students_figure_3",
    ],
    dataset_dir=DATASET_DIRECTORY,
    show_figures=True,
    min_support=0.05,
):
    """
    Run DivExplorer experiments on the StudentsPerformance dataset.

    Pipeline:
      1. Load and preprocess dataset (import_process_students)
      2. Train a classifier on demographic features (via train_predict)
      3. Discretize attributes for pattern mining (discretize)
      4. Extract frequent-pattern divergence (FP_DivergenceExplorer)
      5. Generate:
         - students_table_1      : top-k patterns for FPR/FNR/error/accuracy
         - students_figure_1     : bar plot of top-k d_fpr
         - students_table_2      : top corrective items for FPR and FNR
         - students_figure_2     : Shapley breakdown of most FPR-divergent pattern
         - students_figure_3     : Comparison of computed vs approximate Shapley Values of most FPR-divergent pattern
    """

    # ------------------------------------------------------------------
    # 0. Validate compute_results
    # ------------------------------------------------------------------
    supported_values = [
        "students_table_1",
        "students_figure_1",
        "students_table_2",
        "students_figure_2",
        "students_figure_3",
    ]

    invalid = [cr for cr in compute_results if cr not in supported_values]
    if invalid:
        msg = (
            f"{' '.join(invalid)} are not valid compute_results; "
            f"choose one or more of {supported_values}"
        )
        raise ValueError(msg)

    main_output_dir = os.path.join(os.path.curdir, name_output_dir)
    print(f"Output results in directory {main_output_dir}")

    # ------------------------------------------------------------------
    # 1. Load dataset and define label
    # ------------------------------------------------------------------
    dataset_name = "students"

    dfI, class_map = import_process_students(discretize=False, inputDir=dataset_dir)
    dfI.reset_index(drop=True, inplace=True)

    print("StudentsPerformance dataset loaded.")
    print(f"Rows: {len(dfI)}, Columns: {len(dfI.columns)}")
    print("Class distribution (dfI['class']):")
    print(dfI["class"].value_counts())

    abbreviations = {
        "gender": "gender",
        "race/ethnicity": "race",
        "parental level of education": "parent_ed",
        "lunch": "lunch",
        "test preparation course": "test_prep",
    }

    # ------------------------------------------------------------------
    # 2. Train classifier and get predicted labels
    # ------------------------------------------------------------------
    # Restricted to non-score features
    feature_columns = [
        "gender",
        "race/ethnicity",
        "parental level of education",
        "lunch",
        "test preparation course",
    ]
    df_for_model = dfI[feature_columns + ["class"]]

    (
        X_FP,
        y_FP,
        y_predicted,
        y_predict_prob,
        encoders,
        indexes_FP,
    ) = train_predict(
        df_for_model,
        type_cl="RF",
        labelEncoding=True,
        validation="all", 
        k_cv=10,
        args={},
        retClf=False,
        fold="stratified",
    )

    # Build the dataframe used for divergence analysis:
    df_FP = X_FP.copy()
    df_FP["class"] = y_FP.values
    df_FP["predicted"] = y_predicted

    # ------------------------------------------------------------------
    # 3. Discretize attributes for pattern mining
    # ------------------------------------------------------------------
    attributes = df_FP.columns.drop(["class", "predicted"])

    X_discretized = discretize(
        df_FP,
        bins=4,
        dataset_name=dataset_name,
        attributes=attributes,
        indexes_FP=indexes_FP,
    )

    X_discretized["class"] = y_FP.values
    X_discretized["predicted"] = y_predicted

    # ------------------------------------------------------------------
    # 4. Run frequent pattern divergence analysis
    # ------------------------------------------------------------------
    fp_diver = FP_DivergenceExplorer(
        X_discretized,
        "class",
        "predicted",
        class_map=class_map,
        log_loss_values=y_predict_prob,
        clf=None,
        dataset_name=dataset_name,
        type_cl="RF",
    )

    FP_fm = fp_diver.getFrequentPatternDivergence(
        min_support=min_support,
        metrics=["d_fpr", "d_fnr", "d_error", "d_accuracy"],
    )

    # Save full/all patterns and top-20 by |d_fpr|
    tables_dir = os.path.join(main_output_dir, "tables")
    figures_dir = os.path.join(main_output_dir, "figures")
    Path(tables_dir).mkdir(parents=True, exist_ok=True)
    Path(figures_dir).mkdir(parents=True, exist_ok=True)

    # Overall dataset FPR/FNR
    overall_row = FP_fm.loc[FP_fm.itemsets == frozenset()]
    if not overall_row.empty:
        fpr_dataset, fnr_dataset = overall_row[["fpr", "fnr"]].values[0]
        print(
            f"\nOverall dataset metrics: FPR={fpr_dataset:.3f}, FNR={fnr_dataset:.3f}"
        )
    else:
        fpr_dataset = fnr_dataset = np.nan
        print("\nNo overall (empty) pattern row found; something is off.")

    # Top-20 by |d_fpr|
    patterns = FP_fm[FP_fm["length"] > 0].copy()
    patterns["abs_d_fpr"] = patterns["d_fpr"].abs()
    patterns_top20 = (
        patterns.sort_values("abs_d_fpr", ascending=False)
        .head(20)
        .drop(columns=["abs_d_fpr"])
    )

    all_patterns_path = os.path.join(tables_dir, "students_patterns_all.csv")
    top20_path = os.path.join(tables_dir, "students_patterns_top20.csv")
    FP_fm.to_csv(all_patterns_path, index=False)
    patterns_top20.to_csv(top20_path, index=False)

    print(f"\nSaved all patterns to: {all_patterns_path}")
    print(f"Saved top-20 patterns to: {top20_path}")

    # ------------------------------------------------------------------
    # 5. students_table_1 – top patterns for FPR/FNR/error/accuracy
    # ------------------------------------------------------------------
    if "students_table_1" in compute_results:
        n_rows = 5 

        fp_divergence_fpr = FP_Divergence(FP_fm, "d_fpr")
        div_fpr = fp_divergence_fpr.getDivergence(th_redundancy=0)[
            [
                "support",
                "itemsets",
                fp_divergence_fpr.metric,
                fp_divergence_fpr.t_value_col,
            ]
        ]
        div_pr_fpr = printable(div_fpr.head(n_rows), abbreviations=abbreviations)

        fp_divergence_fnr = FP_Divergence(FP_fm, "d_fnr")
        div_fnr = fp_divergence_fnr.getDivergence(th_redundancy=0)[
            [
                "support",
                "itemsets",
                fp_divergence_fnr.metric,
                fp_divergence_fnr.t_value_col,
            ]
        ]
        div_pr_fnr = printable(div_fnr.head(n_rows), abbreviations=abbreviations)

        fp_divergence_error = FP_Divergence(FP_fm, "d_error")
        div_error = fp_divergence_error.getDivergence(th_redundancy=0)[
            [
                "support",
                "itemsets",
                fp_divergence_error.metric,
                fp_divergence_error.t_value_col,
            ]
        ]
        div_pr_error = printable(div_error.head(n_rows), abbreviations=abbreviations)

        fp_divergence_accuracy = FP_Divergence(FP_fm, "d_accuracy")
        div_accuracy = fp_divergence_accuracy.getDivergence(th_redundancy=0)[
            [
                "support",
                "itemsets",
                fp_divergence_accuracy.metric,
                fp_divergence_accuracy.t_value_col,
            ]
        ]
        div_pr_accuracy = printable(
            div_accuracy.head(n_rows), abbreviations=abbreviations
        )

        dfs = [div_pr_fpr, div_pr_fnr, div_pr_error, div_pr_accuracy]
        div_all = printableAll(dfs)

        print("-----------------------------------------------------------------------")
        print(div_all)
        caption_str = (
            f"students_table_1: Top-{n_rows} divergent patterns for FPR, FNR, "
            f"error rate (ER), and accuracy (ACC) on the StudentsPerformance "
            f"dataset (min_support={min_support})."
        )
        print(caption_str)

        filename = os.path.join(tables_dir, "students_table_1.csv")
        div_all.to_csv(filename, index=False)

    # ------------------------------------------------------------------
    # 6. students_figure_1 – bar plot of top-20 d_fpr
    # ------------------------------------------------------------------
    if "students_figure_1" in compute_results and not patterns_top20.empty:
        try:
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_subplot(111)

            patterns_plot = patterns_top20.copy()
            patterns_plot["itemsets_str"] = patterns_plot["itemsets"].apply(
                lambda s: ", ".join(sorted(list(s)))
            )

            ax.bar(
                patterns_plot["itemsets_str"],
                patterns_plot["d_fpr"],
            )
            ax.set_xticklabels(
                patterns_plot["itemsets_str"],
                rotation=45,
                ha="right",
                fontsize=8,
            )
            ax.set_ylabel("d_fpr")
            ax.set_title(
                "students_figure_1: Top-20 patterns by d_fpr (StudentsPerformance)"
            )

            fig.tight_layout()
            fig_path = os.path.join(figures_dir, "students_figure_1_d_fpr_top20.pdf")
            fig.savefig(fig_path)
            print(f"Saved students_figure_1 to: {fig_path}")
            plt.close(fig)
        except Exception as e:
            print(f"Could not generate students_figure_1: {e}")

    # ------------------------------------------------------------------
    # 7. students_table_2 – top corrective items for FPR and FNR
    # ------------------------------------------------------------------
    if "students_table_2" in compute_results:
        fp_divergence_fpr = FP_Divergence(FP_fm, "d_fpr")
        corrective_fpr = fp_divergence_fpr.getCorrectiveItems()
        corrective_pr_fpr = printableCorrective(
            corrective_fpr.head(3).reset_index(drop=True),
            fp_divergence_fpr.metric_name,
            abbreviations=abbreviations,
        )

        fp_divergence_fnr = FP_Divergence(FP_fm, "d_fnr")
        corrective_fnr = fp_divergence_fnr.getCorrectiveItems()
        corrective_pr_fnr = printableCorrective(
            corrective_fnr.head(3).reset_index(drop=True),
            fp_divergence_fnr.metric_name,
            abbreviations=abbreviations,
        )

        df_merged = printableAll(
            [corrective_pr_fpr, corrective_pr_fnr], rename_cols=False
        )

        print("-----------------------------------------------------------------------")
        print(df_merged)
        caption_str = (
            "students_table_2: Top corrective items for FPR and FNR on "
            "StudentsPerformance dataset."
        )
        print(caption_str)

        filename = os.path.join(tables_dir, "students_table_2.csv")
        df_merged.to_csv(filename, index=False)

    # ------------------------------------------------------------------
    # 8. students_figure_2 – Shapley for most FPR-divergent pattern
    # ------------------------------------------------------------------
    if "students_figure_2" in compute_results:
        fp_divergence_fpr = FP_Divergence(FP_fm, "d_fpr")

        # most FPR-divergent pattern
        top_itemset = list(
            fp_divergence_fpr.getDivergenceTopK(K=1, th_redundancy=0).keys()
        )[0]

        shap_values = fp_divergence_fpr.computeShapleyValue(top_itemset)
        shap_values = abbreviateDict(shap_values, abbreviations)

        output_file_name = os.path.join(figures_dir, "students_figure_2_shapley.pdf")

        print("-----------------------------------------------------------------------")
        print(f"Top FPR-divergent pattern: {top_itemset}")
        print("Shapley contributions:", shap_values)

        try:
            import matplotlib.pyplot as plt

            keys = list(shap_values.keys())
            vals = list(shap_values.values())

            fig, ax = plt.subplots(figsize=(6, 3))
            ax.barh(range(len(keys)), vals)
            ax.set_yticks(range(len(keys)))
            ax.set_yticklabels(keys)
            ax.set_xlabel("Shapley contribution to d_fpr")
            ax.set_title(
                "students_figure_2: Shapley contributions for most FPR-divergent pattern"
            )
            fig.tight_layout()
            fig.savefig(output_file_name, bbox_inches="tight")
            if show_figures:
                plt.show()
            else:
                plt.close(fig)

            print(f"Saved students_figure_2 to: {output_file_name}")
        except Exception as e:
            print(f"Could not generate students_figure_2: {e}")

    # ------------------------------------------------------------------
    # 9. students_figure_3 – Computed VS Approximated Shapley Values for most FPR-divergent pattern
    # ------------------------------------------------------------------
    if "students_figure_3" in compute_results:
        fp_divergence_fpr = FP_Divergence(FP_fm, "d_fpr")

        # most FPR-divergent pattern
        top_itemset = list(
            fp_divergence_fpr.getDivergenceTopK(K=1, th_redundancy=0).keys()
        )[0]

        shap_values = fp_divergence_fpr.computeShapleyValue(top_itemset)
        shap_values = abbreviateDict(shap_values, abbreviations)

        shap_values_approx = fp_divergence_fpr.computeShapleyValue(top_itemset, approximate=True)
        shap_values_approx = abbreviateDict(shap_values_approx, abbreviations)

        output_file_name = os.path.join(figures_dir, "students_figure_3_shapley.pdf")

        print("-----------------------------------------------------------------------")
        print(f"Top FPR-divergent pattern: {top_itemset}")
        print("Computed Shapley contributions:", shap_values)
        print("Approximate Shapley contributions:", shap_values_approx)

        try:
            import matplotlib.pyplot as plt

            keys = list(shap_values_approx.keys())
            vals = list(shap_values_approx.values())

            fig, ax = plt.subplots(figsize=(6, 3))
            ax.barh(range(len(keys)), vals)
            ax.set_yticks(range(len(keys)))
            ax.set_yticklabels(keys)
            ax.set_xlabel("Shapley contribution to d_fpr")
            ax.set_title(
                "students_figure_3: Approximate Shapley contributions for most FPR-divergent pattern"
            )
            fig.tight_layout()
            fig.savefig(output_file_name, bbox_inches="tight")
            if show_figures:
                plt.show()
            else:
                plt.close(fig)

            print(f"Saved students_figure_3 to: {output_file_name}")
        except Exception as e:
            print(f"Could not generate students_figure_3: {e}")


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
            "students_table_1",
            "students_figure_1",
            "students_table_2",
            "students_figure_2",
            "students_figure_3"
        ],
        help=(
            "specify the figures and tables to compute, choose one or more among "
            '["students_table_1", "students_figure_1", '
            '"students_table_2", "students_figure_2", '
            '"students_figure_3"]'
        ),
    )
    parser.add_argument(
        "--dataset_dir",
        default=DATASET_DIRECTORY,
        help="specify the dataset directory",
    )
    parser.add_argument(
        "--no_show_figs",
        action="store_false",
        help=(
            "if set, figures are not shown interactively (only saved to disk). "
            "The results are stored into the specified output dir."
        ),
    )
    parser.add_argument(
        "--min_support",
        type=float,
        default=0.05,
        help="minimum support threshold for frequent patterns",
    )

    args = parser.parse_args()

    students_experiments(
        name_output_dir=args.name_output_dir,
        compute_results=args.compute_results,
        dataset_dir=args.dataset_dir,
        show_figures=args.no_show_figs,
        min_support=args.min_support,
    )
