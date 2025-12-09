import os

DATASET_DIRECTORY = os.path.join(os.path.curdir, "datasets")


def artificial_teams_experiments(
    name_output_dir="output",
    compute_results=["artificial_teams_figure_1"],
    show_figures=True,
):
    """
    Run DivExplorer experiments on an artificial player stats dataset.

    Pipeline:
      1. Create artificial dataset
      2. Discretize attributes for pattern mining
      3. Create false negatives in dataset
      4. Extract frequent-pattern divergence (FP_DivergenceExplorer)
      5. Generate:
         - artificial_teams_figure_1.pdf
    """

    import numpy as np
    import pandas as pd
    import os
    from pathlib import Path

    from divexplorer.FP_DivergenceExplorer import FP_DivergenceExplorer
    from divexplorer.FP_Divergence import FP_Divergence
    from divexplorer.shapley_value_FPx import compareShapleyValues, normalizeMax

    # ------------------------------------------------------------------
    # 0. Validate compute_results
    # ------------------------------------------------------------------

    supported_values = [
        "artificial_teams_figure_1",
    ]

    invalid = [cr for cr in compute_results if cr not in supported_values]
    if invalid:
        msg = (
            f"{' '.join(invalid)} are not valid compute_results; "
            f"choose one or more of {supported_values}"
        )
        raise ValueError(msg)

    main_output_dir = os.path.join(os.path.curdir, name_output_dir)

    # ------------------------------------------------------------------
    # 1. Create artificial dataset for player stats
    # ------------------------------------------------------------------

    dataset_name = "artificial_teams"
    features = ["player_id", "games_won", "games_lost", "attendance"]

    numPlayers = 5000

    np.random.seed(7)
    df_artificial_teams = pd.DataFrame({
        features[0]: np.arange(1, numPlayers + 1),
        features[1]: np.random.randint(0, 50, numPlayers),
        features[2]: np.random.randint(0, 50, numPlayers),
        features[3]: np.random.randint(0, 50, numPlayers),
    })

    # Calculate player avg stats
    df_artificial_teams["player_avg"] = df_artificial_teams[["games_won", "games_lost", "attendance"]].mean(axis=1)

    # Determine if player avg stats is above avg for all players
    global_avg = df_artificial_teams["player_avg"].mean()

    df_artificial_teams["class"] = (df_artificial_teams["player_avg"] > global_avg).astype(int)

    # Discretize stats in low/med/high bins
    df_artificial_teams["wins_bin"] = pd.cut(df_artificial_teams["games_won"], bins=[-1, 10, 20, 30], labels=["low", "mid", "high"])
    df_artificial_teams["loss_bin"] = pd.cut(df_artificial_teams["games_lost"], bins=[-1, 10, 20, 30], labels=["low", "mid", "high"])
    df_artificial_teams["att_bin"] = pd.cut(df_artificial_teams["attendance"], bins=[-1, 15, 30, 50], labels=["low", "mid", "high"])
    df_artificial_teams["avg_bin"] = pd.cut(df_artificial_teams["player_avg"], bins=3, labels=["low", "mid", "high"])

    # We add FP errors
    import random

    indexes = df_artificial_teams.index[df_artificial_teams["class"] == 1]
    indexes_class1 = list(indexes)
    random.Random(7).shuffle(indexes_class1)
    df_artificial_teams["predicted"] = df_artificial_teams["class"]
    biased_group = df_artificial_teams[
        (df_artificial_teams["wins_bin"] == "high") &
        (df_artificial_teams["att_bin"] == "low") &
        (df_artificial_teams["class"] == 1)
    ].index

    # Flip 60% of them to false negatives
    n_flip = int(0.8 * len(biased_group))

    df_artificial_teams.loc[
        biased_group[:n_flip], "predicted"
    ] = 0

    class_map = {"N": 0, "P": 1}

    # # Extract divergence

    # Input:
    # - discretized dataframe
    # - true class column name
    # - predicted class column name

    # Parameters: minimum support of the extracted patterns
    min_sup = 0.01

    fp_diver = FP_DivergenceExplorer(
        df_artificial_teams,
        "class",
        "predicted",
        class_map=class_map,
        dataset_name=dataset_name,
    )
    FP_fm = fp_diver.getFrequentPatternDivergence(
        min_support=min_sup, metrics=["d_fpr", "d_fnr"]
    )

    # Frequent pattern divergence extraction
    fp_divergence_fpr = FP_Divergence(FP_fm, "d_fpr")

    if "artificial_teams_figure_1" in compute_results:

        from divexplorer.FP_Divergence import FP_Divergence

        # Derive the divergence w.r.t. the FPR

        fp_divergence_fnr = FP_Divergence(FP_fm, "d_fnr")

        output_dir_lattice = os.path.join(main_output_dir, "figures")
        Path(output_dir_lattice).mkdir(parents=True, exist_ok=True)
        output_file_name = os.path.join(output_dir_lattice, "artificial_teams_figure_1.pdf")

        # We select an itemset showing a corrective behavior
        # We firstly get the itemset showing a corrective behavior
        corrSign = fp_divergence_fnr.getCorrectiveItems()

        id_col = 0

        print(corrSign)


        # We visualize and save the lattice of the selected itemset
        if len(corrSign) > id_col:
            S_i = corrSign[["S+i"]].head(id_col + 1).values[id_col][0]
            itemsetsOfInterest = [
                frozenset({"wins_bin=high", "att_bin=low"}),    # strong players with low attendance
                frozenset({"wins_bin=low", "loss_bin=high"}),   # struggling players
                frozenset({"wins_bin=high", "avg_bin=mid"}),    # high wins but not globally strong
                frozenset({"att_bin=high", "wins_bin=mid"}),    # high attendance, wins avg amount of the time
            ]


            fig1 = fp_divergence_fnr.plotLatticeItemset(
                S_i,
                Th_divergence=0.15,
                sizeDot="small",
                getLower=True,
                round_v=2,
                displayItemsetLabels=True,
                show=show_figures,
                font_size_div=12,
                font_size_ItemsetLabels=13,
                itemsetsOfInterest=itemsetsOfInterest,
                plot_bgcolor="#FFFFFF",
            )

            import plotly.io as pio

            pio.kaleido.scope.default_format = "pdf"
            fig1.write_image(output_file_name, width=600, height=330, engine="kaleido")

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
        default=["artificial_teams_figure_1"],
        help='specify the figures and tables to compute, specify one or more among ["artificial_teams_figure_1"]',
    )

    parser.add_argument(
        "--no_show_figs",
        action="store_false",
        help="specify not_show_figures to vizualize the plots. The results are stored into the specified outpur dir.",
    )

    args = parser.parse_args()

    artificial_teams_experiments(
        name_output_dir=args.name_output_dir,
        compute_results=args.compute_results,
        show_figures=args.no_show_figs,
    )
