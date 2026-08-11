from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EPS = 1e-12


# ============================================================================
# Command-line arguments
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse FednnUNet client-update disagreement and prepare "
            "empirical calibration evidence for EC-DAPS."
        )
    )

    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help=(
            "G0 experiment directory or fold directory. "
            "The script searches for update_stats.jsonl inside RUN_DIR "
            "and RUN_DIR/fold_0."
        ),
    )

    parser.add_argument(
        "--update-stats",
        type=Path,
        default=None,
        help=(
            "Optional explicit path to update_stats.jsonl. "
            "Normally --run-dir is sufficient."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for analysis CSV, JSON and PNG outputs. "
            "Defaults to <run-dir>/ec_daps_analysis."
        ),
    )

    parser.add_argument(
        "--warmup-rounds",
        type=int,
        default=200,
        help=(
            "Maximum federated round used when summarising candidate "
            "EC-DAPS warm-up features. Default: 200."
        ),
    )

    parser.add_argument(
        "--cosine-threshold",
        type=float,
        default=0.0,
        help=(
            "Candidate directional-conflict threshold. "
            "Cosine similarity <= this value is counted as directional "
            "conflict when estimating persistence. "
            "This is NOT the final EC-DAPS threshold. Default: 0.0."
        ),
    )

    parser.add_argument(
        "--rolling-window",
        type=int,
        default=10,
        help=(
            "Number of logged observations used for rolling persistence. "
            "Default: 10."
        ),
    )

    parser.add_argument(
        "--region-map",
        type=Path,
        default=None,
        help=(
            "Optional JSON mapping broader experimental regions "
            "to logged parameter groups."
        ),
    )

    parser.add_argument(
        "--ablation-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV containing G0 and personalisation results. "
            "Required columns: policy, region, client_id, and either "
            "dice or summary_json."
        ),
    )

    parser.add_argument(
        "--baseline-policy",
        type=str,
        default="G0",
        help="Policy used as the Dice baseline. Default: G0.",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Generate tables only and skip PNG plots.",
    )

    return parser.parse_args()


# ============================================================================
# File helpers
# ============================================================================


def resolve_update_stats_path(
    run_dir: Path,
    explicit_path: Path | None,
) -> Path:
    if explicit_path is not None:
        path = explicit_path.expanduser().resolve()

        if not path.is_file():
            raise FileNotFoundError(
                f"update_stats.jsonl not found: {path}"
            )

        return path

    run_dir = run_dir.expanduser().resolve()

    candidates = [
        run_dir / "update_stats.jsonl",
        run_dir / "fold_0" / "update_stats.jsonl",
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    searched = "\n".join(
        f"  - {candidate}"
        for candidate in candidates
    )

    raise FileNotFoundError(
        "Could not find update_stats.jsonl.\n"
        "Searched:\n"
        f"{searched}\n\n"
        "Use --update-stats if you want to provide the exact path."
    )


def load_json(path: Path) -> Any:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def write_json(
    path: Path,
    payload: Any,
) -> None:
    """
    Safely write an analysis JSON file.

    This intentionally mirrors the safe-write behaviour used by
    fednnunet/experiment_utils.py without requiring the analysis
    directory to be part of the fednnunet package.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )

        file.write("\n")
        file.flush()

    temporary_path.replace(path)


# ============================================================================
# update_stats.jsonl loading
# ============================================================================


def require_keys(
    record: dict[str, Any],
    required_keys: set[str],
    line_number: int,
) -> None:
    missing = sorted(
        required_keys.difference(record)
    )

    if missing:
        raise ValueError(
            f"update_stats.jsonl line {line_number} is missing "
            f"required field(s): {missing}\n"
            f"Found fields: {sorted(record)}"
        )


def numeric_dict(
    value: Any,
    field_name: str,
    line_number: int,
) -> dict[str, float]:
    if not isinstance(value, dict):
        raise TypeError(
            f"Line {line_number}: {field_name} must be a JSON object."
        )

    output: dict[str, float] = {}

    for key, item in value.items():
        if item is None:
            output[str(key)] = math.nan
            continue

        try:
            output[str(key)] = float(item)

        except (TypeError, ValueError) as error:
            raise TypeError(
                f"Line {line_number}: "
                f"{field_name}[{key!r}] must be numeric."
            ) from error

    return output


def read_update_stats(
    path: Path,
) -> pd.DataFrame:
    """
    Read the update_stats.jsonl format currently produced by server.py.

    Expected record structure:

    {
        "round": int,
        "group": str,
        "client_labels": [client_a, client_b],
        "parameter_count": int,
        "update_norms": {
            "client_a": float,
            "client_b": float
        },
        "relative_update_norms": {
            "client_a": float,
            "client_b": float
        },
        "pairwise_cosine_similarity": float | null,
        "pairwise_l2_distance": float
    }

    One record is expected for each logged parameter group at each
    selected federated round.
    """

    required_fields = {
        "round",
        "group",
        "client_labels",
        "parameter_count",
        "update_norms",
        "relative_update_norms",
        "pairwise_cosine_similarity",
        "pairwise_l2_distance",
    }

    rows: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, raw_line in enumerate(
            file,
            start=1,
        ):
            raw_line = raw_line.strip()

            if not raw_line:
                continue

            try:
                record = json.loads(raw_line)

            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}"
                ) from error

            if not isinstance(record, dict):
                raise TypeError(
                    f"Line {line_number}: expected a JSON object."
                )

            require_keys(
                record,
                required_fields,
                line_number,
            )

            # ------------------------------------------------------------
            # Clients
            # ------------------------------------------------------------

            labels_raw = record["client_labels"]

            if (
                not isinstance(labels_raw, list)
                or len(labels_raw) != 2
            ):
                raise ValueError(
                    f"Line {line_number}: client_labels must contain "
                    "exactly two clients because the current disagreement "
                    "analysis is defined for the two-client experiment."
                )

            client_labels = [
                str(label)
                for label in labels_raw
            ]

            client_a, client_b = client_labels

            # ------------------------------------------------------------
            # Client update magnitudes
            # ------------------------------------------------------------

            update_norms = numeric_dict(
                record["update_norms"],
                "update_norms",
                line_number,
            )

            relative_update_norms = numeric_dict(
                record["relative_update_norms"],
                "relative_update_norms",
                line_number,
            )

            for client in client_labels:

                if client not in update_norms:
                    raise ValueError(
                        f"Line {line_number}: update_norms has no "
                        f"entry for client {client}."
                    )

                if client not in relative_update_norms:
                    raise ValueError(
                        f"Line {line_number}: relative_update_norms "
                        f"has no entry for client {client}."
                    )

            norm_a = update_norms[client_a]
            norm_b = update_norms[client_b]

            relative_a = relative_update_norms[
                client_a
            ]
            relative_b = relative_update_norms[
                client_b
            ]

            # ------------------------------------------------------------
            # Pairwise disagreement
            # ------------------------------------------------------------

            cosine_raw = record[
                "pairwise_cosine_similarity"
            ]

            cosine_similarity = (
                math.nan
                if cosine_raw is None
                else float(cosine_raw)
            )

            pairwise_l2 = float(
                record["pairwise_l2_distance"]
            )

            # ============================================================
            # EC-DAPS candidate feature C
            #
            # Directional disagreement:
            #
            #     C = 1 - cosine_similarity
            #
            # cosine = +1 -> C = 0
            # cosine =  0 -> C = 1
            # cosine = -1 -> C = 2
            # ============================================================

            directional_disagreement = (
                math.nan
                if math.isnan(cosine_similarity)
                else 1.0 - cosine_similarity
            )

            # ============================================================
            # EC-DAPS candidate feature L
            #
            # Raw L2 distance depends strongly on layer/update scale.
            # Therefore also calculate a symmetric normalised distance:
            #
            #             ||delta_a - delta_b||
            #     L = -----------------------------
            #          ||delta_a|| + ||delta_b||
            #
            # This is retained as a CANDIDATE definition.
            # The empirical analysis may later justify a different form.
            # ============================================================

            norm_sum = norm_a + norm_b

            if norm_sum > EPS:
                update_distance_disagreement = (
                    pairwise_l2
                    / (norm_sum + EPS)
                )
            else:
                update_distance_disagreement = (
                    math.nan
                )

            # ============================================================
            # EC-DAPS candidate feature R
            #
            # Mean relative update magnitude across the two clients.
            # ============================================================

            relative_update_magnitude = float(
                np.nanmean(
                    [
                        relative_a,
                        relative_b,
                    ]
                )
            )

            # ------------------------------------------------------------
            # Store one round/group observation
            # ------------------------------------------------------------

            rows.append(
                {
                    "round": int(
                        record["round"]
                    ),
                    "group": str(
                        record["group"]
                    ),
                    "parameter_count": int(
                        record["parameter_count"]
                    ),

                    "client_a": client_a,
                    "client_b": client_b,

                    "update_norm_a": norm_a,
                    "update_norm_b": norm_b,

                    "mean_update_norm": float(
                        np.nanmean(
                            [
                                norm_a,
                                norm_b,
                            ]
                        )
                    ),

                    "update_norm_imbalance": abs(
                        norm_a - norm_b
                    ),

                    "relative_update_norm_a": (
                        relative_a
                    ),

                    "relative_update_norm_b": (
                        relative_b
                    ),

                    "relative_update_magnitude": (
                        relative_update_magnitude
                    ),

                    "relative_update_imbalance": abs(
                        relative_a - relative_b
                    ),

                    "cosine_similarity": (
                        cosine_similarity
                    ),

                    "directional_disagreement": (
                        directional_disagreement
                    ),

                    "pairwise_l2_distance": (
                        pairwise_l2
                    ),

                    "update_distance_disagreement": (
                        update_distance_disagreement
                    ),
                }
            )

    if not rows:
        raise ValueError(
            f"No update-stat records were found in {path}"
        )

    dataframe = pd.DataFrame(rows)

    dataframe = dataframe.sort_values(
        [
            "group",
            "round",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )

    # Each group should have only one observation per logged round.

    duplicate_mask = dataframe.duplicated(
        subset=[
            "round",
            "group",
        ],
        keep=False,
    )

    if duplicate_mask.any():

        duplicates = dataframe.loc[
            duplicate_mask,
            [
                "round",
                "group",
            ],
        ]

        raise ValueError(
            "Found duplicate round/group records:\n"
            f"{duplicates.to_string(index=False)}"
        )

    return dataframe


# ============================================================================
# Persistence
# ============================================================================


def add_persistence_features(
    dataframe: pd.DataFrame,
    cosine_threshold: float,
    rolling_window: int,
) -> pd.DataFrame:
    """
    Add candidate temporal disagreement features.

    P is NOT being permanently defined here.

    For the initial empirical analysis, directional conflict is defined as:

        cosine_similarity <= cosine_threshold

    We then measure how often that condition persists over time.
    """

    if rolling_window <= 0:
        raise ValueError(
            "--rolling-window must be positive."
        )

    output = dataframe.copy()

    # Keep missing cosine values as missing rather than silently
    # classifying them as agreement.

    output["directional_conflict"] = np.where(
        output["cosine_similarity"].notna(),
        (
            output["cosine_similarity"]
            <= cosine_threshold
        ).astype(float),
        np.nan,
    )

    # ================================================================
    # Candidate P measure 1:
    #
    # Fraction of all observations up to the current round that
    # showed directional conflict.
    # ================================================================

    output[
        "cumulative_conflict_persistence"
    ] = (
        output.groupby(
            "group",
            sort=False,
        )["directional_conflict"]
        .transform(
            lambda series:
            series.expanding().mean()
        )
    )

    # ================================================================
    # Candidate P measure 2:
    #
    # Recent conflict frequency over a rolling number of LOGGED
    # observations.
    #
    # Important:
    # this is observations rather than literal training rounds because
    # later G0 logging may occur every stats_every rounds.
    # ================================================================

    output[
        "rolling_conflict_persistence"
    ] = (
        output.groupby(
            "group",
            sort=False,
        )["directional_conflict"]
        .transform(
            lambda series:
            series.rolling(
                window=rolling_window,
                min_periods=1,
            ).mean()
        )
    )

    return output


def longest_true_streak(
    series: pd.Series,
) -> int:
    """
    Return the longest consecutive sequence of logged conflict observations.
    """

    longest = 0
    current = 0

    for value in series:

        if pd.isna(value):
            current = 0

        elif float(value) >= 0.5:
            current += 1
            longest = max(
                longest,
                current,
            )

        else:
            current = 0

    return longest


# ============================================================================
# Statistical summaries
# ============================================================================


def linear_slope(
    x: pd.Series,
    y: pd.Series,
) -> float:
    valid = pd.DataFrame(
        {
            "x": x,
            "y": y,
        }
    ).dropna()

    if len(valid) < 2:
        return math.nan

    if valid["x"].nunique() < 2:
        return math.nan

    slope = np.polyfit(
        valid["x"].to_numpy(
            dtype=float
        ),
        valid["y"].to_numpy(
            dtype=float
        ),
        deg=1,
    )[0]

    return float(slope)


def summarise_features(
    dataframe: pd.DataFrame,
    warmup_rounds: int,
    label_column: str = "group",
) -> pd.DataFrame:
    """
    Summarise C, L, R and candidate persistence P over the
    proposed EC-DAPS observation period.
    """

    warmup = dataframe.loc[
        dataframe["round"]
        <= warmup_rounds
    ].copy()

    if warmup.empty:
        raise ValueError(
            "No logged observations were found at or before "
            f"round {warmup_rounds}."
        )

    rows: list[dict[str, Any]] = []

    for label, subset in warmup.groupby(
        label_column,
        sort=True,
    ):

        subset = subset.sort_values(
            "round"
        )

        cumulative_valid = subset[
            "cumulative_conflict_persistence"
        ].dropna()

        final_persistence = (
            float(
                cumulative_valid.iloc[-1]
            )
            if not cumulative_valid.empty
            else math.nan
        )

        rows.append(
            {
                label_column: label,

                "observations": int(
                    len(subset)
                ),

                "first_logged_round": int(
                    subset["round"].min()
                ),

                "last_logged_round": int(
                    subset["round"].max()
                ),

                "parameter_count": int(
                    subset[
                        "parameter_count"
                    ].max()
                ),

                # ----------------------------------------------------
                # C: directional disagreement
                # ----------------------------------------------------

                "C_mean": subset[
                    "directional_disagreement"
                ].mean(),

                "C_median": subset[
                    "directional_disagreement"
                ].median(),

                "C_std": subset[
                    "directional_disagreement"
                ].std(),

                "C_max": subset[
                    "directional_disagreement"
                ].max(),

                "C_slope": linear_slope(
                    subset["round"],
                    subset[
                        "directional_disagreement"
                    ],
                ),

                # ----------------------------------------------------
                # L: normalised update distance
                # ----------------------------------------------------

                "L_mean": subset[
                    "update_distance_disagreement"
                ].mean(),

                "L_median": subset[
                    "update_distance_disagreement"
                ].median(),

                "L_std": subset[
                    "update_distance_disagreement"
                ].std(),

                "L_max": subset[
                    "update_distance_disagreement"
                ].max(),

                # ----------------------------------------------------
                # R: relative update magnitude
                # ----------------------------------------------------

                "R_mean": subset[
                    "relative_update_magnitude"
                ].mean(),

                "R_median": subset[
                    "relative_update_magnitude"
                ].median(),

                "R_std": subset[
                    "relative_update_magnitude"
                ].std(),

                "R_max": subset[
                    "relative_update_magnitude"
                ].max(),

                # ----------------------------------------------------
                # P: candidate persistence measurements
                # ----------------------------------------------------

                "P_conflict_fraction": subset[
                    "directional_conflict"
                ].mean(),

                "P_max_conflict_streak": (
                    longest_true_streak(
                        subset[
                            "directional_conflict"
                        ]
                    )
                ),

                "P_final_cumulative": (
                    final_persistence
                ),

                # ----------------------------------------------------
                # Raw reference measurements
                # ----------------------------------------------------

                "cosine_mean": subset[
                    "cosine_similarity"
                ].mean(),

                "pairwise_l2_mean": subset[
                    "pairwise_l2_distance"
                ].mean(),
            }
        )

    return pd.DataFrame(rows)


def feature_correlation_table(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Examine redundancy between candidate disagreement measurements.

    Spearman correlation is used because relationships do not need
    to be linear.
    """

    columns = [
        "directional_disagreement",
        "update_distance_disagreement",
        "relative_update_magnitude",
        "rolling_conflict_persistence",
    ]

    available = [
        column
        for column in columns
        if (
            column in dataframe.columns
            and dataframe[
                column
            ].notna().any()
        )
    ]

    return dataframe[
        available
    ].corr(
        method="spearman"
    )


# ============================================================================
# Optional functional-region mapping
# ============================================================================


def load_region_map(
    path: Path | None,
) -> dict[str, list[str]] | None:
    """
    Example JSON:

    {
        "early_encoder": [
            "encoder_stage_0",
            "encoder_stage_1"
        ],
        "deep_encoder": [
            "encoder_stage_2",
            "bottleneck"
        ],
        "decoder": [
            "decoder_stage_0",
            "decoder_stage_1"
        ]
    }
    """

    if path is None:
        return None

    path = path.expanduser().resolve()

    payload = load_json(path)

    if not isinstance(payload, dict):
        raise TypeError(
            "--region-map must contain a JSON object."
        )

    region_map: dict[str, list[str]] = {}

    for region, groups in payload.items():

        if (
            not isinstance(groups, list)
            or not groups
        ):
            raise ValueError(
                f"Region {region!r} must map to "
                "a non-empty list of groups."
            )

        region_map[
            str(region)
        ] = [
            str(group)
            for group in groups
        ]

    return region_map


def weighted_mean(
    values: pd.Series,
    weights: pd.Series,
) -> float:
    valid = (
        values.notna()
        & weights.notna()
        & (weights > 0)
    )

    if not valid.any():
        return float(
            values.mean()
        )

    return float(
        np.average(
            values.loc[
                valid
            ].astype(float),
            weights=weights.loc[
                valid
            ].astype(float),
        )
    )


def aggregate_to_regions(
    dataframe: pd.DataFrame,
    region_map: dict[str, list[str]],
    cosine_threshold: float,
    rolling_window: int,
) -> pd.DataFrame:
    """
    Aggregate stage-level logging into broader ablation-level regions.

    IMPORTANT:

    These values are parameter-count-weighted summaries of the
    already-computed group metrics.

    They are useful for descriptive calibration, but they are not
    mathematically identical to recomputing cosine/L2 from one
    concatenated parameter vector.
    """

    known_groups = set(
        dataframe["group"].unique()
    )

    rows: list[dict[str, Any]] = []

    for region, groups in region_map.items():

        missing = sorted(
            set(groups).difference(
                known_groups
            )
        )

        if missing:
            raise ValueError(
                f"Region {region!r} references "
                f"group(s) not found in update_stats.jsonl: "
                f"{missing}\n"
                f"Available groups: "
                f"{sorted(known_groups)}"
            )

        region_data = dataframe.loc[
            dataframe[
                "group"
            ].isin(
                groups
            )
        ].copy()

        for (
            round_number,
            subset,
        ) in region_data.groupby(
            "round"
        ):

            weights = subset[
                "parameter_count"
            ].astype(float)

            rows.append(
                {
                    "round": int(
                        round_number
                    ),

                    "region": region,

                    "parameter_count": int(
                        subset[
                            "parameter_count"
                        ].sum()
                    ),

                    "cosine_similarity": (
                        weighted_mean(
                            subset[
                                "cosine_similarity"
                            ],
                            weights,
                        )
                    ),

                    "directional_disagreement": (
                        weighted_mean(
                            subset[
                                "directional_disagreement"
                            ],
                            weights,
                        )
                    ),

                    "update_distance_disagreement": (
                        weighted_mean(
                            subset[
                                "update_distance_disagreement"
                            ],
                            weights,
                        )
                    ),

                    "relative_update_magnitude": (
                        weighted_mean(
                            subset[
                                "relative_update_magnitude"
                            ],
                            weights,
                        )
                    ),

                    "pairwise_l2_distance": (
                        weighted_mean(
                            subset[
                                "pairwise_l2_distance"
                            ],
                            weights,
                        )
                    ),

                    "mean_update_norm": (
                        weighted_mean(
                            subset[
                                "mean_update_norm"
                            ],
                            weights,
                        )
                    ),
                }
            )

    output = pd.DataFrame(rows)

    output = output.sort_values(
        [
            "region",
            "round",
        ]
    ).reset_index(
        drop=True
    )

    output[
        "directional_conflict"
    ] = np.where(
        output[
            "cosine_similarity"
        ].notna(),

        (
            output[
                "cosine_similarity"
            ]
            <= cosine_threshold
        ).astype(float),

        np.nan,
    )

    output[
        "cumulative_conflict_persistence"
    ] = (
        output.groupby(
            "region",
            sort=False,
        )[
            "directional_conflict"
        ]
        .transform(
            lambda series:
            series.expanding().mean()
        )
    )

    output[
        "rolling_conflict_persistence"
    ] = (
        output.groupby(
            "region",
            sort=False,
        )[
            "directional_conflict"
        ]
        .transform(
            lambda series:
            series.rolling(
                window=rolling_window,
                min_periods=1,
            ).mean()
        )
    )

    return output


# ============================================================================
# Optional nnU-Net Dice / ablation loading
# ============================================================================


def extract_foreground_dice(
    summary_path: Path,
) -> float:
    """
    Read the foreground Dice used by nnU-Net validation.

    We intentionally require foreground_mean["Dice"] rather than
    silently guessing a different metric.
    """

    payload = load_json(
        summary_path
    )

    foreground_mean = payload.get(
        "foreground_mean"
    )

    if not isinstance(
        foreground_mean,
        dict,
    ):
        raise ValueError(
            f"{summary_path} has no foreground_mean object."
        )

    for key, value in foreground_mean.items():

        if str(
            key
        ).lower() == "dice":
            return float(value)

    raise ValueError(
        f"{summary_path} has foreground_mean "
        "but no Dice field."
    )


def read_ablation_results(
    path: Path,
) -> pd.DataFrame:
    """
    Expected CSV format:

    policy,region,client_id,summary_json
    G0,global,301,/path/to/G0/client_301/summary.json
    G0,global,302,/path/to/G0/client_302/summary.json
    N,normalisation,301,/path/to/N/client_301/summary.json
    N,normalisation,302,/path/to/N/client_302/summary.json

    Alternatively, provide a numeric `dice` column instead of
    summary_json.
    """

    path = path.expanduser().resolve()

    dataframe = pd.read_csv(
        path
    )

    required = {
        "policy",
        "region",
        "client_id",
    }

    missing = sorted(
        required.difference(
            dataframe.columns
        )
    )

    if missing:
        raise ValueError(
            f"{path} is missing required column(s): "
            f"{missing}"
        )

    if (
        "dice" not in dataframe.columns
        and "summary_json"
        not in dataframe.columns
    ):
        raise ValueError(
            f"{path} must contain either a dice "
            "column or a summary_json column."
        )

    dice_values: list[float] = []

    for _, row in dataframe.iterrows():

        # Direct Dice value takes precedence if supplied.

        if (
            "dice" in dataframe.columns
            and pd.notna(
                row.get(
                    "dice"
                )
            )
        ):
            dice_values.append(
                float(
                    row["dice"]
                )
            )
            continue

        summary_value = row.get(
            "summary_json"
        )

        if pd.isna(
            summary_value
        ):
            raise ValueError(
                "Each ablation row must provide "
                "either dice or summary_json."
            )

        summary_path = Path(
            str(
                summary_value
            )
        ).expanduser()

        if not summary_path.is_absolute():

            summary_path = (
                path.parent
                / summary_path
            ).resolve()

        if not summary_path.is_file():
            raise FileNotFoundError(
                "Ablation summary.json not found: "
                f"{summary_path}"
            )

        dice_values.append(
            extract_foreground_dice(
                summary_path
            )
        )

    output = dataframe[
        [
            "policy",
            "region",
            "client_id",
        ]
    ].copy()

    output[
        "policy"
    ] = output[
        "policy"
    ].astype(str)

    output[
        "region"
    ] = output[
        "region"
    ].astype(str)

    output[
        "client_id"
    ] = output[
        "client_id"
    ].astype(str)

    output[
        "dice"
    ] = dice_values

    return output


def build_ablation_summary(
    results: pd.DataFrame,
    baseline_policy: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Calculate each personalisation strategy's Dice change relative to G0.
    """

    baseline = results.loc[
        results[
            "policy"
        ]
        == baseline_policy
    ].copy()

    if baseline.empty:
        raise ValueError(
            "No rows were found for baseline policy "
            f"{baseline_policy!r}."
        )

    baseline_by_client = (
        baseline.groupby(
            "client_id"
        )[
            "dice"
        ]
        .mean()
        .to_dict()
    )

    missing_clients = sorted(
        set(
            results[
                "client_id"
            ]
        ).difference(
            baseline_by_client
        )
    )

    if missing_clients:
        raise ValueError(
            f"Baseline {baseline_policy!r} has no "
            f"Dice value for client(s): "
            f"{missing_clients}"
        )

    detailed = results.copy()

    detailed[
        "baseline_dice"
    ] = detailed[
        "client_id"
    ].map(
        baseline_by_client
    )

    detailed[
        "delta_dice"
    ] = (
        detailed[
            "dice"
        ]
        - detailed[
            "baseline_dice"
        ]
    )

    non_baseline = detailed.loc[
        detailed[
            "policy"
        ]
        != baseline_policy
    ].copy()

    summary = (
        non_baseline.groupby(
            [
                "policy",
                "region",
            ],
            as_index=False,
        )
        .agg(
            clients=(
                "client_id",
                "nunique",
            ),

            mean_dice=(
                "dice",
                "mean",
            ),

            mean_delta_dice=(
                "delta_dice",
                "mean",
            ),

            min_delta_dice=(
                "delta_dice",
                "min",
            ),

            max_delta_dice=(
                "delta_dice",
                "max",
            ),
        )
    )

    # ------------------------------------------------------------
    # Create useful per-client columns dynamically.
    # ------------------------------------------------------------

    dice_pivot = non_baseline.pivot_table(
        index=[
            "policy",
            "region",
        ],
        columns="client_id",
        values="dice",
        aggfunc="mean",
    )

    dice_pivot.columns = [
        f"dice_client_{client}"
        for client in dice_pivot.columns
    ]

    delta_pivot = non_baseline.pivot_table(
        index=[
            "policy",
            "region",
        ],
        columns="client_id",
        values="delta_dice",
        aggfunc="mean",
    )

    delta_pivot.columns = [
        f"delta_dice_client_{client}"
        for client in delta_pivot.columns
    ]

    summary = (
        summary.set_index(
            [
                "policy",
                "region",
            ]
        )
        .join(
            dice_pivot
        )
        .join(
            delta_pivot
        )
        .reset_index()
    )

    return (
        detailed,
        summary,
    )


# ============================================================================
# EC-DAPS calibration
# ============================================================================


def build_calibration_table(
    feature_summary: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    feature_label: str,
) -> pd.DataFrame:
    """
    Join G0 disagreement characteristics to the measured effect of
    making the corresponding region local.

    This table is intended to become the central empirical calibration
    dataset for EC-DAPS.
    """

    features = feature_summary.rename(
        columns={
            feature_label: "region"
        }
    )

    calibration = ablation_summary.merge(
        features,
        on="region",
        how="left",
        validate="many_to_one",
    )

    unmatched = calibration.loc[
        calibration[
            "C_mean"
        ].isna(),
        "region",
    ].unique()

    if len(
        unmatched
    ) > 0:
        print(
            "WARNING: no disagreement feature summary "
            "matched the following ablation regions:"
        )

        for region in sorted(
            unmatched
        ):
            print(
                f"  - {region}"
            )

    return calibration


def calibration_correlations(
    calibration: pd.DataFrame,
) -> pd.DataFrame:
    """
    Descriptive relationships between candidate EC-DAPS features and
    measured personalisation benefit.

    These correlations must NOT be treated as strong statistical
    evidence when only a small number of functional regions are available.
    """

    features = [
        "C_mean",
        "L_mean",
        "R_mean",
        "P_conflict_fraction",
    ]

    rows: list[
        dict[str, Any]
    ] = []

    for feature in features:

        subset = calibration[
            [
                feature,
                "mean_delta_dice",
            ]
        ].dropna()

        if len(
            subset
        ) < 3:

            rows.append(
                {
                    "feature": feature,
                    "n": int(
                        len(
                            subset
                        )
                    ),
                    "pearson": math.nan,
                    "spearman": math.nan,
                }
            )

            continue

        rows.append(
            {
                "feature": feature,

                "n": int(
                    len(
                        subset
                    )
                ),

                "pearson": subset[
                    feature
                ].corr(
                    subset[
                        "mean_delta_dice"
                    ],
                    method="pearson",
                ),

                "spearman": subset[
                    feature
                ].corr(
                    subset[
                        "mean_delta_dice"
                    ],
                    method="spearman",
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# Plotting
# ============================================================================


def plot_metric(
    dataframe: pd.DataFrame,
    label_column: str,
    metric: str,
    ylabel: str,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(
            10,
            6,
        )
    )

    for (
        label,
        subset,
    ) in dataframe.groupby(
        label_column,
        sort=True,
    ):

        subset = subset.sort_values(
            "round"
        )

        axis.plot(
            subset[
                "round"
            ],
            subset[
                metric
            ],
            marker="o",
            markersize=3,
            linewidth=1.2,
            label=str(
                label
            ),
        )

    axis.set_xlabel(
        "Federated round"
    )

    axis.set_ylabel(
        ylabel
    )

    axis.set_title(
        f"{ylabel} over federated training"
    )

    axis.grid(
        True,
        alpha=0.25,
    )

    axis.legend(
        bbox_to_anchor=(
            1.02,
            1,
        ),
        loc="upper left",
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
    )

    plt.close(
        figure
    )


def plot_calibration_scatter(
    calibration: pd.DataFrame,
    feature: str,
    xlabel: str,
    output_path: Path,
) -> None:
    subset = calibration[
        [
            feature,
            "mean_delta_dice",
            "policy",
            "region",
        ]
    ].dropna()

    if subset.empty:
        return

    figure, axis = plt.subplots(
        figsize=(
            8,
            6,
        )
    )

    axis.scatter(
        subset[
            feature
        ],
        subset[
            "mean_delta_dice"
        ],
        s=60,
    )

    for _, row in subset.iterrows():

        axis.annotate(
            (
                f"{row['policy']} "
                f"({row['region']})"
            ),
            (
                row[
                    feature
                ],
                row[
                    "mean_delta_dice"
                ],
            ),
            xytext=(
                5,
                5,
            ),
            textcoords="offset points",
        )

    axis.axhline(
        0.0,
        linewidth=1,
    )

    axis.set_xlabel(
        xlabel
    )

    axis.set_ylabel(
        "Mean ΔDice relative to G0"
    )

    axis.set_title(
        f"{xlabel} vs personalisation benefit"
    )

    axis.grid(
        True,
        alpha=0.25,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
    )

    plt.close(
        figure
    )


# ============================================================================
# Main analysis pipeline
# ============================================================================


def main() -> None:
    args = parse_args()

    if args.warmup_rounds <= 0:
        raise ValueError(
            "--warmup-rounds must be positive."
        )

    run_dir = args.run_dir.expanduser().resolve()

    update_stats_path = resolve_update_stats_path(
        run_dir,
        args.update_stats,
    )

    if args.output_dir is not None:

        output_dir = (
            args.output_dir
            .expanduser()
            .resolve()
        )

    else:

        output_dir = (
            run_dir
            / "ec_daps_analysis"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ====================================================================
    # 1. Load raw G0 disagreement data
    # ====================================================================

    print()
    print(
        "=== EC-DAPS ANALYSIS ==="
    )
    print()

    print(
        f"Reading update statistics:\n"
        f"  {update_stats_path}"
    )

    round_level = read_update_stats(
        update_stats_path
    )

    # ====================================================================
    # 2. Derive candidate persistence P
    # ====================================================================

    round_level = add_persistence_features(
        round_level,
        cosine_threshold=(
            args.cosine_threshold
        ),
        rolling_window=(
            args.rolling_window
        ),
    )

    round_level.to_csv(
        output_dir
        / "g0_round_level_features.csv",
        index=False,
    )

    # ====================================================================
    # 3. Summarise candidate C, L, R, P features
    # ====================================================================

    group_summary = summarise_features(
        round_level,
        warmup_rounds=(
            args.warmup_rounds
        ),
        label_column="group",
    )

    group_summary.to_csv(
        output_dir
        / "g0_group_feature_summary.csv",
        index=False,
    )

    # ====================================================================
    # 4. Check possible redundancy between features
    # ====================================================================

    correlation = feature_correlation_table(
        round_level
    )

    correlation.to_csv(
        output_dir
        / "g0_round_level_feature_spearman.csv"
    )

    print()
    print(
        f"Loaded {len(round_level)} "
        "logged group observations."
    )

    print(
        f"Logged rounds: "
        f"{round_level['round'].nunique()}"
    )

    print(
        f"Logged groups: "
        f"{round_level['group'].nunique()}"
    )

    print()

    for group in sorted(
        round_level[
            "group"
        ].unique()
    ):
        print(
            f"  - {group}"
        )

    # ====================================================================
    # 5. Optional broader functional-region analysis
    # ====================================================================

    region_map = load_region_map(
        args.region_map
    )

    feature_summary_for_calibration = (
        group_summary
    )

    feature_label_for_calibration = (
        "group"
    )

    if region_map is not None:

        region_round_level = (
            aggregate_to_regions(
                round_level,
                region_map=region_map,
                cosine_threshold=(
                    args.cosine_threshold
                ),
                rolling_window=(
                    args.rolling_window
                ),
            )
        )

        region_round_level.to_csv(
            output_dir
            / "g0_region_round_level_features.csv",
            index=False,
        )

        region_summary = summarise_features(
            region_round_level,
            warmup_rounds=(
                args.warmup_rounds
            ),
            label_column="region",
        )

        region_summary.to_csv(
            output_dir
            / "g0_region_feature_summary.csv",
            index=False,
        )

        feature_summary_for_calibration = (
            region_summary
        )

        feature_label_for_calibration = (
            "region"
        )

    # ====================================================================
    # 6. G0 plots
    # ====================================================================

    if not args.no_plots:

        plot_metric(
            round_level,
            label_column="group",
            metric=(
                "directional_disagreement"
            ),
            ylabel=(
                "Directional disagreement "
                "C = 1 - cosine"
            ),
            output_path=(
                output_dir
                / "directional_disagreement_vs_round.png"
            ),
        )

        plot_metric(
            round_level,
            label_column="group",
            metric=(
                "update_distance_disagreement"
            ),
            ylabel=(
                "Normalised update-distance "
                "disagreement L"
            ),
            output_path=(
                output_dir
                / "update_distance_disagreement_vs_round.png"
            ),
        )

        plot_metric(
            round_level,
            label_column="group",
            metric=(
                "relative_update_magnitude"
            ),
            ylabel=(
                "Relative update magnitude R"
            ),
            output_path=(
                output_dir
                / "relative_update_magnitude_vs_round.png"
            ),
        )

        plot_metric(
            round_level,
            label_column="group",
            metric=(
                "cumulative_conflict_persistence"
            ),
            ylabel=(
                "Cumulative directional-conflict "
                "persistence P"
            ),
            output_path=(
                output_dir
                / "cumulative_persistence_vs_round.png"
            ),
        )

        plot_metric(
            round_level,
            label_column="group",
            metric=(
                "rolling_conflict_persistence"
            ),
            ylabel=(
                "Rolling conflict persistence "
                f"({args.rolling_window} "
                "logged observations)"
            ),
            output_path=(
                output_dir
                / "rolling_persistence_vs_round.png"
            ),
        )

    # ====================================================================
    # 7. Optional ablation analysis
    # ====================================================================

    if args.ablation_csv is not None:

        print()
        print(
            "Reading personalisation "
            f"results:\n  {args.ablation_csv}"
        )

        ablation_results = (
            read_ablation_results(
                args.ablation_csv
            )
        )

        (
            detailed_ablation,
            ablation_summary,
        ) = build_ablation_summary(
            ablation_results,
            baseline_policy=(
                args.baseline_policy
            ),
        )

        detailed_ablation.to_csv(
            output_dir
            / "ablation_client_results.csv",
            index=False,
        )

        ablation_summary.to_csv(
            output_dir
            / "ablation_policy_summary.csv",
            index=False,
        )

        # ================================================================
        # Join:
        #
        # G0 disagreement characteristics
        #               +
        # Dice effect of localisation
        #
        # This becomes the empirical calibration table for EC-DAPS.
        # ================================================================

        calibration = build_calibration_table(
            feature_summary_for_calibration,
            ablation_summary,
            feature_label=(
                feature_label_for_calibration
            ),
        )

        calibration.to_csv(
            output_dir
            / "ec_daps_calibration_table.csv",
            index=False,
        )

        calibration_corr = (
            calibration_correlations(
                calibration
            )
        )

        calibration_corr.to_csv(
            output_dir
            / (
                "ec_daps_feature_delta_dice_"
                "correlations.csv"
            ),
            index=False,
        )

        if not args.no_plots:

            calibration_plots = [
                (
                    "C_mean",
                    "Mean directional disagreement C",
                    "calibration_C_vs_delta_dice.png",
                ),
                (
                    "L_mean",
                    "Mean update-distance disagreement L",
                    "calibration_L_vs_delta_dice.png",
                ),
                (
                    "R_mean",
                    "Mean relative update magnitude R",
                    "calibration_R_vs_delta_dice.png",
                ),
                (
                    "P_conflict_fraction",
                    "Conflict persistence P",
                    "calibration_P_vs_delta_dice.png",
                ),
            ]

            for (
                feature,
                xlabel,
                filename,
            ) in calibration_plots:

                plot_calibration_scatter(
                    calibration,
                    feature=feature,
                    xlabel=xlabel,
                    output_path=(
                        output_dir
                        / filename
                    ),
                )

    # ====================================================================
    # 8. Analysis manifest
    # ====================================================================

    manifest = {
        "analysis": (
            "EC-DAPS empirical calibration analysis"
        ),

        "update_stats": str(
            update_stats_path
        ),

        "run_dir": str(
            run_dir
        ),

        "output_dir": str(
            output_dir
        ),

        "warmup_rounds": int(
            args.warmup_rounds
        ),

        "cosine_conflict_threshold": float(
            args.cosine_threshold
        ),

        "rolling_window_logged_observations": int(
            args.rolling_window
        ),

        "region_map": (
            str(
                args.region_map
                .expanduser()
                .resolve()
            )
            if args.region_map is not None
            else None
        ),

        "ablation_csv": (
            str(
                args.ablation_csv
                .expanduser()
                .resolve()
            )
            if args.ablation_csv is not None
            else None
        ),

        "baseline_policy": str(
            args.baseline_policy
        ),

        "notes": [
            (
                "C, L, R and P are candidate empirical "
                "features. This analysis does not define "
                "the final EC-DAPS sharing score."
            ),
            (
                "C is currently defined as 1 minus "
                "pairwise cosine similarity."
            ),
            (
                "L is currently represented by pairwise "
                "update distance normalised by the sum of "
                "the two client update norms."
            ),
            (
                "R is the mean relative update magnitude "
                "of the two clients."
            ),
            (
                "P currently describes persistence of "
                "directional conflict under the configured "
                "cosine threshold."
            ),
            (
                "The final EC-DAPS score and threshold "
                "must be chosen only after empirical G0 "
                "and personalisation results are available."
            ),
            (
                "Region-level values, when requested, "
                "are parameter-count-weighted descriptive "
                "summaries of logged group statistics."
            ),
            (
                "Feature-versus-DeltaDice correlations are "
                "descriptive and should not be interpreted "
                "as strong statistical evidence when the "
                "number of functional regions is small."
            ),
        ],
    }

    write_json(
        output_dir
        / "analysis_manifest.json",
        manifest,
    )

    # ====================================================================
    # Finished
    # ====================================================================

    print()
    print(
        "=== ANALYSIS COMPLETE ==="
    )

    print(
        f"Outputs written to:\n"
        f"  {output_dir}"
    )

    print()
    print(
        "No final EC-DAPS sharing score was fitted "
        "or hard-coded."
    )

    print(
        "The generated results are intended to provide "
        "the empirical evidence needed to construct "
        "that rule after G0 and the controlled "
        "personalisation ablations."
    )


if __name__ == "__main__":
    main()