import json
import os
import re
from io import BytesIO
from logging import INFO, WARNING
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import flwr as fl
import torch
from flwr.common import (
    Code,
    EvaluateIns,
    FitIns,
    GetParametersIns,
    GetPropertiesIns,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from fednnunet.experiment_utils import (
    seed_everything,
    write_json,
)


def state_dict_to_bytes(state_dict) -> bytes:
    bytes_io = BytesIO()
    torch.save(state_dict, bytes_io)
    return bytes_io.getvalue()


def state_dict_to_parameters(state_dict) -> Parameters:
    tensors = state_dict_to_bytes(state_dict)
    log(INFO, f"State dict to parameters: {len(tensors)} bytes")
    return Parameters(tensors=[tensors], tensor_type="whatever")


def bytes_to_state_dict(bytes_data: bytes) -> dict:
    """Converts bytes back to a PyTorch state_dict on CPU."""
    bytes_io = BytesIO(bytes_data)
    return torch.load(bytes_io, map_location=torch.device("cpu"))


def parameters_to_state_dict(parameters: Parameters) -> dict:
    """Converts Flower Parameters back to a PyTorch state_dict."""
    bytes_data = parameters.tensors[0]
    return bytes_to_state_dict(bytes_data)


def weighted_average_metrics(results):
    """Aggregate numeric client metrics using num_examples as weights.

    This avoids giving a small client and a large client the same influence
    in the reported server-side metrics.
    """
    if not results:
        return {}

    totals = {}
    total_weights = {}

    for _, fit_res in results:
        weight = fit_res.num_examples if fit_res.num_examples is not None else 1

        for key, value in fit_res.metrics.items():
            # Only aggregate numeric metrics. Skip strings/lists/etc.
            if isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0.0) + float(value) * weight
                total_weights[key] = total_weights.get(key, 0) + weight

    return {
        key: totals[key] / total_weights[key]
        for key in totals
        if total_weights[key] > 0
    }


def weighted_mean(values, weights):
    return sum(value * weight for value, weight in zip(values, weights)) / sum(weights)


def max_value(values, weights):
    values = torch.tensor(values)
    return torch.max(values).item()


def min_value(values, weights):
    values = torch.tensor(values)
    return torch.min(values).item()


def concatenate(values, weights):
    # Concatenate the nested lists maintaining the structure
    return [item for sublist in values for item in sublist]


def aggregate_fingerprints(parameters: List[Parameters]) -> Parameters:
    # Extract the state_dicts from the parameters
    state_dicts = [parameters_to_state_dict(p) for p in parameters]

    # Define aggregation functions for each key in the fingerprint
    aggregation_dict = {
        "max": max_value,
        "min": min_value,
        "mean": weighted_mean,
        "median": weighted_mean,
        "std": weighted_mean,
        "percentile_00_5": weighted_mean,
        "percentile_99_5": weighted_mean,
        "median_relative_size_after_cropping": weighted_mean,
        "shapes_after_crop": concatenate,
        "spacings": concatenate,
    }

    # Infer number of samples on each client by dict 'shapes_after_crop' length
    num_samples = [len(sd["shapes_after_crop"]) for sd in state_dicts]
    print(f"Num samples per client: {num_samples}")

    new_state_dict = {}
    for key in state_dicts[0].keys():
        if isinstance(state_dicts[0][key], dict):
            new_state_dict[key] = {}
            for subkey in state_dicts[0][key].keys():
                if isinstance(state_dicts[0][key][subkey], dict):
                    new_state_dict[key][subkey] = {}
                    for subsubkey in state_dicts[0][key][subkey].keys():
                        new_state_dict[key][subkey][subsubkey] = aggregation_dict[
                            subsubkey
                        ](
                            [sd[key][subkey][subsubkey] for sd in state_dicts],
                            num_samples,
                        )
                else:
                    new_state_dict[key][subkey] = aggregation_dict[subkey](
                        [sd[key][subkey] for sd in state_dicts], num_samples
                    )
        # elif isinstance(state_dicts[0][key], list):
        #     pass
        else:
            new_state_dict[key] = aggregation_dict[key](
                [sd[key] for sd in state_dicts], num_samples
            )

    # Convert the new state_dict back to parameters
    return state_dict_to_parameters(new_state_dict)


class MyStrategy(fl.server.strategy.FedAvg):

    def __init__(
        self,
        task: str,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        inplace: bool = True,
        aggregation_mode: str = "weighted",
        num_rounds: int = 1,
        run_dir: str = ".",
        stats_every: int = 10,
        initial_dataset_id: int = 301,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            inplace=inplace,
        )

        self.task = task
        self.aggregation_mode = aggregation_mode
        self.num_rounds = int(num_rounds)
        self.stats_every = max(1, int(stats_every))
        self.initial_dataset_id = int(initial_dataset_id)

        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Stores the shared state sent to clients at the beginning
        # of the current communication round.
        self.round_start_state = None

        # The compatibility manifest only needs to be written once.
        self.compatibility_manifest_written = False

        log(
            INFO,
            (
                f"Using aggregation_mode={self.aggregation_mode} | "
                f"num_rounds={self.num_rounds} | "
                f"stats_every={self.stats_every} | "
                f"run_dir={self.run_dir}"
            ),
        )

    def initialize_parameters(
        self,
        client_manager: ClientManager,
    ) -> Optional[Parameters]:
        """Request the initial global state from a named dataset client."""
        if self.task != "train":
            return super().initialize_parameters(
                client_manager
            )

        clients_available = client_manager.wait_for(
            num_clients=self.min_available_clients,
            timeout=600,
        )

        if not clients_available:
            raise RuntimeError(
                "Timed out waiting for all clients before "
                "initialising global parameters."
            )

        clients_by_dataset = {}

        for client_id, client_proxy in (
            client_manager.all().items()
        ):
            properties_result = (
                client_proxy.get_properties(
                    ins=GetPropertiesIns(config={}),
                    timeout=60,
                    group_id=0,
                )
            )

            if properties_result.status.code != Code.OK:
                raise RuntimeError(
                    "Could not retrieve properties from "
                    f"Flower client {client_id}: "
                    f"{properties_result.status.message}"
                )

            if "dataset_id" not in properties_result.properties:
                raise RuntimeError(
                    f"Flower client {client_id} did not provide "
                    "a dataset_id property."
                )

            dataset_id = int(
                properties_result.properties["dataset_id"]
            )

            if dataset_id in clients_by_dataset:
                raise RuntimeError(
                    "Multiple connected clients reported the same "
                    f"dataset_id={dataset_id}."
                )

            clients_by_dataset[dataset_id] = (
                client_id,
                client_proxy,
            )

        if self.initial_dataset_id not in clients_by_dataset:
            raise RuntimeError(
                "Requested initial dataset was not connected: "
                f"initial_dataset_id={self.initial_dataset_id}, "
                f"available={sorted(clients_by_dataset)}"
            )

        selected_client_id, selected_client = (
            clients_by_dataset[self.initial_dataset_id]
        )

        parameters_result = selected_client.get_parameters(
            ins=GetParametersIns(config={}),
            timeout=600,
            group_id=0,
        )

        if parameters_result.status.code != Code.OK:
            raise RuntimeError(
                "Could not retrieve initial parameters from "
                f"Dataset {self.initial_dataset_id}: "
                f"{parameters_result.status.message}"
            )

        write_json(
            self.run_dir / "initialization_manifest.json",
            {
                "initial_dataset_id": int(
                    self.initial_dataset_id
                ),
                "flower_client_id": str(
                    selected_client_id
                ),
                "available_dataset_ids": sorted(
                    int(dataset_id)
                    for dataset_id in clients_by_dataset
                ),
                "selection_policy": (
                    "explicit_dataset_id"
                ),
            },
        )

        log(
            INFO,
            (
                "Using deterministic initial parameters from "
                f"Dataset {self.initial_dataset_id}, "
                f"Flower client {selected_client_id}"
            ),
        )

        return parameters_result.parameters

    def find_common_layers(self, state_dicts):
        # Find the common keys in all state_dicts
        if not state_dicts:
            log(WARNING, "find_common_layers received no state_dicts")
            return []

        common_keys = set(state_dicts[0].keys())

        for sd in state_dicts[1:]:
            common_keys.intersection_update(sd.keys())
            log(INFO, f"Number of keys in this dictionary: {len(sd.keys())}")

        log(INFO, f"Number of common keys: {len(common_keys)}")

        # Verify dimensions
        compatible_keys = []
        for key in common_keys:
            dimensions = [sd[key].shape for sd in state_dicts]
            if all(dim == dimensions[0] for dim in dimensions):
                compatible_keys.append(key)

        compatible_keys = sorted(compatible_keys)

        log(INFO, f"Number of compatible keys: {len(compatible_keys)}")
        return compatible_keys

    def is_norm_key(self, key: str) -> bool:
        """Return True if a state_dict key belongs to a normalisation layer.

        The current nnU-Net state_dict uses keys such as:
        encoder.stages.0.0.convs.0.norm.weight
        encoder.stages.0.0.convs.0.norm.bias

        In weighted_no_norm mode, these keys are excluded from server aggregation
        so each client keeps its local normalisation parameters.
        """
        key_lower = key.lower()
        return (
            ".norm." in key_lower
            or key_lower.endswith(".norm.weight")
            or key_lower.endswith(".norm.bias")
            or "batchnorm" in key_lower
            or "instancenorm" in key_lower
        )

    def should_record_update_stats(self, server_round: int) -> bool:
        """Return True when scalar update statistics should be saved."""
        return (
            server_round <= 10
            or server_round % self.stats_every == 0
            or server_round == self.num_rounds
        )

    @staticmethod
    def extract_stage_index(key: str, component: str):
        """Extract an encoder or decoder stage index from a state-dict key."""
        match = re.search(
            rf"{component}\.stages\.(\d+)",
            key.lower(),
        )

        if match is None:
            return None

        return int(match.group(1))

    def get_parameter_group(
        self,
        key: str,
        deepest_compatible_encoder_stage,
    ) -> Optional[str]:
        """Map a model parameter to an architectural analysis group."""
        key_lower = key.lower()

        # dynamic-network-architectures can expose aliases containing
        # all_modules. Excluding these avoids counting a module twice.
        if ".all_modules." in key_lower:
            return None

        if self.is_norm_key(key):
            return "normalisation"

        if (
            "seg_layers." in key_lower
            or "segmentation_head" in key_lower
            or "segmentation_heads" in key_lower
        ):
            match = re.search(
                r"seg_layers\.(\d+)",
                key_lower,
            )

            if match is not None:
                return f"segmentation_head_{match.group(1)}"

            return "segmentation_head"

        if (
            "decoder.transpconvs." in key_lower
            or "decoder.upsampling." in key_lower
            or "decoder.upsample." in key_lower
        ):
            match = re.search(
                r"decoder\.(?:transpconvs|upsampling|upsample)\.(\d+)",
                key_lower,
            )

            if match is not None:
                return f"decoder_upsample_{match.group(1)}"

            return "decoder_upsample"

        encoder_stage = self.extract_stage_index(
            key,
            "encoder",
        )

        if encoder_stage is not None:
            if (
                deepest_compatible_encoder_stage is not None
                and encoder_stage == deepest_compatible_encoder_stage
            ):
                return "bottleneck"

            return f"encoder_stage_{encoder_stage}"

        decoder_stage = self.extract_stage_index(
            key,
            "decoder",
        )

        if decoder_stage is not None:
            return f"decoder_stage_{decoder_stage}"

        return "other"

    def record_update_statistics(
        self,
        server_round: int,
        client_labels,
        client_state_dicts,
        compatible_keys,
    ) -> None:
        """Record scalar disagreement statistics for two client updates."""
        if not self.should_record_update_stats(server_round):
            return

        if self.round_start_state is None:
            log(
                WARNING,
                (
                    f"Round {server_round}: no round-start state "
                    "was available for update analysis."
                ),
            )
            return

        if len(client_state_dicts) != 2:
            log(
                WARNING,
                (
                    "Update-disagreement logging currently expects "
                    "exactly two clients; received "
                    f"{len(client_state_dicts)}."
                ),
            )
            return

        encoder_stages = []

        for key in compatible_keys:
            stage_index = self.extract_stage_index(
                key,
                "encoder",
            )

            if stage_index is not None:
                encoder_stages.append(stage_index)

        deepest_compatible_encoder_stage = (
            max(encoder_stages)
            if encoder_stages
            else None
        )

        grouped_statistics = {}

        for key in compatible_keys:
            group_name = self.get_parameter_group(
                key,
                deepest_compatible_encoder_stage,
            )

            if group_name is None:
                continue

            if key not in self.round_start_state:
                continue

            if (
                key not in client_state_dicts[0]
                or key not in client_state_dicts[1]
            ):
                continue

            round_start_tensor = (
                self.round_start_state[key]
                .detach()
                .cpu()
            )

            client_0_tensor = (
                client_state_dicts[0][key]
                .detach()
                .cpu()
            )

            client_1_tensor = (
                client_state_dicts[1][key]
                .detach()
                .cpu()
            )

            if (
                round_start_tensor.shape != client_0_tensor.shape
                or round_start_tensor.shape != client_1_tensor.shape
            ):
                continue

            if not torch.is_floating_point(round_start_tensor):
                continue

            round_start_float = round_start_tensor.float()

            update_0 = (
                client_0_tensor.float()
                - round_start_float
            )

            update_1 = (
                client_1_tensor.float()
                - round_start_float
            )

            if group_name not in grouped_statistics:
                grouped_statistics[group_name] = {
                    "parameter_count": 0,
                    "base_squared_norm": 0.0,
                    "update_0_squared_norm": 0.0,
                    "update_1_squared_norm": 0.0,
                    "update_dot_product": 0.0,
                    "update_difference_squared_norm": 0.0,
                }

            group_stats = grouped_statistics[group_name]

            group_stats["parameter_count"] += (
                round_start_float.numel()
            )

            group_stats["base_squared_norm"] += float(
                torch.sum(
                    round_start_float * round_start_float
                ).item()
            )

            group_stats["update_0_squared_norm"] += float(
                torch.sum(update_0 * update_0).item()
            )

            group_stats["update_1_squared_norm"] += float(
                torch.sum(update_1 * update_1).item()
            )

            group_stats["update_dot_product"] += float(
                torch.sum(update_0 * update_1).item()
            )

            update_difference = update_0 - update_1

            group_stats[
                "update_difference_squared_norm"
            ] += float(
                torch.sum(
                    update_difference * update_difference
                ).item()
            )

        stats_path = (
            self.run_dir / "update_stats.jsonl"
        )

        with stats_path.open(
            "a",
            encoding="utf-8",
        ) as stats_file:
            for group_name in sorted(grouped_statistics):
                group_stats = grouped_statistics[group_name]

                base_norm = (
                    group_stats["base_squared_norm"] ** 0.5
                )

                update_0_norm = (
                    group_stats["update_0_squared_norm"] ** 0.5
                )

                update_1_norm = (
                    group_stats["update_1_squared_norm"] ** 0.5
                )

                cosine_denominator = (
                    update_0_norm * update_1_norm
                )

                if cosine_denominator > 0:
                    cosine_similarity = (
                        group_stats["update_dot_product"]
                        / cosine_denominator
                    )
                else:
                    cosine_similarity = None

                record = {
                    "round": int(server_round),
                    "group": group_name,
                    "client_labels": [
                        int(label)
                        for label in client_labels
                    ],
                    "parameter_count": int(
                        group_stats["parameter_count"]
                    ),
                    "update_norms": {
                        str(client_labels[0]): float(
                            update_0_norm
                        ),
                        str(client_labels[1]): float(
                            update_1_norm
                        ),
                    },
                    "relative_update_norms": {
                        str(client_labels[0]): float(
                            update_0_norm
                            / max(base_norm, 1e-12)
                        ),
                        str(client_labels[1]): float(
                            update_1_norm
                            / max(base_norm, 1e-12)
                        ),
                    },
                    "pairwise_cosine_similarity": (
                        None
                        if cosine_similarity is None
                        else float(cosine_similarity)
                    ),
                    "pairwise_l2_distance": float(
                        group_stats[
                            "update_difference_squared_norm"
                        ] ** 0.5
                    ),
                }

                stats_file.write(
                    json.dumps(record)
                    + "\n"
                )

        log(
            INFO,
            (
                f"Round {server_round}: wrote "
                f"{len(grouped_statistics)} update-statistic "
                f"group(s) to {stats_path}"
            ),
        )

    def create_compatible_state_dict(self, state_dicts, compatible_keys, weights=None):
        """Create an aggregated state_dict from compatible model parameters.

        aggregation_mode="weighted":
            Sample-weighted FedAvg over all compatible floating-point parameters.

        aggregation_mode="weighted_no_norm":
            FedBN-inspired variant. Normalisation-related parameters are not
            aggregated by the server, allowing each client to keep local
            normalisation behaviour.
        """
        new_state_dict = {}

        if not state_dicts:
            return new_state_dict

        if weights is None:
            weights = [1 for _ in state_dicts]

        total_weight = sum(weights)
        if total_weight <= 0:
            log(WARNING, "Total aggregation weight is zero; falling back to equal weights")
            weights = [1 for _ in state_dicts]
            total_weight = sum(weights)

        normalized_weights = [w / total_weight for w in weights]

        skipped_norm_keys = []

        for key in compatible_keys:
            if self.aggregation_mode == "weighted_no_norm" and self.is_norm_key(key):
                skipped_norm_keys.append(key)
                continue

            tensors = [s[key].detach().cpu() for s in state_dicts]

            # Some state_dict entries can be integer buffers.
            # These should not be averaged as floating-point parameters.
            if not torch.is_floating_point(tensors[0]):
                new_state_dict[key] = tensors[0].clone()
                continue

            stacked = torch.stack(tensors, dim=0)

            weight_tensor = torch.tensor(
                normalized_weights,
                dtype=stacked.dtype,
                device=stacked.device,
            )

            # Reshape weights so they broadcast across parameter dimensions.
            view_shape = [len(normalized_weights)] + [1] * (stacked.dim() - 1)
            weight_tensor = weight_tensor.view(*view_shape)

            new_state_dict[key] = torch.sum(stacked * weight_tensor, dim=0)

        if skipped_norm_keys:
            log(
                INFO,
                (
                    f"FedBN-inspired aggregation skipped "
                    f"{len(skipped_norm_keys)} normalisation parameter(s). "
                    f"Example skipped keys: {skipped_norm_keys[:5]}"
                ),
            )

        return new_state_dict

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure one federated local-training round."""
        config = {}

        if self.on_fit_config_fn is not None:
            config.update(
                self.on_fit_config_fn(server_round)
            )

        config.update(
            {
                "server_round": int(server_round),
                "total_rounds": int(self.num_rounds),
            }
        )

        # Save the exact shared state sent to clients so their
        # returned updates can be measured relative to it.
        if self.task == "train":
            current_state = parameters_to_state_dict(
                parameters
            )

            self.round_start_state = {
                key: value.detach().cpu().clone()
                for key, value in current_state.items()
                if torch.is_tensor(value)
            }

        fit_ins = FitIns(
            parameters,
            config,
        )

        sample_size, min_num_clients = (
            self.num_fit_clients(
                client_manager.num_available()
            )
        )

        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
        )

        return [
            (client, fit_ins)
            for client in clients
        ]

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ):
        """Send evaluation or final synchronisation instructions."""
        if self.task != "train":
            return super().configure_evaluate(
                server_round,
                parameters,
                client_manager,
            )

        if server_round != self.num_rounds:
            return []

        clients = client_manager.sample(
            num_clients=self.min_evaluate_clients,
            min_num_clients=self.min_available_clients,
        )

        evaluate_ins = EvaluateIns(
            parameters,
            {
                "final_sync": True,
                "server_round": int(server_round),
            },
        )

        log(
            INFO,
            (
                f"Round {server_round}: sending final "
                "aggregated parameters to "
                f"{len(clients)} client(s)."
            ),
        )

        return [
            (client, evaluate_ins)
            for client in clients
        ]

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ):
        """Aggregate client training results after each federated round."""

        if failures:
            raise RuntimeError(
                f"Round {rnd} had "
                f"{len(failures)} client failure(s); "
                "formal G0 runs do not accept "
                "partial aggregation."
            )

        successful_results = [
            result
            for result in results
            if result is not None
        ]

        if not successful_results:
            raise RuntimeError(
                f"Round {rnd}: no successful client "
                "results were received."
            )

        if (
            self.task == "train"
            and len(successful_results)
            != self.min_fit_clients
        ):
            raise RuntimeError(
                f"Round {rnd}: expected "
                f"{self.min_fit_clients} client result(s), "
                f"but received {len(successful_results)}."
            )

        # Log per-client metrics for heterogeneity analysis.
        for client, fit_res in successful_results:
            log(
                INFO,
                (
                    f"Round {rnd} | client={client.cid} | "
                    f"num_examples={fit_res.num_examples} | "
                    f"metrics={fit_res.metrics}"
                ),
            )

        if self.task == "extract_fingerprint" or self.task == "plan_and_preprocess":
            return (
                aggregate_fingerprints(
                    [res[1].parameters for res in successful_results]
                ),
                {},
            )

        aggregated_weights = self.aggregate_weights(
            successful_results,
            server_round=rnd,
        )

        if aggregated_weights is None:
            log(WARNING, f"Round {rnd}: aggregation returned None.")
            return None, {}

        # Prefer Flower's metric aggregation function if one was provided.
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [
                (fit_res.num_examples, fit_res.metrics)
                for _, fit_res in successful_results
            ]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        else:
            if rnd == 1:
                log(WARNING, "No fit_metrics_aggregation_fn provided; using weighted_average_metrics().")
            metrics_aggregated = weighted_average_metrics(successful_results)

        log(INFO, f"Round {rnd} aggregated metrics: {metrics_aggregated}")

        return aggregated_weights, metrics_aggregated

    def aggregate_weights(
        self,
        results,
        server_round: int,
    ):
        """Aggregate model parameters using strict sample weighting."""
        client_records = []

        for client_proxy, fit_res in results:
            if (
                fit_res.num_examples is None
                or int(fit_res.num_examples) <= 0
            ):
                raise RuntimeError(
                    f"Invalid num_examples="
                    f"{fit_res.num_examples} "
                    f"from client {client_proxy.cid}"
                )

            dataset_id = int(
                fit_res.metrics.get(
                    "dataset_id",
                    -1,
                )
            )

            if dataset_id < 0:
                raise RuntimeError(
                    "Client result did not include a valid "
                    f"dataset_id: client={client_proxy.cid}, "
                    f"metrics={fit_res.metrics}"
                )

            client_state_dict = parameters_to_state_dict(
                fit_res.parameters
            )

            client_records.append(
                {
                    "dataset_id": dataset_id,
                    "num_examples": int(
                        fit_res.num_examples
                    ),
                    "state_dict": client_state_dict,
                }
            )

        if not client_records:
            raise RuntimeError(
                "aggregate_weights received no client results"
            )

        # Keep client ordering deterministic regardless of the order
        # in which Flower returned results.
        client_records.sort(
            key=lambda record: record["dataset_id"]
        )

        client_labels = [
            record["dataset_id"]
            for record in client_records
        ]

        if len(set(client_labels)) != len(client_labels):
            raise RuntimeError(
                "Duplicate dataset identifiers were received "
                f"during aggregation: {client_labels}"
            )

        weights = [
            record["num_examples"]
            for record in client_records
        ]

        state_dicts = [
            record["state_dict"]
            for record in client_records
        ]

        total_weight = sum(weights)

        if total_weight <= 0:
            raise RuntimeError(
                "The total aggregation weight must be positive."
            )

        normalized_weights = [
            weight / total_weight
            for weight in weights
        ]

        compatible_keys = self.find_common_layers(
            state_dicts
        )

        if not compatible_keys:
            raise RuntimeError(
                "No mutually compatible model parameters "
                "were found for aggregation."
            )

        log(
            INFO,
            (
                f"Round {server_round}: aggregating "
                f"clients={client_labels} | "
                f"counts={weights} | "
                f"normalised_weights="
                f"{normalized_weights}"
            ),
        )

        # Observe the updates before averaging them.
        self.record_update_statistics(
            server_round=server_round,
            client_labels=client_labels,
            client_state_dicts=state_dicts,
            compatible_keys=compatible_keys,
        )

        new_state_dict = self.create_compatible_state_dict(
            state_dicts,
            compatible_keys,
            weights=weights,
        )

        aggregated_keys = sorted(
            new_state_dict.keys()
        )

        if not self.compatibility_manifest_written:
            compatible_key_set = set(
                compatible_keys
            )

            client_non_compatible_keys = {}

            for label, state_dict in zip(
                client_labels,
                state_dicts,
            ):
                client_non_compatible_keys[
                    str(label)
                ] = sorted(
                    set(state_dict.keys())
                    - compatible_key_set
                )

            policy_excluded_compatible_keys = sorted(
                compatible_key_set
                - set(aggregated_keys)
            )

            write_json(
                self.run_dir
                / "compatibility_manifest.json",
                {
                    "round_created": int(server_round),
                    "aggregation_mode": self.aggregation_mode,
                    "client_labels": [
                        int(label)
                        for label in client_labels
                    ],
                    "aggregation_counts": [
                        int(weight)
                        for weight in weights
                    ],
                    "normalized_weights": [
                        float(weight)
                        for weight in normalized_weights
                    ],
                    "client_state_dict_key_counts": {
                        str(label): len(state_dict)
                        for label, state_dict in zip(
                            client_labels,
                            state_dicts,
                        )
                    },
                    "compatible_key_count": len(
                        compatible_keys
                    ),
                    "compatible_keys": compatible_keys,
                    "aggregated_key_count": len(
                        aggregated_keys
                    ),
                    "aggregated_keys": aggregated_keys,
                    "policy_excluded_compatible_keys": (
                        policy_excluded_compatible_keys
                    ),
                    "client_non_compatible_key_counts": {
                        label: len(keys)
                        for label, keys
                        in client_non_compatible_keys.items()
                    },
                    "client_non_compatible_keys": (
                        client_non_compatible_keys
                    ),
                },
            )

            self.compatibility_manifest_written = True

        checkpoint_payload = {
            "round": int(server_round),
            "aggregation_mode": self.aggregation_mode,
            "client_labels": [
                int(label)
                for label in client_labels
            ],
            "aggregation_counts": [
                int(weight)
                for weight in weights
            ],
            "normalized_weights": [
                float(weight)
                for weight in normalized_weights
            ],
            "compatible_keys": compatible_keys,
            "aggregated_keys": aggregated_keys,
            "state_dict": new_state_dict,
        }

        latest_checkpoint_path = (
            self.run_dir
            / "server_shared_latest.pth"
        )

        torch.save(
            checkpoint_payload,
            latest_checkpoint_path,
        )

        if server_round == self.num_rounds:
            final_checkpoint_path = (
                self.run_dir
                / "server_shared_final.pth"
            )

            torch.save(
                checkpoint_payload,
                final_checkpoint_path,
            )

            log(
                INFO,
                (
                    "Saved final server shared state to "
                    f"{final_checkpoint_path}"
                ),
            )

        return state_dict_to_parameters(
            new_state_dict
        )


# Start Flower server with the custom strategy

import argparse

parser = argparse.ArgumentParser(description="Start Flower server")
parser.add_argument(
    "task",
    type=str,
    help="Determines the task to be performed. Options are: extract_fingerprint, plan_and_preprocess or train",
)
parser.add_argument(
    "-n",
    "--num_clients",
    type=int,
    default=2,
    help="Number of clients to wait for before starting the server",
)
parser.add_argument(
    "--port", type=int, required=True, help="Port number for the server to listen on"
)
parser.add_argument(
    "--num_rounds",
    type=int,
    default=None,
    help="Number of federated training rounds. If not set, defaults to 1 for preprocessing tasks and 2000 for training.",
)
parser.add_argument(
    "--aggregation_mode",
    type=str,
    choices=["weighted", "weighted_no_norm"],
    default="weighted",
    help=(
        "Aggregation mode. "
        "'weighted' uses sample-weighted FedAvg. "
        "'weighted_no_norm' uses sample-weighted FedAvg but skips "
        "normalisation-related parameters as a FedBN-inspired mode."
    ),
)

parser.add_argument(
    "--run_dir",
    type=str,
    required=True,
    help=(
        "Per-fold experiment directory used for server "
        "checkpoints, manifests and update statistics."
    ),
)

parser.add_argument(
    "--stats_every",
    type=int,
    default=10,
    help=(
        "Write update-disagreement statistics every N rounds, "
        "in addition to rounds 1-10 and the final round."
    ),
)

parser.add_argument(
    "--seed",
    type=int,
    default=2026,
    help="Seed used for deterministic server-side behaviour.",
)

parser.add_argument(
    "--initial_dataset_id",
    type=int,
    default=301,
    help=(
        "Dataset client used to supply the initial "
        "global model parameters."
    ),
)

args = parser.parse_args()
seed_everything(
    int(args.seed),
    deterministic=True,
)
num_clients = args.num_clients

if args.task == "extract_fingerprint" or args.task == "plan_and_preprocess":
    default_num_rounds = 1
    fraction_evaluate = 1.0
else:
    # nnUNet's default training length
    default_num_rounds = 2000
    # Skip federated evaluation to speed up training by one less parameters transfer
    fraction_evaluate = 0.0

num_rounds = args.num_rounds if args.num_rounds is not None else default_num_rounds

log(
    INFO,
    (
        f"Starting task={args.task} with num_rounds={num_rounds} "
        f"and aggregation_mode={args.aggregation_mode}"
    ),
)

strategy = MyStrategy(
    args.task,
    min_available_clients=num_clients,
    min_fit_clients=num_clients,
    min_evaluate_clients=num_clients,
    fraction_evaluate=fraction_evaluate,
    accept_failures=False,
    aggregation_mode=args.aggregation_mode,
    num_rounds=num_rounds,
    run_dir=args.run_dir,
    stats_every=args.stats_every,
    initial_dataset_id=args.initial_dataset_id,
)


# Start Flower server
fl.server.start_server(
    server_address=f"0.0.0.0:{args.port}",
    strategy=strategy,
    config=fl.server.ServerConfig(num_rounds=num_rounds),
    grpc_max_message_length=2147483647,  # Request a maximum message length to support sending weights from larger, more recent ResEnc architectures
)
