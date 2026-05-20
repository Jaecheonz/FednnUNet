import os
from io import BytesIO
from logging import INFO, WARNING
from typing import Callable, Dict, List, Optional, Tuple

import flwr as fl
import torch
from flwr.common import FitIns, MetricsAggregationFn, NDArrays, Parameters, Scalar
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy


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

        log(INFO, f"Using aggregation_mode={self.aggregation_mode}")

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
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training.

        This avoids extra client.get_parameters calls before every round.
        The actual aggregation compatibility check is still performed after
        clients return their trained parameters in aggregate_fit().
        """
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)

        fit_ins = FitIns(parameters, config)

        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )

        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
        )

        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ):
        """Aggregate client training results after each federated round."""

        if failures:
            log(WARNING, f"Round {rnd} had {len(failures)} failure(s).")

        successful_results = [result for result in results if result is not None]

        if not successful_results:
            log(WARNING, f"Round {rnd}: no successful client results received.")
            return None, {}

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

        aggregated_weights = self.aggregate_weights(successful_results)

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

    def aggregate_weights(self, results):
        """Aggregate client model weights using sample-weighted averaging."""
        dicts = [parameters_to_state_dict(res[1].parameters) for res in results]
        weights = [
            res[1].num_examples if res[1].num_examples is not None else 1
            for res in results
        ]

        if not dicts:
            log(WARNING, "aggregate_weights received no client results")
            return None

        compatible_keys = self.find_common_layers(dicts)

        if not compatible_keys:
            log(WARNING, "aggregate_weights found no compatible keys")
            return None

        log(INFO, f"Aggregating {len(dicts)} client model(s) with weights={weights}")

        new_state_dict = self.create_compatible_state_dict(
            dicts,
            compatible_keys,
            weights=weights,
        )

        return state_dict_to_parameters(new_state_dict)


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
        "'weighted_no_norm' uses sample-weighted FedAvg but skips normalisation-related parameters "
        "as a FedBN-inspired local normalisation preservation mode."
    ),
)

args = parser.parse_args()
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
    aggregation_mode=args.aggregation_mode,
)


# Start Flower server
fl.server.start_server(
    server_address=f"0.0.0.0:{args.port}",
    strategy=strategy,
    config=fl.server.ServerConfig(num_rounds=num_rounds),
    grpc_max_message_length=2147483647,  # Request a maximum message length to support sending weights from larger, more recent ResEnc architectures
)
