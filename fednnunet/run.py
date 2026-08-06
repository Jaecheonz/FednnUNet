import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Convenience script to run federated training on a multi-gpu cluster
# Each node (data-center) is spawned on a determined GPU and communicates with server on the provided network port

parser = argparse.ArgumentParser()

parser.add_argument(
    "task",
    type=str,
    help="Determines the task to be performed. Options are: extract_fingerprint, plan_and_preprocess or train",
)

parser.add_argument(
    "data_centers",
    type=lambda a: json.loads("[" + a.replace(" ", ",") + "]"),
    default="",
    help="List of dataset ids" " (data centers) for federated training",
)

parser.add_argument(
    "configuration", type=str, help="Configuration that should be trained"
)
parser.add_argument(
    "fold",
    type=str,
    nargs="?",
    default=None,
    help="Fold of the 5-fold cross-validation. Should be an int between 0 and 4.",
)
parser.add_argument(
    "--gpu_memory_target",
    type=lambda a: json.loads("[" + a.replace(" ", ",") + "]"),
    default="",
    help="GPU memory target in GB"
    " for each dataset, must have the same length as data_centers",
)
parser.add_argument(
    "--port", type=int, required=True, help="Port number for the server to listen on"
)
parser.add_argument(
    "--num_rounds",
    type=int,
    default=None,
    help="Number of federated training rounds to pass to the server. If not set, server.py uses its default.",
)
parser.add_argument(
    "--aggregation_mode",
    type=str,
    choices=["weighted", "weighted_no_norm"],
    default="weighted",
    help=(
        "Aggregation mode passed to the server. "
        "'weighted' uses sample-weighted FedAvg. "
        "'weighted_no_norm' skips normalisation-related parameters as a FedBN-inspired mode."
    ),
)
parser.add_argument(
    "--server_address",
    type=str,
    default="127.0.0.1",
    help=(
        "Server address passed to clients. "
        "Defaults to 127.0.0.1 for local single-node runs."
    ),
)

parser.add_argument(
    "--seed",
    type=int,
    default=2026,
    help="Experiment seed passed to each federated client.",
)

parser.add_argument(
    "--run_dir",
    type=str,
    required=True,
    help=(
        "Base experiment directory. A fold-specific "
        "subdirectory will be created inside it."
    ),
)

parser.add_argument(
    "--stats_every",
    type=int,
    default=10,
    help=(
        "Frequency of update-disagreement statistics "
        "passed to the server."
    ),
)

parser.add_argument(
    "--initial_dataset_id",
    type=int,
    default=301,
    help=(
        "Dataset whose client supplies the deterministic "
        "initial global parameter state."
    ),
)

args, unknown = parser.parse_known_args()

# Keep dataset order stable while removing duplicates.
datasets = list(dict.fromkeys(args.data_centers))
num_clients = len(datasets)
task = args.task
fold = args.fold

gpu_memory_target = None

if args.gpu_memory_target:
    gpu_memory_target = args.gpu_memory_target
    if len(gpu_memory_target) != num_clients:
        raise ValueError("gpu_memory_target must have the same length as data_centers")
    if task != "plan_and_preprocess":
        print(
            f"WARNING: {task} task does not accept gpu_memory_target argument. It will be ignored."
        )
    # Create a dictionary with the dataset id as key and the gpu memory target as value
    gpu_memory_target_mapping = dict(zip(datasets, gpu_memory_target))

if task == "extract_fingerprint" or task == "plan_and_preprocess":
    folds = [0]
elif fold == "all":
    folds = list(range(5))
elif fold is None:
    raise ValueError("Fold must be specified for the {task} task")
else:
    folds = [int(fold)]

configuration = args.configuration
port = args.port
num_rounds = args.num_rounds
aggregation_mode = args.aggregation_mode
server_address = args.server_address
seed = int(args.seed)
stats_every = max(1, int(args.stats_every))
initial_dataset_id = int(args.initial_dataset_id)

base_run_dir = Path(args.run_dir).resolve()
base_run_dir.mkdir(
    parents=True,
    exist_ok=True,
)

multi_gpu = True

# mnms dataset
# node_mapping = {301: 2, 302: 2, 303: 3, 304: 3, 305: 4}
# fetal dataset
node_mapping = {301: 0, 302: 1}

def stop_process(process):
    """Terminate a child process cleanly, then kill it if necessary."""
    if process is None or process.poll() is not None:
        return

    process.terminate()

    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()

for current_fold in folds:
    print(
        f"Starting {task} for fold {current_fold}",
        flush=True,
    )

    fold_run_dir = (
        base_run_dir / f"fold_{current_fold}"
    )
    fold_run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    server_process = None
    client_processes = {}
    log_handles = []

    try:
        # ---------------------------------------------------------
        # Start the Flower server
        # ---------------------------------------------------------
        print("Starting server", flush=True)

        server_env = os.environ.copy()
        server_env["PYTHONUNBUFFERED"] = "1"
        server_env["EXPERIMENT_SEED"] = str(seed)
        server_env["PYTHONHASHSEED"] = str(seed)

        # The server only aggregates CPU tensors and does not need
        # to reserve one of the two requested GPUs.
        server_env["CUDA_VISIBLE_DEVICES"] = ""

        server_log_path = (
            fold_run_dir / "server.log"
        )

        server_log = server_log_path.open(
            "w",
            buffering=1,
            encoding="utf-8",
        )
        log_handles.append(server_log)

        server_command = [
            sys.executable,
            "-u",
            "fednnunet/server.py",
            task,
            "-n",
            str(num_clients),
            "--port",
            str(port),
            "--aggregation_mode",
            aggregation_mode,
            "--run_dir",
            str(fold_run_dir),
            "--stats_every",
            str(stats_every),
            "--seed",
            str(seed),
            "--initial_dataset_id",
            str(initial_dataset_id),
        ]

        if num_rounds is not None:
            server_command += [
                "--num_rounds",
                str(num_rounds),
            ]

        print(
            "Server command:",
            " ".join(server_command),
            flush=True,
        )
        print(
            f"Server log: {server_log_path}",
            flush=True,
        )
        print(
            f"Server Python: {sys.executable}",
            flush=True,
        )

        server_process = subprocess.Popen(
            server_command,
            env=server_env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Give the server time to bind to the selected port.
        time.sleep(5)

        early_server_status = (
            server_process.poll()
        )

        if early_server_status is not None:
            raise RuntimeError(
                "Flower server exited before the clients "
                f"were launched. Exit code: "
                f"{early_server_status}. "
                f"Check {server_log_path}."
            )

        # ---------------------------------------------------------
        # Start each federated client
        # ---------------------------------------------------------
        for client_dataset in datasets:
            print(
                f"Starting client {client_dataset}",
                flush=True,
            )

            client_env = os.environ.copy()
            client_env["PYTHONUNBUFFERED"] = "1"
            client_env["EXPERIMENT_SEED"] = str(seed)
            client_env["PYTHONHASHSEED"] = str(seed)

            if multi_gpu:
                if client_dataset not in node_mapping:
                    raise RuntimeError(
                        "No GPU mapping is configured for "
                        f"Dataset {client_dataset}."
                    )

                gpu = node_mapping[client_dataset]

                client_env[
                    "CUDA_VISIBLE_DEVICES"
                ] = str(gpu)

                print(
                    (
                        f"Running {task} for dataset "
                        f"{client_dataset} with fold "
                        f"{current_fold} on GPU {gpu}"
                    ),
                    flush=True,
                )

            command = [
                sys.executable,
                "-u",
                "-m",
                "fednnunet.client_entrypoints",
                "--port",
                str(port),
                "--server_address",
                server_address,
                "--seed",
                str(seed),
                "--run_dir",
                str(fold_run_dir),
                task,
            ]

            if task == "plan_and_preprocess":
                command += [
                    "-d",
                    str(client_dataset),
                ]

            elif task == "train":
                command += [
                    str(client_dataset),
                    configuration,
                    str(current_fold),
                ]

            if unknown:
                command += unknown

            if gpu_memory_target:
                command += [
                    "-gpu_memory_target",
                    str(
                        gpu_memory_target_mapping[
                            client_dataset
                        ]
                    ),
                ]

            client_log_path = (
                fold_run_dir
                / f"client_{client_dataset}.log"
            )

            client_log = client_log_path.open(
                "w",
                buffering=1,
                encoding="utf-8",
            )
            log_handles.append(client_log)

            print(
                "Client command:",
                " ".join(command),
                flush=True,
            )
            print(
                f"Client log: {client_log_path}",
                flush=True,
            )
            print(
                f"Client Python: {sys.executable}",
                flush=True,
            )

            client_processes[
                client_dataset
            ] = subprocess.Popen(
                command,
                env=client_env,
                stdout=client_log,
                stderr=subprocess.STDOUT,
                text=True,
            )

        # ---------------------------------------------------------
        # Monitor the server and clients
        # ---------------------------------------------------------
        while True:
            server_status = (
                server_process.poll()
            )

            client_statuses = {
                dataset: process.poll()
                for dataset, process
                in client_processes.items()
            }

            failed_clients = {
                dataset: status
                for dataset, status
                in client_statuses.items()
                if (
                    status is not None
                    and status != 0
                )
            }

            if failed_clients:
                stop_process(server_process)

                for process in (
                    client_processes.values()
                ):
                    stop_process(process)

                raise RuntimeError(
                    "Client process failure(s): "
                    f"{failed_clients}. "
                    f"Check logs in {fold_run_dir}."
                )

            if (
                server_status is not None
                and server_status != 0
            ):
                for process in (
                    client_processes.values()
                ):
                    stop_process(process)

                raise RuntimeError(
                    "Flower server failed with exit "
                    f"code {server_status}. "
                    f"Check {server_log_path}."
                )

            all_clients_finished = all(
                status is not None
                for status in client_statuses.values()
            )

            if all_clients_finished:
                break

            # The Flower server may finish before the clients
            # complete their final local validation. A successful
            # server exit is therefore not treated as a reason to
            # terminate the clients.
            time.sleep(5)

        # Wait for the server if the clients completed first.
        if server_process.poll() is None:
            server_status = (
                server_process.wait()
            )
        else:
            server_status = (
                server_process.returncode
            )

        client_statuses = {
            dataset: process.wait()
            for dataset, process
            in client_processes.items()
        }

        if (
            server_status != 0
            or any(
                status != 0
                for status
                in client_statuses.values()
            )
        ):
            raise RuntimeError(
                "Federated run failed: "
                f"server={server_status}, "
                f"clients={client_statuses}"
            )

        print(
            (
                f"Federated run completed successfully "
                f"for fold {current_fold}."
            ),
            flush=True,
        )
        print(
            f"Outputs: {fold_run_dir}",
            flush=True,
        )

    except KeyboardInterrupt:
        stop_process(server_process)

        for process in client_processes.values():
            stop_process(process)

        print(
            "Server and clients stopped by user.",
            flush=True,
        )

        raise SystemExit(130)

    except Exception:
        stop_process(server_process)

        for process in client_processes.values():
            stop_process(process)

        raise

    finally:
        for log_handle in log_handles:
            log_handle.close()