import argparse
import json
import subprocess
import sys
import os
import time

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

args, unknown = parser.parse_known_args()

datasets = set(args.data_centers)
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

multi_gpu = True

# mnms dataset
# node_mapping = {301: 2, 302: 2, 303: 3, 304: 3, 305: 4}
# fetal dataset
node_mapping = {301: 0, 302: 1}

for fold in folds:
    print(f"Starting {task} for fold {fold}")
    
    server_process = None
    client_processes = []
    
    try:
        print("Starting server")
        server_env = os.environ.copy()
        if multi_gpu:
            server_env["CUDA_VISIBLE_DEVICES"] = "0"

        print(f"Server python: {sys.executable}")
        server_env["PYTHONUNBUFFERED"] = "1"

        server_log = open(f"server_fold{fold}.log", "w", buffering=1)

        server_process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "fednnunet/server.py",
                task,
                "-n",
                str(num_clients),
                "--port",
                str(port),
            ],
            env=server_env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            text=True,
        )

        time.sleep(5)
        for client_dataset in datasets:
            print("Starting client " + str(client_dataset))

            client_env = os.environ.copy()
            if multi_gpu:
                gpu = node_mapping[client_dataset]
                client_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                print(
                    f"Running {task} for dataset {client_dataset} with fold {fold} on GPU {gpu}"
                )

            client_env["PYTHONUNBUFFERED"] = "1"

            command = [
                sys.executable,
                "-u",
                "-m",
                "fednnunet.client_entrypoints",
                "--port",
                str(port),
                task,
            ]

            if task == "plan_and_preprocess":
                command += ["-d", str(client_dataset)]
            elif task == "train":
                command += [str(client_dataset), configuration, str(fold)]

            if unknown:
                command += unknown

            if gpu_memory_target:
                command += [
                    "-gpu_memory_target",
                    str(gpu_memory_target_mapping[client_dataset]),
                ]

            print(" ".join(command))
            print(f"Client python: {sys.executable}")
            client_log = open(f"client_{client_dataset}_fold{fold}.log", "w", buffering=1)

            client_processes.append(
                subprocess.Popen(
                    command,
                    env=client_env,
                    stdout=client_log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            )
        time.sleep(3)
    
        for i, client_process in enumerate(client_processes):
            print(f"\n=== Output from client process {i} ===", flush=True)
            ret = client_process.poll()
            print(f"Client process {i} current status: {ret}", flush=True)

        server_process.wait()

    except KeyboardInterrupt:
        if server_process is not None:
            server_process.terminate()
            server_process.wait()

        for client_process in client_processes:
            client_process.terminate()
            client_process.wait()

        print("Server and clients stopped")
