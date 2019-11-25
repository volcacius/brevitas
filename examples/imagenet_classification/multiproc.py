#
# Source: https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Classification/RN50v1.5/multiproc.py
#

import sys
import subprocess
import os
import time
from argparse import ArgumentParser, REMAINDER

def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(description="PyTorch distributed training launch helper utilty that will spawn up "
                                        "multiple distributed processes")

    # Optional arguments for the launch helper
    parser.add_argument("--nnodes", type=int, default=1,
                        help="The number of nodes to use for distribute training")
    parser.add_argument("--node_rank", type=int, default=0,
                        help="The rank of the node for multi-node distributed training")
    parser.add_argument("--gpus", type=int, default=0,
                        help="The selected GPUs in the node. The number of GPUs per node have to be the same.")
    parser.add_argument("--master_addr", default="127.0.0.1", type=str,
                        help="Master node (rank 0)'s address, should be either the IP address or the hostname of "
                             "node 0, for single node multi-proc training, the --master_addr can simply be 127.0.0.1")
    parser.add_argument("--master_port", default=29500, type=int,
                        help="Master node (rank 0)'s free port that needs to be used for communciation during "
                             "distributed training")

    # Positional
    parser.add_argument("training_script", type=str,
                        help="The full path to the single GPU training program/script to be launched in parallel, "
                             "followed by all the arguments for the training script")

    # Rest from the training program
    parser.add_argument('training_script_args', nargs=REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()

    # World size in terms of number of processes
    gpus_list = [int(x.strip()) for x in args.gpus.split(',')]
    dist_world_size = len(gpus_list) * args.nnodes

    # Set PyTorch distributed related environmental variables
    current_env = os.environ.copy()
    current_env["MASTER_ADDR"] = args.master_addr
    current_env["MASTER_PORT"] = str(args.master_port)
    current_env["WORLD_SIZE"] = str(dist_world_size)
    current_env["NODE_RANK"] = str(args.node_rank)

    #  All used GPUs of the node are visible
    current_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    current_env["CUDA_VISIBLE_DEVICES"] = args.gpus

    processes = []

    for local_rank, gpu in enumerate(gpus_list):

        # Each process's rank
        dist_rank = args.nproc_per_node * args.node_rank + local_rank
        current_env["GLOBAL_RANK"] = str(dist_rank)

        # Spawn the processes passing the local rank as GPU, given that CUDA_VISIBLE_DEVICES is set
        cmd = [sys.executable,
               "-u",
               args.training_script,
               " IS_DISTRIBUTED=True",
               " GPU={}".format(local_rank)] + args.training_script_args

        print(cmd)
        process = subprocess.Popen(cmd, env=current_env, stdout=None)  # Don't redirect stdout
        processes.append(process)

    try:
        up = True
        error = False
        while up and not error:
            up = False
            for p in processes:
                ret = p.poll()
                if ret is None:
                    up = True
                elif ret != 0:
                    error = True
            time.sleep(1)

        if error:
            for p in processes:
                if p.poll() is None:
                    p.terminate()
            exit(1)

    except KeyboardInterrupt:
        for p in processes:
            p.terminate()
        raise
    except SystemExit:
        for p in processes:
            p.terminate()
        raise
    except:
        for p in processes:
            p.terminate()
        raise


if __name__ == "__main__":
    main()
