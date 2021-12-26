import argparse
import datetime
import json
import os


def conditional_value(var, nul, default):
    if var != nul:
        return var
    else:
        return default


def remove_keys(d, keys_to_remove):
    return {key: d[key] for key in d if key not in keys_to_remove}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-mid", "--machine_id", type=int, default=0)
    parser.add_argument("-ps", "--procs_per_machine", type=int, default=1)
    parser.add_argument("-ms", "--machines", type=int, default=1)
    parser.add_argument(
        "-ld",
        "--log_dir",
        type=str,
        default="./{}".format(datetime.datetime.now().isoformat(timespec="minutes")),
    )
    parser.add_argument("-is", "--iterations", type=int, default=1)
    parser.add_argument("-cf", "--config_file", type=str, default="config.ini")
    parser.add_argument("-ll", "--log_level", type=str, default="INFO")
    parser.add_argument("-gf", "--graph_file", type=str, default="36_nodes.edges")
    parser.add_argument("-gt", "--graph_type", type=str, default="edges")
    parser.add_argument("-ta", "--test_after", type=int, default=5)

    args = parser.parse_args()
    return args


def write_args(args, path):
    data = {
        "machine_id": args.machine_id,
        "procs_per_machine": args.procs_per_machine,
        "machines": args.machines,
        "log_dir": args.log_dir,
        "iterations": args.iterations,
        "config_file": args.config_file,
        "log_level": args.log_level,
        "graph_file": args.graph_file,
        "graph_type": args.graph_type,
        "test_after": args.test_after,
    }
    with open(os.path.join(path, "args.json"), "w") as of:
        json.dump(data, of)
