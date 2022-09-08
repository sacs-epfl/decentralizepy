import argparse
import datetime
import json
import os


def conditional_value(var, nul, default):
    """
    Set the value to default if nul.

    Parameters
    ----------
    var : any
        The value
    nul : any
        The null value. Assigns default if var == nul
    default : any
        The default value

    Returns
    -------
    type(var)
        The final value

    """
    if var != nul:
        return var
    else:
        return default


def remove_keys(d, keys_to_remove):
    """
    Removes given keys from the dict. Returns a new list.

    Parameters
    ----------
    d : dict
        The initial dictionary
    keys_to_remove : list
        List of keys to remove from dict

    Returns
    -------
    dict
        A new dictionary with the given keys removed.

    """
    return {key: d[key] for key in d if key not in keys_to_remove}


def get_args():
    """
    Utility to parse arguments.

    Returns
    -------
    args
        Command line arguments

    """
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
    parser.add_argument(
        "-wsd",
        "--weights_store_dir",
        type=str,
        default="./{}_ws".format(datetime.datetime.now().isoformat(timespec="minutes")),
    )
    parser.add_argument("-is", "--iterations", type=int, default=1)
    parser.add_argument("-cf", "--config_file", type=str, default="config.ini")
    parser.add_argument("-ll", "--log_level", type=str, default="INFO")
    parser.add_argument("-gf", "--graph_file", type=str, default="36_nodes.edges")
    parser.add_argument("-gt", "--graph_type", type=str, default="edges")
    parser.add_argument("-ta", "--test_after", type=int, default=5)
    parser.add_argument("-tea", "--train_evaluate_after", type=int, default=1)
    parser.add_argument("-ro", "--reset_optimizer", type=int, default=1)
    parser.add_argument("-ctr", "--centralized_train_eval", type=int, default=0)
    parser.add_argument("-cte", "--centralized_test_eval", type=int, default=0)
    parser.add_argument("-sm", "--server_machine", type=int, default=0)
    parser.add_argument("-sr", "--server_rank", type=int, default=-1)
    parser.add_argument("-wr", "--working_rate", type=float, default=1.0)

    args = parser.parse_args()
    return args


def write_args(args, path):
    """
    Write arguments to a json file

    Parameters
    ----------
    args : args
        Command line args
    path : str
        Location of the file to write to

    """
    data = {
        "machine_id": args.machine_id,
        "procs_per_machine": args.procs_per_machine,
        "machines": args.machines,
        "log_dir": args.log_dir,
        "weights_store_dir": args.weights_store_dir,
        "iterations": args.iterations,
        "config_file": args.config_file,
        "log_level": args.log_level,
        "graph_file": args.graph_file,
        "graph_type": args.graph_type,
        "test_after": args.test_after,
        "train_evaluate_after": args.train_evaluate_after,
        "reset_optimizer": args.reset_optimizer,
        "centralized_train_eval": args.centralized_train_eval,
        "centralized_test_eval": args.centralized_test_eval,
        "working_rate": args.working_rate,
    }
    with open(os.path.join(path, "args.json"), "w") as of:
        json.dump(data, of)


def identity(obj):
    """
    Identity function
    Parameters
    ----------
    obj
        Some object
    Returns
    -------
     obj
        The same object
    """
    return obj
