import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt


def get_stats(l):
    assert len(l) > 0
    mean_dict, stdev_dict, min_dict, max_dict = {}, {}, {}, {}
    for key in l[0].keys():
        all_nodes = [i[key] for i in l]
        all_nodes = np.array(all_nodes)
        mean = np.mean(all_nodes)
        std = np.std(all_nodes)
        min = np.min(all_nodes)
        max = np.max(all_nodes)
        mean_dict[int(key)] = mean
        stdev_dict[int(key)] = std
        min_dict[int(key)] = min
        max_dict[int(key)] = max
    return mean_dict, stdev_dict, min_dict, max_dict


def plot(means, stdevs, mins, maxs, title, label, loc):
    plt.title(title)
    plt.xlabel("communication rounds")
    x_axis = list(means.keys())
    y_axis = list(means.values())
    err = list(stdevs.values())
    plt.errorbar(x_axis, y_axis, yerr=err, label=label)
    plt.legend(loc=loc)


def plot_results(path, data_machine="machine0", data_node=0):
    folders = os.listdir(path)
    folders.sort()
    print("Reading folders from: ", path)
    print("Folders: ", folders)
    bytes_means, bytes_stdevs = {}, {}
    meta_means, meta_stdevs = {}, {}
    data_means, data_stdevs = {}, {}
    for folder in folders:
        folder_path = Path(os.path.join(path, folder))
        if not folder_path.is_dir() or "weights" == folder_path.name:
            continue
        results = []
        machine_folders = os.listdir(folder_path)
        for machine_folder in machine_folders:
            mf_path = os.path.join(folder_path, machine_folder)
            if not os.path.isdir(mf_path):
                continue
            files = os.listdir(mf_path)
            files = [f for f in files if f.endswith("_results.json")]
            for f in files:
                filepath = os.path.join(mf_path, f)
                with open(filepath, "r") as inf:
                    results.append(json.load(inf))

        with open(folder_path / data_machine / f"{data_node}_results.json", "r") as f:
            main_data = json.load(f)
        main_data = [main_data]
        # Plot Training loss
        plt.figure(1)
        means, stdevs, mins, maxs = get_stats([x["train_loss"] for x in results])
        plot(means, stdevs, mins, maxs, "Training Loss", folder, "upper right")
        df = pd.DataFrame(
            {
                "mean": list(means.values()),
                "std": list(stdevs.values()),
                "nr_nodes": [len(results)] * len(means),
            },
            list(means.keys()),
            columns=["mean", "std", "nr_nodes"],
        )
        df.to_csv(
            os.path.join(path, "train_loss_" + folder + ".csv"), index_label="rounds"
        )
        # Plot Testing loss
        plt.figure(2)
        means, stdevs, mins, maxs = get_stats([x["test_loss"] for x in main_data])
        plot(means, stdevs, mins, maxs, "Testing Loss", folder, "upper right")
        df = pd.DataFrame(
            {
                "mean": list(means.values()),
                "std": list(stdevs.values()),
                "nr_nodes": [len(results)] * len(means),
            },
            list(means.keys()),
            columns=["mean", "std", "nr_nodes"],
        )
        df.to_csv(
            os.path.join(path, "test_loss_" + folder + ".csv"), index_label="rounds"
        )
        # Plot Testing Accuracy
        plt.figure(3)
        means, stdevs, mins, maxs = get_stats([x["test_acc"] for x in main_data])
        plot(means, stdevs, mins, maxs, "Testing Accuracy", folder, "lower right")
        df = pd.DataFrame(
            {
                "mean": list(means.values()),
                "std": list(stdevs.values()),
                "nr_nodes": [len(results)] * len(means),
            },
            list(means.keys()),
            columns=["mean", "std", "nr_nodes"],
        )
        df.to_csv(
            os.path.join(path, "test_acc_" + folder + ".csv"), index_label="rounds"
        )
        plt.figure(6)
        means, stdevs, mins, maxs = get_stats([x["grad_std"] for x in results])
        plot(
            means,
            stdevs,
            mins,
            maxs,
            "Gradient Variation over Nodes",
            folder,
            "upper right",
        )
        # Plot Testing loss
        plt.figure(7)
        means, stdevs, mins, maxs = get_stats([x["grad_mean"] for x in results])
        plot(
            means, stdevs, mins, maxs, "Gradient Magnitude Mean", folder, "upper right"
        )
        # Collect total_bytes shared
        bytes_list = []
        for x in results:
            max_key = str(max(list(map(int, x["total_bytes"].keys()))))
            bytes_list.append({max_key: x["total_bytes"][max_key]})
        means, stdevs, mins, maxs = get_stats(bytes_list)
        bytes_means[folder] = list(means.values())[0]
        bytes_stdevs[folder] = list(stdevs.values())[0]

        meta_list = []
        for x in results:
            if x["total_meta"]:
                max_key = str(max(list(map(int, x["total_meta"].keys()))))
                meta_list.append({max_key: x["total_meta"][max_key]})
            else:
                meta_list.append({max_key: 0})
        means, stdevs, mins, maxs = get_stats(meta_list)
        meta_means[folder] = list(means.values())[0]
        meta_stdevs[folder] = list(stdevs.values())[0]

        data_list = []
        for x in results:
            max_key = str(max(list(map(int, x["total_data_per_n"].keys()))))
            data_list.append({max_key: x["total_data_per_n"][max_key]})
        means, stdevs, mins, maxs = get_stats(data_list)
        data_means[folder] = list(means.values())[0]
        data_stdevs[folder] = list(stdevs.values())[0]

    plt.figure(1)
    plt.savefig(os.path.join(path, "train_loss.png"), dpi=300)
    plt.figure(2)
    plt.savefig(os.path.join(path, "test_loss.png"), dpi=300)
    plt.figure(3)
    plt.savefig(os.path.join(path, "test_acc.png"), dpi=300)
    plt.figure(6)
    plt.savefig(os.path.join(path, "grad_std.png"), dpi=300)
    plt.figure(7)
    plt.savefig(os.path.join(path, "grad_mean.png"), dpi=300)
    # Plot total_bytes
    plt.figure(4)
    plt.title("Data Shared")
    x_pos = np.arange(len(bytes_means.keys()))
    plt.bar(
        x_pos,
        np.array(list(bytes_means.values())) // (1024 * 1024),
        yerr=np.array(list(bytes_stdevs.values())) // (1024 * 1024),
        align="center",
    )
    plt.ylabel("Total data shared in MBs")
    plt.xlabel("Fraction of Model Shared")
    plt.xticks(x_pos, list(bytes_means.keys()))
    plt.savefig(os.path.join(path, "data_shared.png"), dpi=300)

    # Plot stacked_bytes
    plt.figure(5)
    plt.title("Data Shared per Neighbor")
    x_pos = np.arange(len(bytes_means.keys()))
    plt.bar(
        x_pos,
        np.array(list(data_means.values())) // (1024 * 1024),
        yerr=np.array(list(data_stdevs.values())) // (1024 * 1024),
        align="center",
        label="Parameters",
    )
    plt.bar(
        x_pos,
        np.array(list(meta_means.values())) // (1024 * 1024),
        bottom=np.array(list(data_means.values())) // (1024 * 1024),
        yerr=np.array(list(meta_stdevs.values())) // (1024 * 1024),
        align="center",
        label="Metadata",
    )
    plt.ylabel("Data shared in MBs")
    plt.xlabel("Fraction of Model Shared")
    plt.xticks(x_pos, list(meta_means.keys()))
    plt.savefig(os.path.join(path, "parameters_metadata.png"), dpi=300)


def plot_parameters(path):
    plt.figure(4)
    folders = os.listdir(path)
    for folder in folders:
        folder_path = os.path.join(path, folder)
        if not os.path.isdir(folder_path):
            continue
        files = os.listdir(folder_path)
        files = [f for f in files if f.endswith("_shared_params.json")]
        for f in files:
            filepath = os.path.join(folder_path, f)
            print("Working with ", filepath)
            with open(filepath, "r") as inf:
                loaded_dict = json.load(inf)
                del loaded_dict["order"]
                del loaded_dict["shapes"]
            assert len(loaded_dict["0"]) > 0
            assert "0" in loaded_dict.keys()
            counts = np.zeros(len(loaded_dict["0"]))
            for key in loaded_dict.keys():
                indices = np.array(loaded_dict[key])
                counts = np.pad(
                    counts,
                    max(np.max(indices) - counts.shape[0], 0),
                    "constant",
                    constant_values=0,
                )
                counts[indices] += 1
            plt.plot(np.arange(0, counts.shape[0]), counts, ".")
        print("Saving scatterplot")
        plt.savefig(os.path.join(folder_path, "shared_params.png"))


if __name__ == "__main__":
    assert len(sys.argv) == 2
    plot_results(sys.argv[1])
    # plot_parameters(sys.argv[1])
