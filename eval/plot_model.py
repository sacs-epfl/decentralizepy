import json
import os
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from pyexpat import model


def plot(x, y, label, *args):
    plt.plot(x, y, *args, label=label)
    plt.legend()


def reject_outliers(data, m=2.0):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.0)
    return data[s < m]


def plot_model(path, title):
    model_path = os.path.join(path, "plots")
    Path(model_path).mkdir(parents=True, exist_ok=True)
    files = [f for f in os.listdir(path) if f.endswith("json")]
    for file in files:
        filepath = os.path.join(path, file)
        with open(filepath, "r") as inf:
            model_vec = json.load(inf)
            del model_vec["order"]
            del model_vec["shapes"]
            model_vec = np.array(model_vec[list(model_vec.keys())[0]])
        num_elements = model_vec.shape[0]
        x_axis = np.arange(1, num_elements + 1)
        plt.clf()
        plt.title(title)
        plot(x_axis, model_vec, "unsorted", ".")
        model_vec = np.sort(model_vec)
        plot(x_axis, model_vec, "sorted")
        plt.savefig(os.path.join(model_path, file[0:-5]))


def plot_ratio(path_change, path_val, title):
    model_path = os.path.join(path_change, "plots_ratio")
    Path(model_path).mkdir(parents=True, exist_ok=True)
    files_change = [f for f in os.listdir(path_change) if f.endswith("json")]
    files_val = [f for f in os.listdir(path_val) if f.endswith("json")]
    for i, file in enumerate(files_change):
        print("Processed ", file)
        filepath_change = os.path.join(path_change, file)
        filepath_val = os.path.join(path_val, files_val[i])
        with open(filepath_change, "r") as inf:
            model_change = json.load(inf)
            del model_change["order"]
            del model_change["shapes"]
            model_change = np.array(model_change[list(model_change.keys())[0]])
        with open(filepath_val, "r") as inf:
            model_val = json.load(inf)
            del model_val["order"]
            del model_val["shapes"]
            model_val = np.array(model_val[list(model_val.keys())[0]])
        num_elements = model_val.shape[0]
        x_axis = np.arange(1, num_elements + 1)
        plt.clf()
        plt.title(title)
        model_vec = np.divide(
            model_change,
            model_val,
            out=np.zeros_like(model_change),
            where=model_val != 0.0,
        )
        model_vec = reject_outliers(model_vec)
        num_elements = model_vec.shape[0]
        x_axis = np.arange(1, num_elements + 1)
        plot(x_axis, model_vec, "unsorted", ".")
        model_vec = np.sort(model_vec)
        plot(x_axis, model_vec, "sorted")
        plt.savefig(os.path.join(model_path, file[0:-5]))


if __name__ == "__main__":
    assert len(sys.argv) == 3
    plot_model(
        os.path.join(sys.argv[1], "model_change", sys.argv[2]), "Change in Weights"
    )
    plot_model(os.path.join(sys.argv[1], "model_val", sys.argv[2]), "Model Parameters")
    plot_ratio(
        os.path.join(sys.argv[1], "model_change", sys.argv[2]),
        os.path.join(sys.argv[1], "model_val", sys.argv[2]),
        "Ratio",
    )
