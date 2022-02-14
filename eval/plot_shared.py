import json
import os
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


def plot(x, y, label, *args):
    plt.plot(x, y, *args, label=label)
    plt.legend()

def plot_shared(path, title):
    model_path = os.path.join(path, "plots")
    Path(model_path).mkdir(parents=True, exist_ok=True)
    files = [f for f in os.listdir(path) if f.endswith("json")]
    assert len(files) > 0
    for i, file in enumerate(files):
        filepath = os.path.join(path, file)
        with open(filepath, "r") as inf:
            model_vec = json.load(inf)
            del model_vec["order"]
            if i == 0:
                total_params = 0
                for l in model_vec["shapes"].values():
                    current_params = 1
                    for v in l:
                        current_params *= v
                    total_params += current_params
                print("Total Params: ", str(total_params))
                shared_count = np.zeros(total_params, dtype = int)
            del model_vec["shapes"]
            model_vec = np.array(model_vec[list(model_vec.keys())[0]])
        shared_count[model_vec] += 1
    print("sum: ", np.sum(shared_count))
    num_elements = shared_count.shape[0]
    x_axis = np.arange(1, num_elements + 1)
    plt.clf()
    plt.title(title)
    plot(x_axis, shared_count, "unsorted", ".")
    shared_count = np.sort(shared_count)
    plot(x_axis, shared_count, "sorted")
    plt.savefig(os.path.join(model_path, "shared_plot.png"))


if __name__ == "__main__":
    assert len(sys.argv) == 2
    plot_shared(sys.argv[1], "Shared Parameters")
