import distutils
import json
import os
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot(x_axis, means, stdevs, pos, nb_plots, title, label, loc, xlabel):
    cmap = plt.get_cmap("gist_rainbow")
    plt.title(title)
    plt.xlabel(xlabel)
    y_axis = list(means)
    err = list(stdevs)
    print("label:", label)
    print("color: ", cmap(1 / nb_plots * pos))
    plt.errorbar(
        list(x_axis), y_axis, yerr=err, label=label, color=cmap(1 / nb_plots * pos)
    )
    plt.legend(loc=loc)


def plot_results(path, epochs, global_epochs="True"):
    if global_epochs.lower() in ["true", "1", "t", "y", "yes"]:
        global_epochs = True
    else:
        global_epochs = False
    epochs = int(epochs)
    # rounds = int(rounds)
    folders = os.listdir(path)
    folders.sort()
    print("Reading folders from: ", path)
    print("Folders: ", folders)
    bytes_means, bytes_stdevs = {}, {}
    meta_means, meta_stdevs = {}, {}
    data_means, data_stdevs = {}, {}

    files = os.listdir(path)
    files = [f for f in files if f.endswith(".csv")]
    train_loss = sorted([f for f in files if f.startswith("train_loss")])
    test_acc = sorted([f for f in files if f.startswith("test_acc")])
    test_loss = sorted([f for f in files if f.startswith("test_loss")])
    min_losses = []
    for i, f in enumerate(train_loss):
        filepath = os.path.join(path, f)
        with open(filepath, "r") as inf:
            results_csv = pd.read_csv(inf)
        # Plot Training loss
        plt.figure(1)
        if global_epochs:
            rounds = results_csv["rounds"].iloc[0]
            print("Rounds: ", rounds)
            results_cr = results_csv[results_csv.rounds <= epochs * rounds]
            means = results_cr["mean"].to_numpy()
            stdevs = results_cr["std"].to_numpy()
            x_axis = (
                results_cr["rounds"].to_numpy() / rounds
            )  # list(np.arange(0, len(means), 1))
            x_label = "global epochs"
        else:
            results_cr = results_csv[results_csv.rounds <= epochs]
            means = results_cr["mean"].to_numpy()
            stdevs = results_cr["std"].to_numpy()
            x_axis = results_cr["rounds"].to_numpy()
            x_label = "communication rounds"
        min_losses.append(np.min(means))

        plot(
            x_axis,
            means,
            stdevs,
            i,
            len(train_loss),
            "Training Loss",
            f[len("train_loss") + 1 : -len(":2022-03-24T17:54.csv")],
            "upper right",
            x_label,
        )

    min_tlosses = []
    for i, f in enumerate(test_loss):
        filepath = os.path.join(path, f)
        with open(filepath, "r") as inf:
            results_csv = pd.read_csv(inf)
        if global_epochs:
            rounds = results_csv["rounds"].iloc[0]
            print("Rounds: ", rounds)
            results_cr = results_csv[results_csv.rounds <= epochs * rounds]
            means = results_cr["mean"].to_numpy()
            stdevs = results_cr["std"].to_numpy()
            x_axis = (
                results_cr["rounds"].to_numpy() / rounds
            )  # list(np.arange(0, len(means), 1))
            x_label = "global epochs"
        else:
            results_cr = results_csv[results_csv.rounds <= epochs]
            means = results_cr["mean"].to_numpy()
            stdevs = results_cr["std"].to_numpy()
            x_axis = results_cr["rounds"].to_numpy()
            x_label = "communication rounds"
        print("x axis:", x_axis)
        min_tlosses.append(np.min(means))
        # Plot Testing loss
        plt.figure(2)
        plot(
            x_axis,
            means,
            stdevs,
            i,
            len(test_loss),
            "Testing Loss",
            f[len("test_loss") + 1 : -len(":2022-03-24T17:54.csv")],
            "upper right",
            x_label,
        )

    max_taccs = []
    for i, f in enumerate(test_acc):
        filepath = os.path.join(path, f)
        with open(filepath, "r") as inf:
            results_csv = pd.read_csv(inf)
        if global_epochs:
            rounds = results_csv["rounds"].iloc[0]
            print("Rounds: ", rounds)
            results_cr = results_csv[results_csv.rounds <= epochs * rounds]
            means = results_cr["mean"].to_numpy()
            stdevs = results_cr["std"].to_numpy()
            x_axis = (
                results_cr["rounds"].to_numpy() / rounds
            )  # list(np.arange(0, len(means), 1))
            x_label = "global epochs"
        else:
            results_cr = results_csv[results_csv.rounds <= epochs]
            means = results_cr["mean"].to_numpy()
            stdevs = results_cr["std"].to_numpy()
            x_axis = results_cr["rounds"].to_numpy()
            x_label = "communication rounds"
        max_taccs.append(np.max(means))
        # Plot Testing Accuracy
        plt.figure(3)
        plot(
            x_axis,
            means,
            stdevs,
            i,
            len(test_acc),
            "Testing Accuracy",
            f[len("test_acc") + 1 : -len(":2022-03-24T17:54.csv")],
            "lower right",
            x_label,
        )

    names_loss = [
        f[len("train_loss") + 1 : -len(":2022-03-24T17:54.csv")] for f in train_loss
    ]
    names_acc = [
        f[len("test_acc") + 1 : -len(":2022-03-24T17:54.csv")] for f in test_acc
    ]
    print(names_loss)
    print(names_acc)
    pf = pd.DataFrame(
        {
            "test_accuracy": max_taccs,
            "test_losses": min_tlosses,
            "train_losses": min_losses,
        },
        names_loss,
    )
    pf = pf.sort_values(["test_accuracy"], 0, ascending=False)
    pf.to_csv(os.path.join(path, "best_results.csv"))

    plt.figure(1)
    plt.savefig(os.path.join(path, "ge_train_loss.png"), dpi=300)
    plt.figure(2)
    plt.savefig(os.path.join(path, "ge_test_loss.png"), dpi=300)
    plt.figure(3)
    plt.savefig(os.path.join(path, "ge_test_acc.png"), dpi=300)


if __name__ == "__main__":
    assert len(sys.argv) == 4
    # The args are:
    # 1: the folder with the csv files,
    # 2: the number of epochs / comm rounds to plot for,
    # 3: True/False with True meaning plot global epochs and False plot communication rounds
    print(sys.argv[1], sys.argv[2], sys.argv[3])
    plot_results(sys.argv[1], sys.argv[2], sys.argv[3])
