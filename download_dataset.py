import torchvision

if __name__ == "__main__":
    torchvision.datasets.CIFAR10(root="./eval/data/", train=True, download=True)
    torchvision.datasets.CIFAR10(root="./eval/data/", train=False, download=True)

    # TODO: download the other datasets
