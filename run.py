import torch
import torch.multiprocessing as mp

x = [1, 2]


def f(id, a):
    print(id, x)
    print(id, a)


if __name__ == "__main__":
    x.append(3)
    mp.spawn(f, nprocs=2, args=(x,))
