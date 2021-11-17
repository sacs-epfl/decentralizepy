from torch import multiprocessing as mp

from decentralizepy.communication.Communication import Communication
from decentralizepy.mappings.Linear import Linear


def f(rank, m_id, total_procs, filePath, mapping):
    c = Communication(rank, m_id, total_procs, filePath, mapping)

    c.connect_neighbours([i for i in range(total_procs) if i != c.uid])
    send = {}
    send["message"] = "Hi I am rank {}".format(rank)
    c.send((c.uid + 1) % total_procs, send)
    print(c.uid, c.receive())


if __name__ == "__main__":
    l = Linear(2, 2)
    m_id = int(input())
    mp.spawn(fn=f, nprocs=2, args=[m_id, 4, "ip_addr.json", l])
