import sys

from decentralizepy.datasets.Reddit import Reddit
from decentralizepy.mappings import Linear

if __name__ == "__main__":
    mapping = Linear(6, 16)
    f = Reddit(0, 0, mapping)
    assert len(sys.argv) == 3
    frm = sys.argv[1]
    to = sys.argv[2]
    f.file_per_user(frm, to)
