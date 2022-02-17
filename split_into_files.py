import sys
from decentralizepy.datasets.Femnist import Femnist


if __name__ == "__main__":
    f = Femnist(None, None, None)
    assert len(sys.argv) == 3
    frm = sys.argv[1]
    to = sys.argv[2]
    f.file_per_user(frm, to)
