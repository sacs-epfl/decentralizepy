from decentralizepy.node.DPSGDWithPeerSampler import DPSGDWithPeerSampler


class EL_Oracle_Client(DPSGDWithPeerSampler):
    """
    This class defines the client class for Epidemic Learning with Oracle.
    The client requests the peer sampler for neighbors each round.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
