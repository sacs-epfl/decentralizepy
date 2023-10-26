from decentralizepy.node.PeerSamplerDynamic import PeerSamplerDynamic


class EL_Oracle_TopologyBuilder(PeerSamplerDynamic):
    """
    This class defines the topology builder that responds to neighbor requests from the clients.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
