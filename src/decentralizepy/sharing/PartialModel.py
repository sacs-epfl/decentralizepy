from decentralizepy.sharing.Sharing import Sharing

class PartialModel(Sharing):
    def __init__(self, rank, machine_id, communication, mapping, graph, model, dataset):
        super().__init__(rank, machine_id, communication, mapping, graph, model, dataset)
        
