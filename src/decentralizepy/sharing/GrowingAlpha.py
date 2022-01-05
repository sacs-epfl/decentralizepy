import logging

from decentralizepy.sharing.PartialModel import PartialModel
from decentralizepy.sharing.Sharing import Sharing


class GrowingAlpha(PartialModel):
    def __init__(
        self,
        rank,
        machine_id,
        communication,
        mapping,
        graph,
        model,
        dataset,
        log_dir,
        init_alpha=0.0,
        max_alpha=1.0,
        k=10,
        metadata_cap=0.6,
        dict_ordered=True,
        save_shared=False,
    ):
        super().__init__(
            rank,
            machine_id,
            communication,
            mapping,
            graph,
            model,
            dataset,
            log_dir,
            init_alpha,
            dict_ordered,
            save_shared,
        )
        self.init_alpha = init_alpha
        self.max_alpha = max_alpha
        self.k = k
        self.metadata_cap = metadata_cap
        self.base = None

    def step(self):
        if (self.communication_round + 1) % self.k == 0:
            self.alpha += (self.max_alpha - self.init_alpha) / self.k

        if self.alpha == 0.0:
            logging.info("Not sending/receiving data (alpha=0.0)")
            self.communication_round += 1
            return

        if self.alpha > self.metadata_cap:
            if self.base == None:
                self.base = Sharing(
                    self.rank,
                    self.machine_id,
                    self.communication,
                    self.mapping,
                    self.graph,
                    self.model,
                    self.dataset,
                    self.log_dir,
                )
                self.base.communication_round = self.communication_round
            self.base.step()
        else:
            super().step()
