import logging

from decentralizepy.sharing.PartialModel import PartialModel


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
        dict_ordered=True,
        save_shared=False,
        metadata_cap=1.0,
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
            metadata_cap,
        )
        self.init_alpha = init_alpha
        self.max_alpha = max_alpha
        self.k = k

    def step(self):
        if (self.communication_round + 1) % self.k == 0:
            self.alpha += (self.max_alpha - self.init_alpha) / self.k
            self.alpha = min(self.alpha, 1.00)

        if self.alpha == 0.0:
            logging.info("Not sending/receiving data (alpha=0.0)")
            self.communication_round += 1
            return

        super().step()
