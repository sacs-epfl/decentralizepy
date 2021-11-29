import logging

from decentralizepy.training.Training import Training


class GradientAccumulator(Training):
    def __init__(
        self, model, optimizer, loss, epochs_per_round="", batch_size="", shuffle=""
    ):
        super().__init__(model, optimizer, loss, epochs_per_round, batch_size, shuffle)

    def train(self, dataset):
        """
        One training iteration with accumulation of gradients in model.accumulated_gradients
        Parameters
        ----------
        dataset : decentralizepy.datasets.Dataset
            The training dataset. Should implement get_trainset(batch_size, shuffle)
        """
        trainset = dataset.get_trainset(self.batch_size, self.shuffle)
        self.model.accumulated_gradients = []

        for epoch in range(self.epochs_per_round):
            epoch_loss = 0.0
            count = 0
            for data, target in trainset:
                self.model.zero_grad()
                output = self.model(data)
                loss_val = self.loss(output, target)
                epoch_loss += loss_val.item()
                loss_val.backward()
                self.model.accumulated_gradients.append(
                    {
                        k: v.grad.clone().detach()
                        for k, v in zip(
                            self.model.state_dict(), self.model.parameters()
                        )
                    }
                )
                self.optimizer.step()
                count += 1
            logging.info("Epoch: {} loss: {}".format(epoch, epoch_loss / count))
