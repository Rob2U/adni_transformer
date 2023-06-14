from typing import Any
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
import time


class SamplesPerSecondBenchmark(Callback):
    def __init__(self, batch_interval=100):
        super().__init__()
        self.batch_interval = batch_interval    # defines the limit of batches for when the metric is computed the next time
        self.start_time = None
        self.num_samples = 0
        #self.batches_seen = 0

    def on_train_epoch_start(self, trainer, pl_module):
        self.start_time = time.time()
        self.num_samples = 0
        #self.batches_seen = 0
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # calculate how many samples have been passed so far
        _, label_batch = batch
        batch_size = label_batch.size(0)
        self.num_samples += batch_size

        # compute and log the samplesPerSecond based on the number of samples
        if (batch_idx+1) % self.batch_interval == 0:  # log every batch_interval batches
            curr_time = time.time()
            elapsed_time = curr_time - self.start_time
            samplesPerSecond = self.num_samples / elapsed_time
            trainer.logger.experiment.log({"SamplesPerSecond": samplesPerSecond})
    
    """ def on_validation_end(self, trainer, pl_module):
        # do not take the time for validation into account when calculating the metric
        self.start_time = time.time()
        self.num_samples = 0 """

