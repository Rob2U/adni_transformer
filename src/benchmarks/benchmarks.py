# collection of useful benchmarks
from __future__ import annotations

from typing import Any
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
import time
# used for nvitop callback
from lightning.pytorch.utilities import rank_zero_only  # pylint: disable=import-error
from lightning.pytorch.utilities.exceptions import (  # pylint: disable=import-error
    MisconfigurationException,
)
from nvitop.api import libnvml
from nvitop.callbacks.utils import get_devices_by_logical_ids, get_gpu_stats
from nvitop import Device, CudaDevice, MiB


class SamplesPerSecondBenchmark(Callback):
    def __init__(self, batch_interval=50):
        super().__init__()
        self.batch_interval = batch_interval    # defines the limit of batches for when the metric is computed the next time
        self.start_time = None
        self.num_samples = 0
        #self.batches_seen = 0

    def on_train_epoch_start(self, trainer, pl_module):
        self.start_time = time.time()
        self.num_samples = 0
        self.batch_interval = min(self.batch_interval, len(trainer.train_dataloader)//10)   # set the observation interval to 10% of the batches of one epoch
        
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


class GpuMetricsBenchmark(Callback):  # pylint: disable=too-many-instance-attributes
    def __init__(  # pylint: disable=too-many-arguments
        self,
        memory_utilization: bool = True,
        gpu_utilization: bool = True,
        intra_step_time: bool = True,
        inter_step_time: bool = True,
        fan_speed: bool = True,
        temperature: bool = True,
        power_usage: bool = True,
        power_relative: bool = True,
    ) -> None:
        super().__init__()

        try:
            libnvml.nvmlInit()
        except libnvml.NVMLError as ex:
            raise MisconfigurationException(
                'Cannot use GpuStatsLogger callback because NVIDIA driver is not installed.',
            ) from ex

        self._memory_utilization = memory_utilization
        self._gpu_utilization = gpu_utilization
        self._intra_step_time = intra_step_time
        self._inter_step_time = inter_step_time
        self._fan_speed = fan_speed
        self._temperature = temperature
        self._power_usage = power_usage
        self._power_relative = power_relative

    def on_train_start(self, trainer, pl_module) -> None:
        if not trainer.logger:
            raise MisconfigurationException(
                'Cannot use GpuStatsLogger callback with Trainer that has no logger.',
            )

        if trainer.strategy.root_device.type != 'cuda':
            raise MisconfigurationException(
                f'You are using GpuStatsLogger but are not running on GPU. '
                f'The root device type is {trainer.strategy.root_device.type}.',
            )

        #device_ids = trainer.data_parallel_device_ids
        try:
            #self._devices = get_devices_by_logical_ids(device_ids, unique=True)
            self._devices = Device.cuda.all()
        except (libnvml.NVMLError, RuntimeError) as ex:
            raise ValueError(
                f'Cannot use GpuStatsLogger callback because devices unavailable. '
                f'Received: `gpus={device_ids}`',
            ) from ex

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._snap_intra_step_time = None
        self._snap_inter_step_time = None

    @rank_zero_only
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:  # pylint: disable=arguments-differ
        if self._intra_step_time:
            self._snap_intra_step_time = time.monotonic()

        logs = self._get_gpu_stats()

        if self._inter_step_time and self._snap_inter_step_time:
            # First log at beginning of second step
            logs['batch_time/inter_step (ms)'] = 1000.0 * (
                time.monotonic() - self._snap_inter_step_time
            )

        trainer.logger.log_metrics(logs, step=trainer.global_step)

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:  # pylint: disable=arguments-differ
        if self._inter_step_time:
            self._snap_inter_step_time = time.monotonic()

        logs = self._get_gpu_stats()

        if self._intra_step_time and self._snap_intra_step_time:
            logs['batch_time/intra_step (ms)'] = 1000.0 * (
                time.monotonic() - self._snap_intra_step_time
            )

        trainer.logger.log_metrics(logs, step=trainer.global_step)

    def get_all_gpu_stats(
        self,
        devices: list[Device],
        memory_utilization: bool = True,
        gpu_utilization: bool = True,
        fan_speed: bool = True,
        temperature: bool = True,
        power_usage: bool = True,
        power_relative: bool = True,
        ) -> dict[str, float]:
        """Get the GPU status from NVML queries."""
        stats = {}
        for device in devices:
            prefix = f'gpu_id: {device.cuda_index}'
            if device.cuda_index != device.physical_index:
                prefix += f' (physical index: {device.physical_index})'
            with device.oneshot():
                if memory_utilization or gpu_utilization:
                    utilization = device.utilization_rates()
                    if memory_utilization:
                        stats[f'{prefix}/utilization.memory (%)'] = float(utilization.memory)
                    if gpu_utilization:
                        stats[f'{prefix}/utilization.gpu (%)'] = float(utilization.gpu)
                if memory_utilization:
                    stats[f'{prefix}/memory.used (MiB)'] = float(device.memory_used()) / MiB
                    stats[f'{prefix}/memory.free (MiB)'] = float(device.memory_free()) / MiB
                if fan_speed:
                    stats[f'{prefix}/fan.speed (%)'] = float(device.fan_speed())
                if temperature:
                    stats[f'{prefix}/temperature.gpu (C)'] = float(device.fan_speed())
                if power_usage:
                    stats[f'{prefix}/power.used (W)'] = float(device.power_usage()) / 1000
                if power_relative:
                    stats[f'{prefix}/power.relative (%)'] = float(device.power_usage()/device.power_limit())

        return stats

    def _get_gpu_stats(self) -> dict[str, float]:
        """Get the gpu status from NVML queries."""
        return self.get_all_gpu_stats(       # this was get_gpu_stats before
        #return get_gpu_stats(
            self._devices,
            self._memory_utilization,
            self._gpu_utilization,
            self._fan_speed,
            self._temperature,
            self._power_usage,
            self._power_relative,
        )