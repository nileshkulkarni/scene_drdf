import os
import sched
import shutil

import torch
from torch.nn.parallel import DistributedDataParallel


class State:
    """
    Container for objects that we want to checkpoint. Represents the
    current "state" of the worker. This object is mutable.
    """

    def __init__(self, model, optimizer, scheduler):
        self.epoch = -1
        self.step_index = -1
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def capture_snapshot(self):
        """
        Essentially a ``serialize()`` function, returns the state as an
        object compatible with ``torch.save()``. The following should work
        ::
        snapshot = state_0.capture_snapshot()
        state_1.apply_snapshot(snapshot)
        assert state_0 == state_1
        """
        return {
            "epoch": self.epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            # "scheduler": self.scheduler.state_dict(),
        }

    def apply_snapshot(self, obj, device_id):
        """
        The complimentary function of ``capture_snapshot()``. Applies the
        snapshot object that was returned by ``capture_snapshot()``.
        This function mutates this state object.
        """

        self.epoch = obj["epoch"]
        self.state_dict = obj["state_dict"]
        self.model.load_state_dict(obj["state_dict"])
        self.optimizer.load_state_dict(obj["optimizer"])
        # self.scheduler.load_state_dict(obj["scheduler"])

    def save(self, f):
        torch.save(self.capture_snapshot(), f)

    def load(self, f, device_id):
        # Map model to be loaded to specified single gpu.
        snapshot = torch.load(f, map_location=f"cuda:{device_id}")
        self.apply_snapshot(snapshot, device_id)


def load_checkpoint(
    checkpoint_file: str,
    device_id: int,
    model: DistributedDataParallel,
    optimizer,  # SGD
    scheduler,
    enable_checkpointing=True,
) -> State:
    """
    Loads a local checkpoint (if any). Otherwise, checks to see if any of
    the neighbors have a non-zero state. If so, restore the state
    from the rank that has the most up-to-date checkpoint.
    .. note:: when your job has access to a globally visible persistent storage
              (e.g. nfs mount, S3) you can simply have all workers load
              from the most recent checkpoint from such storage. Since this
              example is expected to run on vanilla hosts (with no shared
              storage) the checkpoints are written to local disk, hence
              we have the extra logic to broadcast the checkpoint from a
              surviving node.
    """

    state = State(model, optimizer, scheduler)
    loaded = False
    if os.path.isfile(checkpoint_file) and enable_checkpointing:
        print(f"=> loading checkpoint file: {checkpoint_file}")
        state.load(checkpoint_file, device_id)
        loaded = True
        print(f"=> loaded checkpoint file: {checkpoint_file}")
    else:
        print(f"=> No previous checkpoint to load from")
    return state, loaded


def save_checkpoint(state: State, filename: str, enable_checkpointing=True):
    if enable_checkpointing:
        checkpoint_dir = os.path.dirname(filename)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # save to tmp, then commit by moving the file in case the job
        # gets interrupted while writing the checkpoint
        tmp_filename = filename + "_elastic.tmp"
        torch.save(state.capture_snapshot(), tmp_filename)
        os.rename(tmp_filename, filename)
        print(f"=> saved checkpoint for epoch {state.epoch} at {filename}")
