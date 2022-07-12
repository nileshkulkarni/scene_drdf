import os
import os.path as osp
import time

import ipdb
import torch
from loguru import logger
from torch.profiler import ProfilerActivity, profile, record_function

from ..html_utils import base_html as base_html_utils
from ..html_utils import scene_html
from ..nnutils import net_blocks as nb
from ..utils import elastic as elastic_utils
from ..utils import tensorboard_utils
from ..utils.timer import Timer


class BaseTrainer:
    def __init__(self, opts):
        self.opts = opts
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.save_dir = osp.join(opts.CHECKPOINT_DIR, opts.NAME)
        self.log_dir = osp.join(opts.LOGGING_DIR, opts.NAME)
        tf_dir = osp.join(opts.TENSORBOARD_DIR, opts.NAME)
        self.sc_dict = {}
        self.dataloader = None

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        log_file = os.path.join(self.save_dir, "opts.log")
        with open(log_file, "w") as f:
            f.write(opts.dump())

        self.dataloader = None
        self.model = None
        self.optimizer = None
        self.scheduler = None

        tf_dir = osp.join(opts.TENSORBOARD_DIR, opts.NAME)
        self.tensorboard_writer = tensorboard_utils.TensorboardWriter(tf_dir)
        return

    def init_dataset(
        self,
    ):
        raise NotImplementedError

    def init_model(
        self,
    ):
        raise NotImplementedError

    def init_optimizer(
        self,
    ):
        raise NotImplementedError

    def initialize(
        self,
    ):
        opts = self.opts
        self.init_dataset()  ## define self.dataloader
        self.init_model()  ## define self.model
        self.init_optimizer()  ## define self.optimizer
        self.define_criterion()  ## define self.criterion
        return

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_id=None):
        print("Saving to deice ")
        save_filename = f"{network_label}_net_{epoch_label}.pth"
        save_path = os.path.join(self.save_dir, save_filename)
        if isinstance(network, torch.nn.DataParallel):
            torch.save(network.module.state_dict(), save_path)
        elif isinstance(network, torch.nn.parallel.DistributedDataParallel):
            torch.save(network.module.state_dict(), save_path)
        else:
            torch.save(network.state_dict(), save_path)
        # if gpu_id is not None and torch.cuda.is_available():
        #     network.cuda(device=gpu_id)
        return

    def load_network(self, network, network_label, epoch_label, network_dir=None):
        save_filename = f"{network_label}_net_{epoch_label}.pth"
        if network_dir is None:
            network_dir = self.save_dir
        save_path = os.path.join(network_dir, save_filename)
        print(f"Loading model : {save_path}")
        network.load_state_dict(torch.load(save_path, map_location="cpu"))
        return

    def set_input(self, batch):
        raise NotImplementedError

    def log_step(self, total_steps):
        raise NotImplementedError

    def log_visuals(self, total_steps):
        raise NotImplementedError

    def forward(
        self,
    ):
        raise NotImplementedError

    def backward(
        self,
    ):
        raise NotImplementedError

    def train(
        self,
    ):
        opts = self.opts
        total_steps = 0
        optim_steps = 0
        self.real_iter = 0
        self.valid_frac = 0
        dataset_size = len(self.dataloader)
        start_time = time.time()
        loss_went_up = 0
        self.lr = opts.OPTIM.LEARNING_RATE
        num_steps = len(self.dataloader) * opts.TRAIN.NUM_EPOCHS

        elastic_checkpoint = osp.join(self.save_dir, "checkpoint_elastic.pth")
        save_state, loaded = elastic_utils.load_checkpoint(
            checkpoint_file=elastic_checkpoint,
            device_id=0,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            enable_checkpointing=opts.ENABLE_ELASTIC_CHECKPOINTING,
        )

        start_epoch = save_state.epoch + 1
        total_steps = step_index = save_state.step_index + 1
        if opts.TRAIN.NUM_PRETRAIN_EPOCHS > 0 and not loaded:
            start_epoch = opts.TRAIN.NUM_PRETRAIN_EPOCHS
        logger.info(f"=> start_epoch: {start_epoch}")
        self.model.train()
        self.real_iter = 0
        self.dataset_size = len(self.dataloader)
        self.smoothed_total_loss = 0.0
        self.val_loss = 1.0
        self.html_vis = html_vis = scene_html.HTMLWriter(opts)
        exp_html_writer = base_html_utils.ExpHTMLWriter(opts, self.html_vis)
        self.total_steps = 0
        batch_timer = Timer()
        for epoch in range(start_epoch, opts.TRAIN.NUM_EPOCHS):
            self.curr_epoch = epoch
            epoch_iter = 0
            datqa_iterator = iter(self.dataloader)
            self.time_per_batch = 0.0
            for i in range(len(self.dataloader)):
                batch_timer.tic()
                batch = next(datqa_iterator)
                batch_timer.toc()
                self.time_per_batch += batch_timer.get_time()

                # logger.debug(f"Time per batch: {self.time_per_batch}")
                # for i, batch in enumerate(self.dataloader):
                self.invalid_batch = False

                if epoch > opts.TRAIN.BN_OFF_EPOCH:
                    nb.turnNormOff(self.model)

                self.set_input(batch)
                if not self.invalid_batch:
                    self.real_iter += 1
                    self.forward()
                    self.backward()
                    total_steps += 1
                    epoch_iter += 1
                    self.total_steps = total_steps
                    self.epoch_iter = epoch_iter
                    if total_steps % opts.LOGGING.PRINT_FREQ == 0:
                        ## logging frequency. Log stuff.
                        self.log_step(total_steps, epoch, epoch_iter)

                    # if total_steps % opts.LOGGING.VALID_FREQ == 0:
                    #     ## validation loss
                    #     logger.info(f"=> Validation at step {total_steps}")
                    #     self.val_loss = self.val()

                    if total_steps % opts.LOGGING.SAVE_CHECKPOINT_FREQ == 0:
                        ## save frequency. Save stuff.
                        self.save_network(self.model, "model", f"epoch_{epoch}")
                        self.save_network(self.optimizer, "optimizer", f"epoch_{epoch}")
                        logger.info(
                            f"=> Epoch: {epoch}/{opts.TRAIN.NUM_EPOCHS}, "
                            f"  Step: {total_steps}/{num_steps}, "
                        )

                    if (
                        total_steps % opts.LOGGING.SAVE_VIS_FREQ == 0
                        and opts.LOGGING.SAVE_VIS
                    ):
                        self.log_visuals(total_steps)

            save_state.step_index = step_index
            save_state.epoch = epoch
            elastic_utils.save_checkpoint(
                save_state,
                elastic_checkpoint,
                enable_checkpointing=opts.ENABLE_ELASTIC_CHECKPOINTING,
            )
        return
