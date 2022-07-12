import json
import ntpath
import os
import os.path as osp
import pdb
import time

import ipdb
import numpy as np
import visdom
from loguru import logger

# server = 'http://fouheylab.eecs.umich.edu'
# flags.DEFINE_string('env_name', 'main', 'env name for experiments')
# flags.DEFINE_integer('display_id', 1, 'Display Id')
# flags.DEFINE_integer('display_winsize', 256, 'Display Size')
# flags.DEFINE_integer('display_port', 8098, 'Display port')
# flags.DEFINE_integer('display_single_pane_ncols', 0, 'if positive, display all images in a single visdom web panel with certain number of images per row.')


class Visualizer:
    def __init__(self, opt):
        # self.opt = opt
        self.display_id = 1
        self.winsize = 256
        server = opt.LOGGING.WEB_VIS_SERVER
        self.win_size = 256
        self.name = opt.NAME
        if opt.ENV_NAME == "main":
            self.env_name = opt.NAME
        else:
            self.env_name = opt.ENV_NAME

        self.result_dir = osp.join(opt.RESULT_DIR, opt.DATALOADER.SPLIT)
        if self.display_id > 0:
            print(f"Visdom Env Name {self.env_name}")
            self.vis = visdom.Visdom(
                server=server,
                port=opt.LOGGING.WEB_VIS_PORT,
                env=self.env_name,
                use_incoming_socket=False,
            )
            self.display_single_pane_ncols = 0

        self.log_name = os.path.join(opt.CHECKPOINT_DIR, opt.NAME, "loss_log.txt")
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write(
                "================ Training Loss (%s) ================\n" % now
            )

    def display_current_results(self, visuals, epoch):
        if self.display_id > 0:  # show images in the browser
            idx = 1
            for label, image_numpy in visuals.items():
                # import pdb;pdb.set_trace()
                try:
                    self.vis.image(
                        image_numpy.transpose([2, 0, 1]),
                        opts=dict(title=label),
                        win=self.display_id + idx,
                    )
                    idx += 1
                except:
                    pass

    # scalars: dictionary of scalar labels and values
    def plot_current_scalars(self, epoch, counter_ratio, opt, scalars):
        if not hasattr(self, "plot_data"):
            self.plot_data = {"X": [], "Y": [], "legend": list(scalars.keys())}
        self.plot_data["X"].append(epoch + counter_ratio)
        self.plot_data["Y"].append([scalars[k] for k in self.plot_data["legend"]])
        self.vis.line(
            X=np.stack(
                [np.array(self.plot_data["X"])] * len(self.plot_data["legend"]), 1
            ),
            Y=np.array(self.plot_data["Y"]),
            opts={
                "title": self.name + " loss over time",
                "legend": self.plot_data["legend"],
                "xlabel": "epoch",
                "ylabel": "loss",
            },
            win=self.display_id,
        )

    # scatter plots
    def plot_current_points(self, points, disp_offset=10):
        idx = disp_offset
        for label, pts in points.items():
            # image_numpy = np.flipud(image_numpy)
            self.vis.scatter(
                pts, opts=dict(title=label, markersize=1), win=self.display_id + idx
            )
            idx += 1

    # scalars: same format as |scalars| of plot_current_scalars
    def print_current_scalars(self, epoch, i, scalars):
        # message = '(epoch: %d, iters: %d) ' % (epoch, i)
        message = "(epoch: %d, iters: %d) " % (epoch, i)
        for k, v in scalars.items():
            message += f"{k}: {v:.4f} "

        logger.info(message)
        """
        message = {}
        message['time'] = '%0.3f' % t
        message['epoch'] = '%d' % epoch
        message['iters'] = '%d' % i
        message = '(time : %0.3f, epoch: %d, iters: %d) ' % (t, epoch, i)
        for k, v in scalars.items():
        message += '%s: %.3f ' % (k, v)
        message.update(scalars)
        message = json.dumps(message)
        print(message)
        """
        with open(self.log_name, "a") as log_file:
            log_file.write("%s\n" % message)
