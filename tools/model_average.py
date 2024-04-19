# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2024.04.06
""" Model average over all saved checkpoints. """

import os
import glob
import glog
import torch


def model_average(chkpt_dir) -> None:
    # Average all saved chkpts ensuring stable performance.
    glog.info("Checkpoint average specified.")
    assert os.path.exists(chkpt_dir)
    if os.path.exists(os.path.join(chkpt_dir, "averaged.chkpt")):
        glog.info("Averaged checkpoint found. use saved one.")
        return

    aver_model = None
    for count, chkpt in enumerate(glob.glob("{}/*.ckpt".format(chkpt_dir))):
        model = torch.load(chkpt, map_location="cpu")
        if aver_model is None:
            aver_model = model
        else:
            for k in model["state_dict"]:
                aver_model["state_dict"][k] += model["state_dict"][k]
    count += 1  # Num of tracked chkpts.

    for k in aver_model["state_dict"]:
        aver_model["state_dict"][k] = torch.true_divide(
            aver_model["state_dict"][k], count)
    out_path = os.path.join(chkpt_dir, "averaged.chkpt")
    glog.info("Saving averaged model into {}...".format(out_path))
    torch.save(aver_model, out_path)
