# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2024.04.06
""" Model average over all saved checkpoints. """

import os
import glob
import glog
import torch


def _find_latest_chkpts(dir):
    """ Find the latest checkpoint within given directory """
    chkpts = [f for f in glob.glob("{}/*.ckpt".format(dir))]
    chkpt = max(chkpts, key=lambda f: os.path.getctime(os.path.join(dir, f)))
    return os.path.join(dir, chkpt)


def _topk_chkpt_pool(latest_chkpt, num_aver, descending=False):
    chkpt_pool = []
    for k in latest_chkpt["callbacks"]:
        for pt in latest_chkpt["callbacks"][k]["best_k_models"]:
            chkpt_pool.append(
                tuple([pt, latest_chkpt["callbacks"][k]["best_k_models"][pt]]))

    # NOTE: If checkpointing by monitoring wer, descending is False, which means
    # best chkpt corresponded with lower wer. If checkpointing by monitoring acc, like
    # in ssl task or other classify task, descending should be True ,vice versa.
    chkpt_pool.sort(key=lambda x: x[1], reverse=descending)
    if num_aver is not None:
        return chkpt_pool[:num_aver]  # Return topK chkpts for average
    else:
        return chkpt_pool  # Return all stored chkpts for average


def model_average(chkpt_dir, aver_best_k=None, descending=False) -> None:
    # Average all saved chkpts ensuring stable performance.
    glog.info("Checkpoint average specified, with aver_best_k of {}.".format(
        aver_best_k))
    assert os.path.exists(chkpt_dir)
    if os.path.exists(os.path.join(chkpt_dir, "averaged.chkpt")):
        glog.info("Averaged checkpoint found. use saved one.")
        return

    latest_chkpt = torch.load(_find_latest_chkpts(chkpt_dir),
                              map_location="cpu")
    chkpts_to_aver = _topk_chkpt_pool(latest_chkpt=latest_chkpt,
                                      num_aver=aver_best_k,
                                      descending=descending)

    aver_model = None
    for count, chkpt in enumerate(chkpts_to_aver):
        model = torch.load(chkpt[0], map_location="cpu")
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
