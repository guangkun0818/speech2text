# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2024.08.09
""" Designed sampler for torch dataset. """

import glog
import math
import torch
import random

from torch.utils.data.sampler import BatchSampler
from torch.utils.data.distributed import DistributedSampler

from dataset.dataset import BaseDataset


class DynamicBucketBatchSampler(BatchSampler):
    """ Dynamic batching sampler with bucketing strategy """

    def __init__(self,
                 sampler: DistributedSampler,
                 dataset: BaseDataset,
                 num_bucket: int = 30,
                 key: str = "duration",
                 min_batch_size: int = 8,
                 volume_threshold: int = 800) -> None:
        super(DynamicBucketBatchSampler, self).__init__(sampler,
                                                        min_batch_size,
                                                        drop_last=False)

        self._key = key
        self._dataset = dataset
        self._volume_threshold = volume_threshold
        self._create_bucket(num_bucket=num_bucket,
                            low_bound=self._dataset.lower_bound,
                            high_bound=self._dataset.high_bound)
        assert hasattr(dataset, "fetch_data_k_info")

    def _create_bucket(self, num_bucket, low_bound, high_bound):
        self._buckets = {
            idx: {
                "bucket_id": idx,
                "data": [],
                "bounds": (0.0, 0.0),
                "volume": 0.0
            } for idx in range(num_bucket)
        }
        interval_step = (float(high_bound) -
                         float(low_bound)) / float(num_bucket)

        for idx in self._buckets:
            self._buckets[idx]["bounds"] = (
                self._buckets[idx]["bucket_id"] * interval_step + low_bound,
                (self._buckets[idx]["bucket_id"] + 1) * interval_step +
                low_bound)

        glog.info(
            "Buckets created on rank {} with low_bound = {} and high_bound = {}, once created it will be maintained through training."
            .format(self.sampler.rank, low_bound, high_bound))

    def _select_bucket(self, idx, v):
        for idx in self._buckets:
            if self._buckets[idx]["bounds"][0] <= v <= self._buckets[idx][
                    "bounds"][1]:
                return idx

    def _push_in_bucket(self, bucket_id, sample_id, v):
        self._buckets[bucket_id]["data"].append(sample_id)
        self._buckets[bucket_id]["volume"] += v

    def __iter__(self):
        while True:
            for sample_id in self.sampler:
                v = self._dataset.fetch_data_k_info(sample_id, k=self._key)
                bucket_id = self._select_bucket(sample_id, v)
                self._push_in_bucket(bucket_id, sample_id, v)
                if self._buckets[bucket_id][
                        "volume"] > self._volume_threshold and len(
                            self._buckets[bucket_id]["data"]) > self.batch_size:
                    yield self._buckets[bucket_id]["data"]
                    self._buckets[bucket_id]["data"] = []
                    self._buckets[bucket_id]["volume"] = 0.0

            glog.info(
                "Data slice of rank {} fully comsumed, continue training within this epoch"
                .format(self.sampler.rank))

    def __len__(self) -> int:
        """ The __len__ determine the number of iteration within one epoch. But in this 
            sampler design, iteration will cover more data than empirically speaking single 
            epoch due to dynamic batching strategy.
        """

        return math.ceil(
            math.ceil(self._dataset.total_data_amount /
                      self.sampler.num_replicas) / self._volume_threshold)
