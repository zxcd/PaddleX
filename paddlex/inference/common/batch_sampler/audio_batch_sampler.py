# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import ast
from pathlib import Path
import numpy as np

from ....utils import logging
from ....utils.cache import CACHE_DIR
from ....utils.download import download
from .base_batch_sampler import BaseBatchSampler


class AudioBatchSampler(BaseBatchSampler):
    def __init__(self):
        super().__init__()
        self.batch_size = 1

    # XXX: auto download for url
    def _download_from_url(self, in_path):
        file_name = Path(in_path).name
        save_path = Path(CACHE_DIR) / "predict_input" / file_name
        download(in_path, save_path, overwrite=True)
        return save_path.as_posix()

    def sample(self, inputs):
        if isinstance(inputs, str):
            if inputs.startswith("http"):
                inputs = self._download_from_url(inputs)
            yield [inputs]
        else:
            logging.warning(
                f"Not supported input data type! Only `str` are supported, but got: {input}."
            )
