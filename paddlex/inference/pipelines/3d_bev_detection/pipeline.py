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

from typing import Any, Dict, Optional, Union, List
import numpy as np
from importlib import import_module
from ...utils.pp_option import PaddlePredictorOption
from ..base import BasePipeline

module_3d_bev_detection_result = import_module(
    ".result", "paddlex.inference.models.3d_bev_detection"
)
BEV3DDetResult = getattr(module_3d_bev_detection_result, "BEV3DDetResult")


class BEVDet3DPipeline(BasePipeline):
    """3D Detection Pipeline"""

    entities = "3d_bev_detection"

    def __init__(
        self,
        config: Dict,
        device: str = None,
        pp_option: PaddlePredictorOption = None,
        use_hpip: bool = False,
    ) -> None:
        """
        Initializes the class with given configurations and options.

        Args:
            config (Dict): Configuration dictionary containing model and other parameters.
            device (str): The device to run the prediction on. Default is None.
            pp_option (PaddlePredictorOption): Options for PaddlePaddle predictor. Default is None.
            use_hpip (bool): Whether to use high-performance inference (hpip) for prediction. Defaults to False.
        """
        super().__init__(device=device, pp_option=pp_option, use_hpip=use_hpip)

        bev_detection_3d_model_config = config["SubModules"]["3DBEVDetection"]
        self.bev_detection_3d_model = self.create_model(bev_detection_3d_model_config)

    def predict(
        self,
        input: Union[str, List[str], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> BEV3DDetResult:
        """Predicts 3D detection results for the given input.

        Args:
            input (str | list[str] | np.ndarray | list[np.ndarray]): The input path(s) to the 3d annotation pickle file.
            **kwargs: Additional keyword arguments that can be passed to the function.

        Returns:
            BEV3DDetResult: The predicted 3d detection results.
        """
        yield from self.bev_detection_3d_model(input)
