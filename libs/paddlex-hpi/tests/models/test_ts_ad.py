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

import json

from paddlex.inference.results import TSAdResult
from tests.models.base import BaseTestPredictor

from paddlex_hpi.models import TSAdPredictor

MODEL_URL = "https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hpi/tests/models/ts_ad_model.zip"
INPUT_DATA_URL = "https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hpi/tests/models/ts_ad_input.csv"
EXPECTED_RESULT_URL = "https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hpi/tests/models/ts_ad_result.json"


class TestTSAdPredictor(BaseTestPredictor):
    @property
    def model_url(self):
        return MODEL_URL

    @property
    def input_data_url(self):
        return INPUT_DATA_URL

    @property
    def expected_result_url(self):
        return EXPECTED_RESULT_URL

    @property
    def predictor_cls(self):
        return TSAdPredictor

    def _check_result(self, result, expected_result):
        assert isinstance(result, TSAdResult)
        assert "input_ts" in result
        result.pop("input_ts")
        assert set(result) == set(expected_result)
        expected_result = json.loads(expected_result["anomaly"])
        result = result["anomaly"].to_dict(orient="records")
        assert result == expected_result
