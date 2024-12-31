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

from ....utils.func_register import FuncRegister
from ...common.batch_sampler import AudioBatchSampler

from ..base import BasicPredictor
from .result import WhisperResult
from ...utils.io import AudioReader
from .processors import (
    ModelDimensions,
    log_mel_spectrogram,
    Whisper,
    LANGUAGES,
    TO_LANGUAGE_CODE,
)
from ....utils.download import download_and_extract
import lazy_paddle as paddle


class WhisperPredictor(BasicPredictor):

    entities = [
        "whisper_large",
        "whisper_base",
        "whisper_base",
        "whisper_tiny",
        "whisper_small",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.audio_reader = self._build()
        download_and_extract(
            self.config["resource_path"], self.config["resource_dir"], "assets"
        )

    def _build_batch_sampler(self):
        return AudioBatchSampler()

    def _get_result_class(self):
        return WhisperResult

    def _build(self):
        audio_reader = AudioReader(backend="wav")
        return audio_reader

    def process(self, batch_data):
        audio, sample_rate = self.audio_reader.read(batch_data[0])
        audio = paddle.to_tensor(audio)
        audio = audio[:, 0]
        audio = log_mel_spectrogram(audio, resource_path=self.config["resource_dir"])
        model_dict = paddle.load(self.config["model_file"])
        dims = ModelDimensions(**model_dict["dims"])
        self.model = Whisper(dims)
        self.model.load_dict(model_dict)
        self.model.eval()

        result = self.model.transcribe(
            audio,
            verbose=self.config["verbose"],
            task=self.config["task"],
            language=self.config["language"],
            resource_path=self.config["resource_dir"],
            temperature=self.config["temperature"],
            compression_ratio_threshold=self.config["compression_ratio_threshold"],
            logprob_threshold=self.config["logprob_threshold"],
            best_of=self.config["best_of"],
            beam_size=self.config["beam_size"],
            patience=self.config["patience"],
            length_penalty=self.config["length_penalty"],
            initial_prompt=self.config["initial_prompt"],
            condition_on_previous_text=self.config["condition_on_previous_text"],
            no_speech_threshold=self.config["no_speech_threshold"],
        )
        return {
            "input_path": batch_data,
            "result": [result],
        }
