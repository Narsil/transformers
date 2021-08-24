# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import subprocess
from typing import TYPE_CHECKING, Union

import numpy as np

from ..utils import logging
from .base import Pipeline


if TYPE_CHECKING:
    from ...feature_extraction_sequence_utils import SequenceFeatureExtractor

logger = logging.get_logger(__name__)


def ffmpeg_read(bpayload: bytes, sampling_rate: int) -> np.array:
    """
    Helper function to read an audio file through ffmpeg.
    """
    ar = f"{sampling_rate}"
    ac = "1"
    format_for_conversion = "f32le"
    ffmpeg_command = [
        "ffmpeg",
        "-i",
        "pipe:0",
        "-ac",
        ac,
        "-ar",
        ar,
        "-f",
        format_for_conversion,
        "-hide_banner",
        "-loglevel",
        "quiet",
        "pipe:1",
    ]

    try:
        ffmpeg_process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    except FileNotFoundError:
        raise ValueError("ffmpeg was not found but is required to load audio files from filename")
    output_stream = ffmpeg_process.communicate(bpayload)
    out_bytes = output_stream[0]

    audio = np.frombuffer(out_bytes, np.float32)
    if audio.shape[0] == 0:
        raise ValueError("Malformed soundfile")
    return audio


class AutomaticSpeechRecognitionPipeline(Pipeline):
    """
    Pipeline that aims at extracting spoken text contained within some audio.

    The input can be either a raw waveform or a audio file. In case of the audio file, ffmpeg should be installed for
    to support multiple audio formats
    """

    def __init__(self, feature_extractor: Union["SequenceFeatureExtractor", str], *args, **kwargs):
        """
        Arguments:
            feature_extractor (:obj:`~transformers.SequenceFeatureExtractor`):
                The feature extractor that will be used by the pipeline to encode waveform for the model.
            model (:obj:`~transformers.PreTrainedModel` or :obj:`~transformers.TFPreTrainedModel`):
                The model that will be used by the pipeline to make predictions. This needs to be a model inheriting
                from :class:`~transformers.PreTrainedModel` for PyTorch and :class:`~transformers.TFPreTrainedModel`
                for TensorFlow.
            tokenizer (:obj:`~transformers.PreTrainedTokenizer`):
                The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
                :class:`~transformers.PreTrainedTokenizer`.
            modelcard (:obj:`str` or :class:`~transformers.ModelCard`, `optional`):
                Model card attributed to the model for this pipeline.
            framework (:obj:`str`, `optional`):
                The framework to use, either :obj:`"pt"` for PyTorch or :obj:`"tf"` for TensorFlow. The specified
                framework must be installed.

                If no framework is specified, will default to the one currently installed. If no framework is specified
                and both frameworks are installed, will default to the framework of the :obj:`model`, or to PyTorch if
                no model is provided.
            device (:obj:`int`, `optional`, defaults to -1):
                Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the
                model on the associated CUDA device id.
        """
        super().__init__(*args, **kwargs)

        self.feature_extractor = feature_extractor

        if self.framework == "tf":
            raise ValueError("The AutomaticSpeechRecognitionPipeline is only available in PyTorch.")

    def __call__(
        self,
        inputs: Union[np.ndarray, bytes, str],
        **kwargs,
    ):
        """
        Classify the sequence(s) given as inputs. See the :obj:`~transformers.AutomaticSpeechRecognitionPipeline`
        documentation for more information.

        Args:
            inputs (:obj:`np.ndarray` or :obj:`bytes` or :obj:`str`):
                The inputs is either a raw waveform (:obj:`np.ndarray` of shape (n, ) of type :obj:`np.float32` or
                :obj:`np.float64`) at the correct sampling rate (no further check will be done) or a :obj:`str` that is
                the filename of the audio file, the file will be read at the correct sampling rate to get the waveform
                using `ffmpeg`. This requires `ffmpeg` to be installed on the system. If `inputs` is :obj:`bytes` it is
                supposed to be the content of an audio file and is interpreted by `ffmpeg` in the same way.

        Return:
            A :obj:`dict` with the following keys:

            - **text** (:obj:`str`) -- The recognized text.
        """
        return super().__call__(inputs, **kwargs)

    def set_parameters(self, **kwargs):
        # No parameters on this pipeline right now
        pass

    def preprocess(self, inputs):
        if isinstance(inputs, str):
            with open(inputs, "rb") as f:
                inputs = f.read()

        if isinstance(inputs, bytes):
            inputs = ffmpeg_read(inputs, self.feature_extractor.sampling_rate)

        if not isinstance(inputs, np.ndarray):
            raise ValueError("We expect a numpy ndarray as input")
        if len(inputs.shape) != 1:
            raise ValueError("We expect a single channel audio input for AutomaticSpeechRecognitionPipeline")

        processed = self.feature_extractor(
            inputs, sampling_rate=self.feature_extractor.sampling_rate, return_tensors="pt"
        )
        return processed

    def forward(self, model_inputs):
        model_inputs = self.ensure_tensor_on_device(**model_inputs)
        name = self.model.__class__.__name__
        batch_size = self.feature_extractor.sampling_rate * 60  # 1mn batch
        if "input_features" in model_inputs:
            input_ids = model_inputs["input_features"]
        else:
            input_ids = model_inputs["input_values"]
        n = input_ids.shape[-1]
        all_tokens = []
        for i in range(0, n, batch_size):
            if name.endswith("ForConditionalGeneration") or name.endswith("EncoderDecoderModel"):
                encoder = self.model.get_encoder()
                # we need to pass `processed.get("attention_mask")` here since audio encoder
                # attention mask  length is different from expected text decoder `encoder_attention_mask` length
                # `generate` magic to create the mask automatically won't work, we basically need to help
                # it here.
                attention_mask = model_inputs["attention_mask"][:, i : i + batch_size]
                encoder_outputs = encoder(input_ids[:, i : i + batch_size], attention_mask=attention_mask)
                tokens = self.model.generate(encoder_outputs=encoder_outputs, attention_mask=attention_mask)
                tokens = tokens.squeeze(0)
            elif name.endswith("ForCTC"):
                outputs = self.model(**model_inputs)
                tokens = outputs.logits.squeeze(0).argmax(dim=-1)
            all_tokens.append(tokens)
        import torch

        return torch.cat(all_tokens, axis=0)

    def postprocess(self, model_outputs):
        skip_special_tokens = False if "CTC" in self.tokenizer.__class__.__name__ else True
        recognized_string = self.tokenizer.decode(model_outputs, skip_special_tokens=skip_special_tokens)
        return {"text": recognized_string}
