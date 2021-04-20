# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
""" Auto Processor class. """

from collections import OrderedDict

from ...configuration_utils import PretrainedConfig
from ..speech_to_text.processing_speech_to_text import Speech2TextProcessor
from ..wav2vec2.processing_wav2vec2 import Wav2Vec2Processor
from .configuration_auto import AutoConfig, Speech2TextConfig, Wav2Vec2Config, replace_list_option_in_docstrings


PROCESSOR_MAPPING = OrderedDict(
    [
        (Wav2Vec2Config, Wav2Vec2Processor),
        (Speech2TextConfig, Speech2TextProcessor),
    ]
)


def processor_class_from_name(class_name: str):
    for processor in PROCESSOR_MAPPING.values():
        if processor.__name__ == class_name:
            return processor


class AutoProcessor:
    r"""
    This is a generic process class that will be instantiated as one of the processor classes of the library when
    created with the :meth:`AutoProcessor.from_pretrained` class method.

    This class cannot be instantiated directly using ``__init__()`` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoProcessor is designed to be instantiated "
            "using the `AutoProcessor.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    @replace_list_option_in_docstrings(PROCESSOR_MAPPING)
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        r"""
        Instantiate one of the processor classes of the library from a pretrained directory.

        The processor class to instantiate is selected based on the :obj:`model_type` property of the config object
        (either passed as an argument or loaded from :obj:`pretrained_model_name_or_path` if possible), or when it's
        missing, by falling back to using pattern matching on :obj:`pretrained_model_name_or_path`:

        List options

        Params:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                Can be either:

                    - A string, the `model id` of a predefined processor hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like ``bert-base-uncased``, or namespaced under
                      a user or organization name, like ``dbmdz/bert-base-german-cased``.
                    - A path to a `directory` containing vocabulary files required by the processor, for instance saved
                      using the :func:`~transformers.PreTrainedProcessor.save_pretrained` method, e.g.,
                      ``./my_model_directory/``.
                    - A path or url to a single saved vocabulary file if and only if the processor only requires a
                      single vocabulary file (like Bert or XLNet), e.g.: ``./my_model_directory/vocab.txt``. (Not
                      applicable to all derived classes)
            inputs (additional positional arguments, `optional`):
                Will be passed along to the Processor ``__init__()`` method.
            config (:class:`~transformers.PreTrainedConfig`, `optional`)
                The configuration object used to dertermine the processor class to instantiate.
            cache_dir (:obj:`str` or :obj:`os.PathLike`, `optional`):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to force the (re-)download the model weights and configuration files and override the
                cached versions if they exist.
            resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (:obj:`Dict[str, str]`, `optional`):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., :obj:`{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            revision(:obj:`str`, `optional`, defaults to :obj:`"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
                identifier allowed by git.
            subfolder (:obj:`str`, `optional`):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co, specify it here.
            kwargs (additional keyword arguments, `optional`):
                Will be passed to the Processor ``__init__()`` method. See parameters in the ``__init__()`` for more details.

        Examples::

            >>> from transformers import AutoProcessor

            >>> # Download processor from huggingface.co and cache.
            >>> processor = AutoProcessor.from_pretrained('facebook/wav2vec2-base-960h')

            >>> # Download vocabulary from huggingface.co (user-uploaded) and cache.
            >>> processor = AutoProcessor.from_pretrained('theainerd/Wav2Vec2-large-xlsr-hindi')

            >>> # If vocabulary files are in a directory (e.g. processor was saved using `save_pretrained('./test/s2t_saved_model/')`)
            >>> processor = AutoProcessor.from_pretrained('./test/s2t_saved_model/')

        """
        config = kwargs.pop("config", None)
        kwargs["_from_auto"] = True
        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        if config.processor_class is not None:
            processor_class_candidate = config.processor_class
            processor_class = processor_class_from_name(processor_class_candidate)
            if processor_class is None:
                raise ValueError(
                    f"Processor class {processor_class_candidate} does not exist or is not currently imported."
                )
            return processor_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)

        if type(config) in PROCESSOR_MAPPING.keys():
            processor_class_py = PROCESSOR_MAPPING[type(config)]
            return processor_class_py.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)

        raise ValueError(
            f"Unrecognized configuration class {config.__class__} to build an AutoProcessor.\n"
            f"Model type should be one of {', '.join(c.__name__ for c in PROCESSOR_MAPPING.keys())}."
        )
