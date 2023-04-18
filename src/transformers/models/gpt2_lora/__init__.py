# Copyright 2023 The HuggingFace Team. All rights reserved.
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

from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_keras_nlp_available,
    is_tensorflow_text_available,
    is_torch_available,
)


_import_structure = {
    "configuration_gpt2_lora": ["GPT2_LORA_PRETRAINED_CONFIG_ARCHIVE_MAP", "GPT2_LoRAConfig", "GPT2_LoRAOnnxConfig"],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_gpt2_lora"] = [
        "GPT2_LORA_PRETRAINED_MODEL_ARCHIVE_LIST",
        "GPT2_LoRADoubleHeadsModel",
        "GPT2_LoRAForSequenceClassification",
        "GPT2_LoRAForTokenClassification",
        "GPT2_LoRALMHeadModel",
        "GPT2_LoRAModel",
        "GPT2_LoRAPreTrainedModel",
        "load_tf_weights_in_gpt2_lora",
    ]

try:
    if not is_keras_nlp_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_gpt2_lora_tf"] = ["TFGPT2Tokenizer"]

if TYPE_CHECKING:
    from .configuration_gpt2_lora import GPT2_LORA_PRETRAINED_CONFIG_ARCHIVE_MAP, GPT2_LoRAConfig, GPT2_LoRAOnnxConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_gpt2_lora import (
            GPT2_LORA_PRETRAINED_MODEL_ARCHIVE_LIST,
            GPT2_LoRADoubleHeadsModel,
            GPT2_LoRAForSequenceClassification,
            GPT2_LoRAForTokenClassification,
            GPT2_LoRALMHeadModel,
            GPT2_LoRAModel,
            GPT2_LoRAPreTrainedModel,
            load_tf_weights_in_gpt2_lora,
        )

    try:
        if not is_keras_nlp_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        pass

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
