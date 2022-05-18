# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
#

import datetime
import math
import unittest

from transformers import BloomConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...generation.test_generation_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask


if is_torch_available():
    import torch

    from transformers import BLOOM_PRETRAINED_MODEL_ARCHIVE_LIST, BloomForCausalLM, BloomModel, BloomTokenizerFast


@require_torch
class BloomModelTester:
    def __init__(
        self,
        parent,
        batch_size=14,
        seq_length=7,
        is_training=True,
        use_token_type_ids=True,
        use_input_mask=True,
        use_labels=True,
        use_mc_token_ids=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_token_type_ids = use_token_type_ids
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.use_mc_token_ids = use_mc_token_ids
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = None
        self.bos_token_id = vocab_size - 1
        self.eos_token_id = vocab_size - 1
        self.pad_token_id = vocab_size - 1

    def get_large_model_config(self):
        return BloomConfig.from_pretrained("bloom")

    def prepare_config_and_inputs(self, gradient_checkpointing=False):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        config = self.get_config(gradient_checkpointing=gradient_checkpointing)

        return (config, input_ids, input_mask)

    def get_config(self, gradient_checkpointing=False):
        return BloomConfig(
            vocab_size=self.vocab_size,
            seq_length=self.seq_length,
            hidden_size=self.hidden_size,
            n_layer=self.num_hidden_layers,
            n_head=self.num_attention_heads,
            resid_pdrop=self.hidden_dropout_prob,
            attn_pdrop=self.attention_probs_dropout_prob,
            n_positions=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
            use_cache=True,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            gradient_checkpointing=gradient_checkpointing,
            dtype="float32",
        )

    def create_and_check_bloom_model(self, config, input_ids, input_mask, *args):
        model = BloomModel(config=config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(len(result.past_key_values), config.n_layer)

    def create_and_check_bloom_model_past(self, config, input_ids, input_mask, *args):
        model = BloomModel(config=config)

        # model = BloomForCausalLM(config=config)
        model.to(torch_device)
        model.eval()

        # first forward pass
        outputs = model(input_ids, attention_mask=torch.ones_like(input_ids), use_cache=True)
        outputs_use_cache_conf = model(input_ids, attention_mask=torch.ones_like(input_ids))
        outputs_no_past = model(input_ids, use_cache=False, attention_mask=torch.ones_like(input_ids))

        self.parent.assertTrue(len(outputs) == len(outputs_use_cache_conf))
        self.parent.assertTrue(len(outputs) == len(outputs_no_past) + 1)

        # output, past = outputs.to_tuple()
        past = outputs["past_key_values"]

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size)

        # append to next input_ids and token_type_ids
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)

        # output_from_no_past = model(next_input_ids, output_hidden_states=True)["hidden_states"][-1]
        # output_from_past = model(next_tokens, past_key_values=past, output_hidden_states=True)["hidden_states"][-1]

        output_from_no_past = model(next_input_ids)["last_hidden_state"]
        output_from_past = model(next_tokens, past_key_values=past)["last_hidden_state"]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -1, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, 0, random_slice_idx].detach()

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def create_and_check_bloom_model_attention_mask_past(self, config, input_ids, input_mask, *args):
        model = BloomModel(config=config)
        model.to(torch_device)
        model.eval()

        # create attention mask
        attn_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)
        half_seq_length = self.seq_length // 2
        attn_mask[:, half_seq_length:] = 0

        # first forward pass
        output, past = model(input_ids, attention_mask=attn_mask).to_tuple()

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size)

        # change a random masked slice from input_ids
        random_seq_idx_to_change = ids_tensor((1,), half_seq_length).item() + 1
        random_other_next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size).squeeze(-1)
        input_ids[:, -random_seq_idx_to_change] = random_other_next_tokens

        # append to next input_ids and attn_mask
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        attn_mask = torch.cat(
            [attn_mask, torch.ones((attn_mask.shape[0], 1), dtype=torch.long, device=torch_device)],
            dim=1,
        )

        # get two different outputs
        # output_from_no_past = model(next_input_ids, attention_mask=attn_mask, output_hidden_states=True)["hidden_states"][-1]
        # output_from_past = model(next_tokens, past_key_values=past, attention_mask=attn_mask, output_hidden_states=True)["hidden_states"][-1]

        output_from_no_past = model(next_input_ids, attention_mask=attn_mask)["last_hidden_state"]
        output_from_past = model(next_tokens, past_key_values=past, attention_mask=attn_mask)["last_hidden_state"]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -1, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, 0, random_slice_idx].detach()

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=4e-2))

    def create_and_check_bloom_model_past_large_inputs(self, config, input_ids, input_mask, *args):
        model = BloomModel(config=config)
        model.to(torch_device)
        model.eval()

        # first forward pass
        outputs = model(input_ids, attention_mask=input_mask, use_cache=True)

        output, past = outputs.to_tuple()

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_mask = ids_tensor((self.batch_size, 3), vocab_size=2)

        # append to next input_ids and token_type_ids
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        next_attention_mask = torch.cat([input_mask, next_mask], dim=-1)

        output_from_no_past = model(next_input_ids, attention_mask=next_attention_mask)["last_hidden_state"]
        output_from_past = model(next_tokens, attention_mask=next_attention_mask)["last_hidden_state"]
        self.parent.assertTrue(output_from_past.shape[1] == next_tokens.shape[1])

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-1))

    def create_and_check_lm_head_model(self, config, input_ids, input_mask, *args):
        model = BloomForCausalLM(config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids, labels=input_ids)
        self.parent.assertEqual(result.loss.shape, ())
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_forward_and_backwards(
        self, config, input_ids, input_mask, *args, gradient_checkpointing=False
    ):
        model = BloomForCausalLM(config)
        model.to(torch_device)
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()

        result = model(input_ids, labels=input_ids)
        self.parent.assertEqual(result.loss.shape, ())
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))
        result.loss.backward()

    def create_and_check_bloom_weight_initialization(self, config, *args):
        model = BloomModel(config)
        model_std = model.config.initializer_range / math.sqrt(2 * model.config.n_layer)
        for key in model.state_dict().keys():
            if "c_proj" in key and "weight" in key:
                self.parent.assertLessEqual(abs(torch.std(model.state_dict()[key]) - model_std), 0.001)
                self.parent.assertLessEqual(abs(torch.mean(model.state_dict()[key]) - 0.0), 0.01)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()

        config, input_ids, input_mask = config_and_inputs

        inputs_dict = {"input_ids": input_ids}

        return config, inputs_dict


@require_torch
class BloomModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):

    all_model_classes = (BloomModel, BloomForCausalLM) if is_torch_available() else ()
    all_generative_model_classes = (BloomForCausalLM,) if is_torch_available() else ()
    all_parallelizable_model_classes = (BloomForCausalLM,) if is_torch_available() else ()
    fx_compatible = False
    test_missing_keys = False
    test_model_parallel = False
    test_pruning = False

    def setUp(self):
        self.model_tester = BloomModelTester(self)
        self.config_tester = ConfigTester(self, config_class=BloomConfig, n_embd=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_bloom_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_bloom_model(*config_and_inputs)

    def test_bloom_model_past(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_bloom_model_past(*config_and_inputs)

    def test_bloom_model_att_mask_past(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_bloom_model_attention_mask_past(*config_and_inputs)

    def test_bloom_model_past_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_bloom_model_past_large_inputs(*config_and_inputs)

    def test_bloom_lm_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lm_head_model(*config_and_inputs)

    def test_bloom_gradient_checkpointing(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_forward_and_backwards(*config_and_inputs, gradient_checkpointing=True)

    def test_bloom_weight_initialization(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_bloom_weight_initialization(*config_and_inputs)

    @slow
    def test_batch_generation(self):
        model = BloomForCausalLM.from_pretrained("bloom")
        model.to(torch_device)
        tokenizer = BloomTokenizerFast.from_pretrained("bloom")

        tokenizer.padding_side = "left"

        # Define PAD Token = EOS Token = 50256
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

        # use different length sentences to test batching
        sentences = [
            "Hello, my dog is a little",
            "Today, I",
        ]

        inputs = tokenizer(sentences, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(torch_device)
        token_type_ids = torch.cat(
            [
                input_ids.new_full((input_ids.shape[0], input_ids.shape[1] - 1), 0),
                input_ids.new_full((input_ids.shape[0], 1), 500),
            ],
            dim=-1,
        )

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"].to(torch_device),
        )

        outputs_tt = model.generate(
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"].to(torch_device),
            token_type_ids=token_type_ids,
        )

        inputs_non_padded = tokenizer(sentences[0], return_tensors="pt").input_ids.to(torch_device)
        output_non_padded = model.generate(input_ids=inputs_non_padded)

        num_paddings = inputs_non_padded.shape[-1] - inputs["attention_mask"][-1].long().sum().cpu().item()
        inputs_padded = tokenizer(sentences[1], return_tensors="pt").input_ids.to(torch_device)
        output_padded = model.generate(input_ids=inputs_padded, max_length=model.config.max_length - num_paddings)

        batch_out_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        batch_out_sentence_tt = tokenizer.batch_decode(outputs_tt, skip_special_tokens=True)
        non_padded_sentence = tokenizer.decode(output_non_padded[0], skip_special_tokens=True)
        padded_sentence = tokenizer.decode(output_padded[0], skip_special_tokens=True)

        expected_output_sentence = [
            "Hello, my dog is a little bit of a mess. I'm not sure if he's going",
            "Today, I'm going to be doing a lot of research on this. I",
        ]
        self.assertListEqual(expected_output_sentence, batch_out_sentence)
        self.assertTrue(batch_out_sentence_tt != batch_out_sentence)  # token_type_ids should change output
        self.assertListEqual(expected_output_sentence, [non_padded_sentence, padded_sentence])

    @slow
    def test_model_from_pretrained(self):
        for model_name in BLOOM_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = BloomModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


@require_torch
class BloomModelLanguageGenerationTest(unittest.TestCase):
    def _test_lm_generate_bloom_helper(
        self,
        gradient_checkpointing=False,
        verify_outputs=True,
    ):
        model = BloomForCausalLM.from_pretrained(
            "bloom",
        )
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_disable()
        model.to(torch_device)

        # The dog
        input_ids = torch.tensor([[464, 3290]], dtype=torch.long, device=torch_device)

        # The dog was found in a field near the intersection of West and West Streets.\n\nThe dog
        # fmt: off
        expected_output_ids = [
            464, 3290, 373, 1043, 287, 257, 2214, 1474, 262, 16246, 286, 2688, 290, 2688, 27262, 13, 198, 198, 464, 3290,
        ]
        # fmt: on
        output_ids = model.generate(input_ids, do_sample=False)
        if verify_outputs:
            self.assertListEqual(output_ids[0].tolist(), expected_output_ids)

    @slow
    def test_lm_generate_bloom(self):
        self._test_lm_generate_bloom_helper()

    @slow
    def test_lm_generate_bloom_with_gradient_checkpointing(self):
        self._test_lm_generate_bloom_helper(gradient_checkpointing=True)

    @slow
    def test_bloom_sample(self):
        tokenizer = BloomTokenizerFast.from_pretrained("bloom")
        model = BloomForCausalLM.from_pretrained("bloom")
        model.to(torch_device)

        torch.manual_seed(0)
        tokenized = tokenizer("Today is a nice day and", return_tensors="pt", return_token_type_ids=True)
        input_ids = tokenized.input_ids.to(torch_device)
        output_ids = model.generate(input_ids, do_sample=True)
        output_str = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        token_type_ids = tokenized.token_type_ids.to(torch_device)
        output_seq = model.generate(input_ids=input_ids, do_sample=True, num_return_sequences=5)
        output_seq_tt = model.generate(
            input_ids=input_ids, token_type_ids=token_type_ids, do_sample=True, num_return_sequences=5
        )
        output_seq_strs = tokenizer.batch_decode(output_seq, skip_special_tokens=True)
        output_seq_tt_strs = tokenizer.batch_decode(output_seq_tt, skip_special_tokens=True)

        EXPECTED_OUTPUT_STR = (
            "Today is a nice day and if you don't know anything about the state of play during your holiday"
        )
        self.assertEqual(output_str, EXPECTED_OUTPUT_STR)
        self.assertTrue(
            all([output_seq_strs[idx] != output_seq_tt_strs[idx] for idx in range(len(output_seq_tt_strs))])
        )  # token_type_ids should change output

    @slow
    def test_bloom_sample_max_time(self):
        tokenizer = BloomTokenizerFast.from_pretrained("bloom")
        model = BloomForCausalLM.from_pretrained("bloom")
        model.to(torch_device)

        torch.manual_seed(0)
        tokenized = tokenizer("Today is a nice day and", return_tensors="pt", return_token_type_ids=True)
        input_ids = tokenized.input_ids.to(torch_device)

        MAX_TIME = 0.5

        start = datetime.datetime.now()
        model.generate(input_ids, do_sample=True, max_time=MAX_TIME, max_length=256)
        duration = datetime.datetime.now() - start
        self.assertGreater(duration, datetime.timedelta(seconds=MAX_TIME))
        self.assertLess(duration, datetime.timedelta(seconds=1.5 * MAX_TIME))

        start = datetime.datetime.now()
        model.generate(input_ids, do_sample=False, max_time=MAX_TIME, max_length=256)
        duration = datetime.datetime.now() - start
        self.assertGreater(duration, datetime.timedelta(seconds=MAX_TIME))
        self.assertLess(duration, datetime.timedelta(seconds=1.5 * MAX_TIME))

        start = datetime.datetime.now()
        model.generate(input_ids, do_sample=False, num_beams=2, max_time=MAX_TIME, max_length=256)
        duration = datetime.datetime.now() - start
        self.assertGreater(duration, datetime.timedelta(seconds=MAX_TIME))
        self.assertLess(duration, datetime.timedelta(seconds=1.5 * MAX_TIME))

        start = datetime.datetime.now()
        model.generate(input_ids, do_sample=True, num_beams=2, max_time=MAX_TIME, max_length=256)
        duration = datetime.datetime.now() - start
        self.assertGreater(duration, datetime.timedelta(seconds=MAX_TIME))
        self.assertLess(duration, datetime.timedelta(seconds=1.5 * MAX_TIME))

        start = datetime.datetime.now()
        model.generate(input_ids, do_sample=False, max_time=None, max_length=256)
        duration = datetime.datetime.now() - start
        self.assertGreater(duration, datetime.timedelta(seconds=1.5 * MAX_TIME))


@require_torch
class BloomEmbeddingTest(unittest.TestCase):
    """
    The goal here is to compare the embeddings generated by the model trained
    using Megatron-LM with the one from the transformers library, with a small GPT2-like model
    to ensure that the conversion from Megatron-LM to transformers has been done successfully.
    The script compares the logits of the embedding layer and the transformer layers.

    WARNING: It is expected that these logits will not have exactly the same statistics when running
    the code on CPU or GPU. For more info, please visit:
      - https://github.com/pytorch/pytorch/issues/76052#issuecomment-1103193548
      - https://discuss.pytorch.org/t/reproducibility-issue-between-intel-and-amd-cpus/144779/9


    You need to install tokenizers following this readme:
        - https://huggingface.co/bigscience-catalogue-data-dev/byte-level-bpe-tokenizer-no-norm-250k-whitespace-and-eos-regex-alpha-v3-dedup-lines-articles

    Tokenizer used during training:
        - https://huggingface.co/bigscience-catalogue-data-dev/byte-level-bpe-tokenizer-no-norm-250k-whitespace-and-eos-regex-alpha-v3-dedup-lines-articles

    # TODO change the script (or just add skip) when building the env with tokenizers 0.12.0
    """

    def setUp(self):
        super().setUp()
        self.path_bigscience_model = "bigscience/bigscience-small-testing"

    @require_torch
    def test_load_transformers_model(self):
        # TODO load the model (.bin file)
        try:
            _ = BloomForCausalLM.from_pretrained(self.path_bigscience_model)
        except BaseException:
            self.fail("Failed loading the model")

    @require_torch
    def test_embeddings(self):
        model = BloomForCausalLM.from_pretrained(self.path_bigscience_model)
        model.eval()

        EMBEDDINGS_DS_BEFORE_LN_BF_16_MEAN = {
            3478: 0.0002307891845703125,
            368: -0.000568389892578125,
            109586: -0.0003910064697265625,
            35433: -0.000194549560546875,
            2: 0.0004138946533203125,
            77: 0.000659942626953125,
            132619: -0.00031280517578125,
            2175: 0.000457763671875,
            23714: 0.000263214111328125,
            73173: -0.000286102294921875,
            144252: 0.00052642822265625,
        }
        EMBEDDINGS_DS_BEFORE_LN_BF_16_MIN = {
            3478: -0.00921630859375,
            368: -0.010009765625,
            109586: -0.01031494140625,
            35433: -0.01177978515625,
            2: -0.0074462890625,
            77: -0.00848388671875,
            132619: -0.009521484375,
            2175: -0.0074462890625,
            23714: -0.0145263671875,
            73173: -0.007415771484375,
            144252: -0.01007080078125,
        }
        EMBEDDINGS_DS_BEFORE_LN_BF_16_MAX = {
            3478: 0.0128173828125,
            368: 0.01214599609375,
            109586: 0.0111083984375,
            35433: 0.01019287109375,
            2: 0.0157470703125,
            77: 0.0174560546875,
            132619: 0.0078125,
            2175: 0.0113525390625,
            23714: 0.0146484375,
            73173: 0.01116943359375,
            144252: 0.01141357421875,
        }
        EMBEDDINGS_DS_BEFORE_LN_BF_16_SUM = {"value": 0.08203125}

        EMBEDDINGS_DS_BEFORE_LN_F_16_MEAN = {
            132619: -0.00031256675720214844,
            3478: 0.00023090839385986328,
            368: -0.0005702972412109375,
            109586: -0.00039124488830566406,
            35433: -0.000194549560546875,
            2: 0.0004146099090576172,
            2175: 0.0004572868347167969,
            23714: 0.00026416778564453125,
            73173: -0.0002865791320800781,
            144252: 0.0005254745483398438,
            77: 0.0006618499755859375,
        }
        EMBEDDINGS_DS_BEFORE_LN_F_16_MIN = {
            3478: -0.00921630859375,
            368: -0.010009765625,
            109586: -0.01031494140625,
            35433: -0.01177978515625,
            2: -0.0074462890625,
            77: -0.00848388671875,
            132619: -0.009521484375,
            2175: -0.0074462890625,
            23714: -0.0145263671875,
            73173: -0.007415771484375,
            144252: -0.01007080078125,
        }
        EMBEDDINGS_DS_BEFORE_LN_F_16_MAX = {
            3478: 0.0128173828125,
            368: 0.01214599609375,
            109586: 0.0111083984375,
            35433: 0.01019287109375,
            2: 0.0157470703125,
            77: 0.0174560546875,
            132619: 0.0078125,
            2175: 0.0113525390625,
            23714: 0.0146484375,
            73173: 0.01116943359375,
            144252: 0.01141357421875,
        }
        EMBEDDINGS_DS_BEFORE_LN_F_16_SUM = {"value": 0.0821533203125}

        EMBEDDINGS_DS_BEFORE_LN_F_32_MEAN = {
            132619: -0.00031267106533050537,
            3478: 0.00023087859153747559,
            368: -0.0005701072514057159,
            109586: -0.0003911703824996948,
            35433: -0.0001944899559020996,
            2: 0.0004146844148635864,
            2175: 0.00045740045607089996,
            23714: 0.0002641640603542328,
            73173: -0.0002864748239517212,
            144252: 0.0005256589502096176,
            77: 0.0006617321632802486,
        }
        EMBEDDINGS_DS_BEFORE_LN_F_32_MIN = {
            3478: -0.00921630859375,
            368: -0.010009765625,
            109586: -0.01031494140625,
            35433: -0.01177978515625,
            2: -0.0074462890625,
            77: -0.00848388671875,
            132619: -0.009521484375,
            2175: -0.0074462890625,
            23714: -0.0145263671875,
            73173: -0.007415771484375,
            144252: -0.01007080078125,
        }
        EMBEDDINGS_DS_BEFORE_LN_F_32_MAX = {
            3478: 0.0128173828125,
            368: 0.01214599609375,
            109586: 0.0111083984375,
            35433: 0.01019287109375,
            2: 0.0157470703125,
            77: 0.0174560546875,
            132619: 0.0078125,
            2175: 0.0113525390625,
            23714: 0.0146484375,
            73173: 0.01116943359375,
            144252: 0.01141357421875,
        }
        EMBEDDINGS_DS_BEFORE_LN_F_32_SUM = {"value": 0.08217757940292358}

        TEST_EMBEDDINGS = {
            "torch.bfloat16": {
                "mean": EMBEDDINGS_DS_BEFORE_LN_BF_16_MEAN,
                "max": EMBEDDINGS_DS_BEFORE_LN_BF_16_MAX,
                "min": EMBEDDINGS_DS_BEFORE_LN_BF_16_MIN,
                "sum": EMBEDDINGS_DS_BEFORE_LN_BF_16_SUM,
            },
            "torch.float32": {
                "mean": EMBEDDINGS_DS_BEFORE_LN_F_32_MEAN,
                "max": EMBEDDINGS_DS_BEFORE_LN_F_32_MAX,
                "min": EMBEDDINGS_DS_BEFORE_LN_F_32_MIN,
                "sum": EMBEDDINGS_DS_BEFORE_LN_F_32_SUM,
            },
            "torch.float": {
                "mean": EMBEDDINGS_DS_BEFORE_LN_F_32_MEAN,
                "max": EMBEDDINGS_DS_BEFORE_LN_F_32_MAX,
                "min": EMBEDDINGS_DS_BEFORE_LN_F_32_MIN,
                "sum": EMBEDDINGS_DS_BEFORE_LN_F_32_SUM,
            },
            "torch.float16": {
                "mean": EMBEDDINGS_DS_BEFORE_LN_F_16_MEAN,
                "max": EMBEDDINGS_DS_BEFORE_LN_F_16_MAX,
                "min": EMBEDDINGS_DS_BEFORE_LN_F_16_MIN,
                "sum": EMBEDDINGS_DS_BEFORE_LN_F_16_SUM,
            },
        }

        EXAMPLE_IDS = [
            3478,
            368,
            109586,
            35433,
            2,
            77,
            132619,
            3478,
            368,
            109586,
            35433,
            2,
            2175,
            23714,
            73173,
            144252,
            2,
            77,
            132619,
            3478,
        ]

        EMBEDDINGS_DS_AFTER_LN_MEAN = {
            3478: -6.580352783203125e-05,
            368: 0.0001316070556640625,
            109586: -0.00030517578125,
            35433: 4.00543212890625e-05,
            2: -7.2479248046875e-05,
            77: -8.96453857421875e-05,
            132619: 0.0001583099365234375,
            2175: 2.1219253540039062e-05,
            23714: -0.000247955322265625,
            73173: -0.00021839141845703125,
            144252: -0.0001430511474609375,
        }
        EMBEDDINGS_DS_AFTER_LN_MIN = {
            3478: -1.6953125,
            368: -1.6875,
            109586: -1.6875,
            35433: -2.125,
            2: -1.390625,
            77: -1.5390625,
            132619: -1.875,
            2175: -1.4609375,
            23714: -2.296875,
            73173: -1.3515625,
            144252: -1.78125,
        }
        EMBEDDINGS_DS_AFTER_LN_MAX = {
            3478: 2.265625,
            368: 2.28125,
            109586: 1.953125,
            35433: 1.90625,
            2: 2.703125,
            77: 2.828125,
            132619: 1.65625,
            2175: 2.015625,
            23714: 2.234375,
            73173: 2.171875,
            144252: 1.828125,
        }

        EMBEDDINGS_DS_AFTER_LN = {
            "mean": EMBEDDINGS_DS_AFTER_LN_MEAN,
            "min": EMBEDDINGS_DS_AFTER_LN_MIN,
            "max": EMBEDDINGS_DS_AFTER_LN_MAX,
        }

        tensor_ids = torch.LongTensor([EXAMPLE_IDS])
        # position_ids = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]])
        with torch.no_grad():
            embeddings = model.transformer.word_embeddings(tensor_ids)
            embeddings_ln = model.transformer.word_embeddings_layernorm(embeddings)  #
        # first check the embeddings before LN
        output_dict = {"min": {}, "max": {}, "mean": {}, "sum": {"value": embeddings.sum().item()}}
        for i, idx in enumerate(EXAMPLE_IDS):
            output_dict["min"][idx] = embeddings.min(dim=-1).values[0][i].item()
            output_dict["max"][idx] = embeddings.max(dim=-1).values[0][i].item()
            output_dict["mean"][idx] = embeddings.mean(dim=-1)[0][i].item()

        for key in TEST_EMBEDDINGS[str(model.dtype)].keys():
            self.assertDictEqual(TEST_EMBEDDINGS[str(model.dtype)][key], output_dict[key])

        output_dict_norm = {"min": {}, "max": {}, "mean": {}}
        for i, idx in enumerate(EXAMPLE_IDS):
            output_dict_norm["min"][idx] = embeddings_ln.min(dim=-1).values[0][i].item()
            output_dict_norm["max"][idx] = embeddings_ln.max(dim=-1).values[0][i].item()
            output_dict_norm["mean"][idx] = embeddings_ln.mean(dim=-1)[0][i].item()
        # word_embeddings

        # This test does not pass when places = 2
        for i, key in enumerate(output_dict_norm.keys()):
            for j, idx in enumerate(output_dict[key].keys()):
                self.assertAlmostEqual(EMBEDDINGS_DS_AFTER_LN[key][idx], output_dict_norm[key][idx], places=1)
        # self.assertDictEqual(EMBEDDINGS_DS_AFTER_LN, output_dict_norm)

    @require_torch
    def test_hidden_states_transformers(self):
        # TODO ifelse device
        cuda_available = torch.cuda.is_available()
        device = torch.device("cuda:0" if cuda_available else "cpu")
        model = BloomModel.from_pretrained(self.path_bigscience_model, use_cache=False).to(device)
        model.eval()

        EXAMPLE_IDS = [
            3478,
            368,
            109586,
            35433,
            2,
            77,
            132619,
            3478,
            368,
            109586,
            35433,
            2,
            2175,
            23714,
            73173,
            144252,
            2,
            77,
            132619,
            3478,
        ]

        # a = torch.randn(1, 1, 20, 20)
        # ATTN_MASK = (torch.triu(a, diagonal=1) != 0).to(device)

        MEAN_VALUE_LAST_LM = -4.3392181396484375e-05
        MIN_MAX_DICT = {"min": -2.0625, "max": 2.75}
        tensor_ids = torch.LongTensor([EXAMPLE_IDS])

        with torch.no_grad():
            logits = model(tensor_ids.to(device))
        output_dict = {
            "min": logits.last_hidden_state.min(dim=-1).values[0][0].item(),
            "max": logits.last_hidden_state.max(dim=-1).values[0][0].item(),
        }

        if cuda_available:
            self.assertEqual(MEAN_VALUE_LAST_LM, logits.last_hidden_state.mean().item())
        else:
            self.assertAlmostEqual(MEAN_VALUE_LAST_LM, logits.last_hidden_state.mean().item(), places=3)

        self.assertDictEqual(MIN_MAX_DICT, output_dict)

    @require_torch
    def test_logits(self):
        cuda_available = torch.cuda.is_available()
        # cuda_available = False
        device = torch.device("cuda:3" if cuda_available else "cpu")
        model = BloomForCausalLM.from_pretrained(self.path_bigscience_model, use_cache=False).to(device)
        model.eval()

        EXAMPLE_IDS = [
            3478,
            368,
            109586,
            35433,
            2,
            77,
            132619,
            3478,
            368,
            109586,
            35433,
            2,
            2175,
            23714,
            73173,
            144252,
            2,
            77,
            132619,
            3478,
        ]

        MEAN_LOGITS_GPU_1 = -1.823902130126953e-05
        MEAN_LOGITS_GPU_2 = 1.9431114196777344e-05

        tensor_ids = torch.LongTensor([EXAMPLE_IDS]).to(device)
        with torch.no_grad():
            output = model(tensor_ids).logits

        output_gpu_1, output_gpu_2 = output.split(125440, dim=-1)
        if cuda_available:
            self.assertEqual(output_gpu_1.mean().item(), MEAN_LOGITS_GPU_1)
            self.assertEqual(output_gpu_2.mean().item(), MEAN_LOGITS_GPU_2)
        else:
            self.assertAlmostEqual(output_gpu_1.mean().item(), MEAN_LOGITS_GPU_1, places=6)  # 1e-06 precision!!
            self.assertAlmostEqual(output_gpu_2.mean().item(), MEAN_LOGITS_GPU_2, places=6)