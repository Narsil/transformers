import unittest

import numpy as np

from transformers.modeling_tf_pytorch_utils import TransposeType


class TransposeMeta(type):
    def __new__(mcs, name, bases, dic):
        def gen_test(tensor, transpose):
            def test(self):
                self.is_invertible(tensor, transpose)

            return test

        # Prime dimensions to ensure no multiplications can induce errors
        tensors = [
            np.random.random((2,)),
            np.random.random((2, 3)),
            np.random.random((2, 3, 5)),
            np.random.random((2, 3, 5, 7)),
            np.random.random((2, 3, 5, 7, 11)),
            # Squeeze requires dim=1
            np.random.random((1, 2, 3, 5)),
        ]

        transposes = list(TransposeType)
        for tensor in tensors:
            for transpose in transposes:
                try:
                    transpose.apply(tensor)
                except Exception:
                    # Some transposes cannot be applied to certain tensors

                    continue

                test_name = f"test_invertible_{transpose}_rank_{len(tensor.shape)}"
                dic[test_name] = gen_test(tensor, transpose)
        return type.__new__(mcs, name, bases, dic)


class TranposeTypeTestCase(unittest.TestCase, metaclass=TransposeMeta):
    def is_invertible(self, tensor, transpose):
        applied = transpose.apply(tensor)
        reverted = transpose.apply_reverse(applied)

        self.assertTrue(
            np.allclose(tensor, reverted), "The TransposeType is not invertible {tensor.shape} != {reverted.shape}"
        )
