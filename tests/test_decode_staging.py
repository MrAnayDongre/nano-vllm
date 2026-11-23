import unittest

import torch
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import reset_context


class DecodeStagingTest(unittest.TestCase):
    def test_prepare_decode_after_graph_capture(self):
        self.addCleanup(reset_context)
        runner = ModelRunner.__new__(ModelRunner)
        runner.enforce_eager = False
        runner.block_size = Sequence.block_size
        runner.graph_bs = [2]
        runner.graphs = {}
        # capture_cudagraph allocates the real staging tensors in inference mode.
        with torch.inference_mode():
            runner.decode_staging = {
                "input_ids": torch.empty(2, dtype=torch.int64),
                "positions": torch.empty(2, dtype=torch.int64),
                "slot_mapping": torch.empty(2, dtype=torch.int32),
                "context_lens": torch.empty(2, dtype=torch.int32),
                "block_tables": torch.empty((2, 1), dtype=torch.int32),
            }
        seq = Sequence([11, 22])
        seq.block_table = [3]
        ids, positions = runner.prepare_decode([seq])
        self.assertEqual(ids.tolist(), [22])
        self.assertEqual(positions.tolist(), [1])
        self.assertEqual(runner.decode_staging["slot_mapping"].tolist(), [3 * runner.block_size + 1, -1])
        self.assertEqual(runner.decode_staging["context_lens"].tolist(), [2, 0])
        self.assertEqual(runner.decode_staging["block_tables"].tolist(), [[3], [0]])


if __name__ == "__main__":
    unittest.main()
