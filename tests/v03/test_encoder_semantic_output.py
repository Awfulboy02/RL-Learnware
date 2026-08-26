from __future__ import annotations

import numpy as np
import pytest

from policy_learnware_v0.v03.encoder_protocol import (
    EncoderProtocolError,
    SemanticSampleBatch,
)


def test_semantic_output_requires_boolean_nonempty_valid_support() -> None:
    values = np.asarray([[0.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    with pytest.raises(EncoderProtocolError, match="must be boolean"):
        SemanticSampleBatch(
            values=values,
            valid_mask=np.asarray([1, 0], dtype=np.int64),
            window_ids=("window-a", "window-b"),
        )
    with pytest.raises(EncoderProtocolError, match="at least one"):
        SemanticSampleBatch(
            values=values,
            valid_mask=np.asarray([False, False]),
            window_ids=("window-a", "window-b"),
        )
