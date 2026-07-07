from pathlib import Path

import numpy as np
import pytest

from t8_agent.train.linear_policy import LinearPolicy


def test_linear_policy_load_rejects_wrong_shape(tmp_path: Path) -> None:
    checkpoint = tmp_path / "bad_policy.npz"
    np.savez_compressed(checkpoint, weights=np.zeros((1, 1), dtype=np.float32))

    with pytest.raises(ValueError, match="weights shape"):
        LinearPolicy.load(checkpoint)
