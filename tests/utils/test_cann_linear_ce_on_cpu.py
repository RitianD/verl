# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""CPU unit tests for Ascend CANN fused linear-CE policy helpers."""

from __future__ import annotations

import pytest
import torch

from verl.utils.kernel.npu.cann_linear_ce import (
    CannLinearCrossEntropy,
    _resolve_vocab_range,
    should_use_cann_linear_ce,
)


def test_should_use_cann_linear_ce_non_npu_device(monkeypatch):
    monkeypatch.delenv("VERL_NPU_LCE_BACKEND", raising=False)
    monkeypatch.setattr(
        "verl.utils.kernel.npu.cann_linear_ce.is_cann_linear_ce_available",
        lambda: True,
    )
    assert should_use_cann_linear_ce(torch.device("cpu")) is False


def test_should_use_cann_linear_ce_triton_backend(monkeypatch):
    monkeypatch.setenv("VERL_NPU_LCE_BACKEND", "triton")
    monkeypatch.setattr(
        "verl.utils.kernel.npu.cann_linear_ce.is_cann_linear_ce_available",
        lambda: True,
    )

    class _NpuDev:
        type = "npu"

    assert should_use_cann_linear_ce(_NpuDev()) is False


def test_should_use_cann_linear_ce_auto_when_available(monkeypatch):
    monkeypatch.setenv("VERL_NPU_LCE_BACKEND", "auto")
    monkeypatch.setattr(
        "verl.utils.kernel.npu.cann_linear_ce.is_cann_linear_ce_available",
        lambda: True,
    )

    class _NpuDev:
        type = "npu"

    assert should_use_cann_linear_ce(_NpuDev()) is True


def test_should_use_cann_linear_ce_cann_backend_missing_apis(monkeypatch):
    monkeypatch.setenv("VERL_NPU_LCE_BACKEND", "cann")
    monkeypatch.setattr(
        "verl.utils.kernel.npu.cann_linear_ce.is_cann_linear_ce_available",
        lambda: False,
    )

    class _NpuDev:
        type = "npu"

    with pytest.raises(RuntimeError, match="VERL_NPU_LCE_BACKEND=cann"):
        should_use_cann_linear_ce(_NpuDev())


def test_cann_backward_rejects_nonzero_dentropy():
    ctx = object.__new__(object)
    with pytest.raises(RuntimeError, match="non-zero dentropy"):
        CannLinearCrossEntropy.backward(
            ctx,
            torch.ones(4, dtype=torch.float32),
            torch.ones(4, dtype=torch.float32),
        )


def test_resolve_vocab_range_is_half_open():
    """Ascend/Megatron use [start, end); end == start + local_vocab."""
    weight = torch.empty(2048, 512)
    start, end = _resolve_vocab_range(weight, None)
    assert (start, end) == (0, 2048)
    # Last valid id is end - 1.
    assert end - 1 == weight.shape[0] - 1


def test_resolve_vocab_range_with_tp(monkeypatch):
    weight = torch.empty(1024, 256)

    class _FakeGroup:
        pass

    group = _FakeGroup()
    monkeypatch.setattr(
        "verl.utils.kernel.npu.cann_linear_ce.dist.get_rank",
        lambda pg: 2 if pg is group else 0,
    )
    start, end = _resolve_vocab_range(weight, group)
    assert (start, end) == (2048, 3072)
