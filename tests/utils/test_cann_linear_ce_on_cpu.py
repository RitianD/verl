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
    disable_npu_fused_kernels_if_entropy_enabled,
    should_use_cann_linear_ce,
)


@pytest.mark.parametrize(
    ("use_fused", "entropy_coeff", "device_name", "expected"),
    [
        (False, 0.0, "npu", False),
        (False, 0.1, "npu", False),
        (True, 0.0, "npu", True),
        (True, 0.0, "cuda", True),
        # Non-zero entropy: keep fused on non-NPU, disable on NPU.
        (True, 0.01, "cuda", True),
        (True, 0.01, "npu", False),
        (True, 1.0, "npu", False),
    ],
)
def test_disable_npu_fused_kernels_if_entropy_enabled(monkeypatch, use_fused, entropy_coeff, device_name, expected):
    monkeypatch.setattr(
        "verl.utils.device.get_device_name",
        lambda: device_name,
    )
    assert (
        disable_npu_fused_kernels_if_entropy_enabled(
            use_fused_kernels=use_fused,
            entropy_coeff=entropy_coeff,
            context="unit-test",
        )
        is expected
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


def test_actor_config_validate_disables_fused_on_npu(monkeypatch):
    from omegaconf import OmegaConf

    from verl.workers.config.actor import ActorConfig

    monkeypatch.setattr("verl.utils.device.get_device_name", lambda: "npu")

    cfg = ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        use_fused_kernels=True,
        entropy_coeff=0.01,
    )
    model_cfg = OmegaConf.create({"use_fused_kernels": True})
    cfg.validate(n_gpus=1, train_batch_size=256, model_config=model_cfg)

    assert cfg.use_fused_kernels is False
    assert model_cfg.use_fused_kernels is False


def test_actor_config_validate_keeps_fused_when_entropy_zero(monkeypatch):
    from omegaconf import OmegaConf

    from verl.workers.config.actor import ActorConfig

    monkeypatch.setattr("verl.utils.device.get_device_name", lambda: "npu")

    cfg = ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        use_fused_kernels=True,
        entropy_coeff=0.0,
    )
    model_cfg = OmegaConf.create({"use_fused_kernels": True})
    cfg.validate(n_gpus=1, train_batch_size=256, model_config=model_cfg)

    assert cfg.use_fused_kernels is True
    assert model_cfg.use_fused_kernels is True
