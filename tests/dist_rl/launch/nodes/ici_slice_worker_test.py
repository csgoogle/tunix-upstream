# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for `ici_slice_worker` torus-rank dispatch and validation."""

from __future__ import annotations

import os
import types
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from tunix.common import datatypes as common_datatypes
from tunix.dist_rl.distributed.runtime import context as context_lib
from tunix.dist_rl.launch.nodes import ici_slice_worker
from tunix.dist_rl.launch.nodes import node_utils


_BASE_ICI_ARGV = [
    "cluster_config.target=k8s",
    "cluster_config.k8s.tunix_image=img:latest",
    "cluster_config.trainer_replicas=2",
    "cluster_config.rollout_replicas=2",
    "weight_sync_config.mode=ici",
    "rollout_engine=inprocess_vllm",
]


def _fake_context() -> Any:
  ctx = mock.create_autospec(context_lib.ProcessContext, instance=True)
  ctx.ipc = types.SimpleNamespace(discovery=object())
  return ctx


class IciSliceWorkerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="low_rank_trainer",
          rank=1,
          stale_env=node_utils.ROLLOUT_REPLICA_ENV,
          expected_role="trainer",
          expected_env=node_utils.TRAINER_REPLICA_ENV,
          expected_replica="1",
      ),
      dict(
          testcase_name="high_rank_rollout",
          rank=3,
          stale_env=node_utils.TRAINER_REPLICA_ENV,
          expected_role="rollout",
          expected_env=node_utils.ROLLOUT_REPLICA_ENV,
          expected_replica="1",
      ),
  )
  def test_dispatches_rank_to_role(
      self,
      rank: int,
      stale_env: str,
      expected_role: str,
      expected_env: str,
      expected_replica: str,
  ):
    ctx = _fake_context()
    call_order: list[str] = []
    ctx.prepare_env.side_effect = lambda *a, **kw: call_order.append(
        "prepare_env"
    )
    ctx.initialize_runtime.side_effect = lambda *a, **kw: call_order.append(
        "initialize_runtime"
    )
    fakes = {
        "trainer": types.SimpleNamespace(main=mock.MagicMock()),
        "rollout": types.SimpleNamespace(main=mock.MagicMock()),
    }

    def _fake_import(name: str) -> Any:
      suffix = name.rsplit(".", 1)[-1]
      if suffix in fakes:
        return fakes[suffix]
      raise AssertionError(f"Unexpected import: {name}")

    with (
        mock.patch.dict(os.environ, {stale_env: "9"}, clear=False),
        mock.patch.object(jax, "process_index", return_value=rank),
        mock.patch.object(jax, "process_count", return_value=4),
        mock.patch.object(
            jax, "local_devices", return_value=[types.SimpleNamespace(id=rank)]
        ),
        mock.patch.object(
            ici_slice_worker.importlib,
            "import_module",
            side_effect=_fake_import,
        ),
    ):
      ici_slice_worker.dispatch_slice_worker(_BASE_ICI_ARGV, context=ctx)
      self.assertEqual(os.environ[expected_env], expected_replica)
      self.assertNotIn(stale_env, os.environ)

    self.assertEqual(call_order, ["prepare_env", "initialize_runtime"])
    self.assertEqual(
        ctx.prepare_env.call_args.kwargs["weight_sync_mode"],
        common_datatypes.WeightSyncMode.ICI,
    )
    other_role = "rollout" if expected_role == "trainer" else "trainer"
    fakes[expected_role].main.assert_called_once_with(_BASE_ICI_ARGV, ctx)
    fakes[other_role].main.assert_not_called()

  def test_rejects_world_size_mismatch(self):
    ctx = _fake_context()
    with (
        mock.patch.object(jax, "process_index", return_value=0),
        mock.patch.object(jax, "process_count", return_value=3),
    ):
      with self.assertRaisesRegex(ValueError, "world size"):
        ici_slice_worker.dispatch_slice_worker(_BASE_ICI_ARGV, context=ctx)

  def test_rejects_non_ici_weight_sync_mode(self):
    ctx = _fake_context()
    argv = [
        "cluster_config.target=k8s",
        "cluster_config.k8s.tunix_image=img:latest",
        "weight_sync_config.mode=raiden",
    ]
    with self.assertRaisesRegex(ValueError, "weight_sync_config.mode='ici'"):
      ici_slice_worker.dispatch_slice_worker(argv, context=ctx)


if __name__ == "__main__":
  absltest.main()
