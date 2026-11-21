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

"""Unit tests for sub-world Orbax checkpointing utilities."""

from __future__ import annotations

import contextlib
import sys
from typing import Any
from unittest import mock

from absl.testing import absltest
import jax
import orbax.checkpoint as ocp_v0
from orbax.checkpoint import v1 as ocp_v1
from tunix.utils import subworld_checkpoint


class _FakeDistClient:
  """Minimal in-memory fake of JAX's distributed coordination client."""

  def __init__(self, initial_kv: dict[str, str] | None = None):
    self.kv: dict[str, str] = dict(initial_kv or {})
    self.barrier_calls: list[tuple[str, int, Any]] = []

  def key_value_set(
      self, key: str, value: str, allow_overwrite: bool = False
  ) -> None:
    if key in self.kv and not allow_overwrite:
      raise RuntimeError(f"ALREADY_EXISTS: {key}")
    self.kv[key] = value

  def blocking_key_value_get(self, key: str, timeout_in_ms: int) -> str:
    del timeout_in_ms
    if key not in self.kv:
      raise RuntimeError(f"NOT_FOUND: {key}")
    return self.kv[key]

  def wait_at_barrier(
      self,
      barrier_id: str,
      timeout_in_ms: int,
      process_ids: Any = None,
  ) -> None:
    self.barrier_calls.append((barrier_id, timeout_in_ms, process_ids))


@contextlib.contextmanager
def _patch_subworld_env(
    raw_client: _FakeDistClient,
    *,
    process_count: int = 4,
    runtime_id: int = 1,
    dist_id: int = 5,
):
  """Patches JAX + Orbax multihost globals to simulate a GKE TPU world."""
  mh = subworld_checkpoint._ocp_multihost()  # pylint: disable=protected-access
  with (
      mock.patch.object(jax, "process_count", return_value=process_count),
      mock.patch.object(jax, "process_index", return_value=runtime_id),
      mock.patch.object(
          subworld_checkpoint,
          "_own_distributed_process_id",
          return_value=dist_id,
      ),
      mock.patch.object(
          mh, "is_jax_distributed_client_initialized", return_value=True
      ),
      mock.patch.object(
          mh, "get_jax_distributed_client", return_value=raw_client
      ),
      mock.patch.object(
          mh, "use_experimental_distributed_process_id", return_value=False
      ),
  ):
    yield mh


class SubworldCheckpointTest(absltest.TestCase):

  def tearDown(self):
    subworld_checkpoint.reset_for_testing()
    super().tearDown()

  def test_barrier_client_translates_runtime_to_distributed_ids(self):
    raw = _FakeDistClient(
        initial_kv={"tunix/subworld_ckpt/runtime_to_dist/2": "7"}
    )
    with _patch_subworld_env(raw, runtime_id=1, dist_id=5) as mh:
      self.assertTrue(
          subworld_checkpoint.configure_subworld_checkpointing({1, 2})
      )
      wrapped = mh.get_jax_distributed_client()
      # Sub-world barrier translates runtime {1, 2} -> distributed [5, 7].
      wrapped.wait_at_barrier("ckpt_sync", 1000, process_ids=[1, 2])
      # Global barrier (all 4 runtime IDs) translates to process_ids=None.
      wrapped.wait_at_barrier("global_sync", 1000, process_ids=[0, 1, 2, 3])
      # Unknown runtime process ID fails fast instead of falling back.
      with self.assertRaisesRegex(
          RuntimeError, "Missing runtime->distributed process ID mapping"
      ):
        wrapped.wait_at_barrier("bad_sync", 1000, process_ids=[1, 3])

    self.assertEqual(
        raw.barrier_calls,
        [
            ("ckpt_sync", 1000, [5, 7]),
            ("global_sync", 1000, None),
        ],
    )

  def test_v1_process_index_matches_runtime_id_on_permuted_world(self):
    raw = _FakeDistClient(
        initial_kv={"tunix/subworld_ckpt/runtime_to_dist/2": "7"}
    )
    v1_mh = sys.modules.get(
        "orbax.checkpoint.experimental.v1._src.synchronization.multihost"
    )
    self.assertIsNotNone(v1_mh)
    assert v1_mh is not None
    with _patch_subworld_env(raw, runtime_id=1, dist_id=5):
      subworld_checkpoint.configure_subworld_checkpointing({1, 2})
      self.assertEqual(v1_mh.process_index(), 1)
      self.assertTrue(v1_mh.is_primary_host(1))

  def test_async_checkpointer_receives_subworld_multiprocessing_options(self):
    raw = _FakeDistClient(
        initial_kv={"tunix/subworld_ckpt/runtime_to_dist/2": "2"}
    )
    with _patch_subworld_env(raw, runtime_id=1, dist_id=1):
      subworld_checkpoint.configure_subworld_checkpointing({1, 2})
      ckptr = ocp_v0.AsyncCheckpointer(
          ocp_v0.PyTreeCheckpointHandler(use_ocdbt=False)
      )
      try:
        # pylint: disable=protected-access
        self.assertEqual(ckptr._primary_host, 1)
        self.assertEqual(ckptr._active_processes, {1, 2})
        # pylint: enable=protected-access
      finally:
        ckptr.close()

  def test_barrier_client_rejects_experimental_distributed_id_mode(self):
    raw = _FakeDistClient()
    client = subworld_checkpoint._DistributedIdBarrierClient(raw)
    mh = subworld_checkpoint._ocp_multihost()
    with mock.patch.object(
        mh, "use_experimental_distributed_process_id", return_value=True
    ):
      with self.assertRaisesRegex(
          NotImplementedError, "use_experimental_distributed_process_id"
      ):
        client.wait_at_barrier("b", 1000, process_ids=[1, 2])

  def test_configure_patches_array_handler_and_preserves_prefix(self):
    raw = _FakeDistClient(
        initial_kv={"tunix/subworld_ckpt/runtime_to_dist/2": "2"}
    )
    with _patch_subworld_env(raw, runtime_id=1, dist_id=1):
      self.assertTrue(
          subworld_checkpoint.configure_subworld_checkpointing({1, 2})
      )
      # ArrayHandler default primary_host becomes min({1, 2}) == 1.
      default_handler = ocp_v0.type_handlers.ArrayHandler()
      self.assertEqual(
          default_handler._primary_host, 1  # pylint: disable=protected-access
      )
      # Passing positional metadata_key still injects primary_host=1.
      keyed_handler = ocp_v0.type_handlers.ArrayHandler("meta_key")
      self.assertEqual(
          keyed_handler._primary_host, 1  # pylint: disable=protected-access
      )
      # Explicit primary_host=0 (positional or keyword) must be preserved.
      explicit_handler = ocp_v0.type_handlers.ArrayHandler(None, 0)
      self.assertEqual(
          explicit_handler._primary_host, 0  # pylint: disable=protected-access
      )

      # Global type handler registry is patched in place.
      global_arr_handler = ocp_v0.type_handlers.get_type_handler(jax.Array)
      self.assertIsInstance(
          global_arr_handler, ocp_v0.type_handlers.ArrayHandler
      )
      self.assertEqual(
          global_arr_handler._primary_host,  # pylint: disable=protected-access
          1,
      )

      # v0_multiprocessing_options preserves existing barrier_sync_key_prefix.
      base_mp = ocp_v0.options.MultiprocessingOptions(
          primary_host=0, barrier_sync_key_prefix="my_prefix"
      )
      scoped = subworld_checkpoint.v0_multiprocessing_options(
          {1, 2}, base=base_mp
      )
      self.assertEqual(scoped.primary_host, 1)
      self.assertEqual(scoped.active_processes, {1, 2})
      self.assertEqual(scoped.barrier_sync_key_prefix, "my_prefix")

      # Reconfiguring to a different active_processes set fails fast.
      with self.assertRaisesRegex(ValueError, "already configured"):
        subworld_checkpoint.configure_subworld_checkpointing({1, 3})

  def test_configure_v1_context_scopes_primary_host_and_prefix(self):
    base = ocp_v1.Context()
    base.multiprocessing.barrier_sync_key_prefix = "keep_me"
    with mock.patch.object(jax, "process_count", return_value=4):
      scoped = subworld_checkpoint.configure_v1_context(base, {1, 2})
    self.assertEqual(scoped.multiprocessing_options.primary_host, 1)
    self.assertEqual(scoped.multiprocessing_options.active_processes, {1, 2})
    self.assertEqual(
        scoped.multiprocessing_options.barrier_sync_key_prefix, "keep_me"
    )
    # The base context is left unscoped.
    self.assertEqual(base.multiprocessing.primary_host, 0)
    self.assertIsNone(base.multiprocessing.active_processes)

  def test_publish_kv_idempotent_accepts_equal_and_rejects_conflicting(self):
    client = _FakeDistClient()
    subworld_checkpoint.publish_kv_idempotent(client, "k", "1", 1000)
    subworld_checkpoint.publish_kv_idempotent(client, "k", "1", 1000)
    with self.assertRaisesRegex(RuntimeError, "Conflicting KV value"):
      subworld_checkpoint.publish_kv_idempotent(client, "k", "2", 1000)
    self.assertEqual(client.kv["k"], "1")


if __name__ == "__main__":
  absltest.main()
