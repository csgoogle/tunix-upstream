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

"""Dynamic JAX process discovery for multi-host ICI trainers."""

from __future__ import annotations

import time

from absl import logging
import jax
from tunix.dist_rl.weight_sync import ici_transport
from tunix.utils import subworld_checkpoint

# KV-store timeout used when `cluster_config.init_timeout_s` is null: the
# int32 maximum in milliseconds (~24.8 days), effectively no deadline.
_NO_DEADLINE_MS = 2**31 - 1


def _remaining_ms(deadline: float | None) -> int:
  """Returns the milliseconds left until `deadline` (None: no deadline)."""
  if deadline is None:
    return _NO_DEADLINE_MS
  return max(1, int((deadline - time.monotonic()) * 1000))


def discover_trainer_process_indices(
    *,
    trainer_replicas: int,
    replica: int | None,
    timeout_s: float | None,
) -> list[int]:
  """Resolves the actual `jax.process_index()` set for all trainer hosts.

  On GKE TPU slices, `jax.distributed.initialize()` orders `jax.process_index()`
  by physical torus coordinate rather than Kubernetes `POD_INDEX`, so multi-host
  trainers must not assume their ranks equal `0..trainer_replicas-1`. With more
  than one trainer, every trainer replica publishes its actual
  `jax.process_index()` to the shared JAX distributed coordinator KV store and
  reads the others'.

  Args:
    trainer_replicas: Total number of trainer processes in this SPMD group.
    replica: This trainer's 0-based replica index (required when
      `trainer_replicas > 1`).
    timeout_s: Rendezvous timeout in seconds, or None for no deadline.

  Returns:
    Sorted list of distinct `jax.process_index()` values for the trainer group.

  Raises:
    ValueError: If `replica` is missing or out of range.
    RuntimeError: If the JAX distributed client is not initialized, or the
      trainers did not publish `trainer_replicas` distinct process indices.
  """
  if trainer_replicas <= 1:
    return [int(jax.process_index())]

  if replica is None or not 0 <= replica < trainer_replicas:
    raise ValueError(
        f"Trainer replica index {replica!r} is out of range for"
        f" trainer_replicas={trainer_replicas}."
    )

  client = subworld_checkpoint.jax_distributed_client()
  if client is None:
    raise RuntimeError(
        "JAX distributed client is not initialized; cannot discover"
        f" {trainer_replicas} multi-host trainer process indices."
    )
  # The coordinator KV store is scoped to this JAX world, so the keys need no
  # run id (a per-process default one would split the rendezvous).
  prefix = "tunix_ici_trainer_proc"
  deadline = None if timeout_s is None else time.monotonic() + timeout_s
  my_proc = int(jax.process_index())
  subworld_checkpoint.publish_kv_idempotent(
      client, f"{prefix}/{replica}", str(my_proc), _remaining_ms(deadline)
  )
  discovered: list[int] = []
  for idx in range(trainer_replicas):
    val = client.blocking_key_value_get(
        f"{prefix}/{idx}", _remaining_ms(deadline)
    )
    discovered.append(int(val))
  process_indices = sorted(set(discovered))
  if len(process_indices) != trainer_replicas:
    raise RuntimeError(
        f"Expected {trainer_replicas} distinct trainer JAX processes, got"
        f" {discovered}."
    )
  logging.info(
      "Dynamically discovered trainer JAX process indices via coordinator KV"
      " store: %s (this replica=%d -> process_index=%d).",
      process_indices,
      replica,
      my_proc,
  )
  return process_indices


def resolve_ici_trainer_devices(
    *,
    trainer_replicas: int,
    replica: int | None,
    timeout_s: float | None,
) -> list[jax.Device]:
  """Returns the trainer's device subset inside a shared ICI JAX world.

  Single-process, every device belongs to the trainer. In a multi-process ICI
  world, resolves every trainer process's actual `jax.process_index()` and
  returns their devices.

  Args:
    trainer_replicas: Total number of trainer processes in this SPMD group.
    replica: This trainer's 0-based replica index (required when
      `trainer_replicas > 1`).
    timeout_s: Rendezvous timeout in seconds, or None for no deadline.

  Returns:
    The trainer sub-world's devices, ordered by process index then device id.

  Raises:
    ValueError: If `trainer_replicas > 1` in a single-process JAX world, or if
      discovered process indices fall outside this JAX world or exclude this
      process.
  """
  if jax.process_count() <= 1:
    if trainer_replicas > 1:
      raise ValueError(
          f"cluster_config.trainer_replicas={trainer_replicas} requires a"
          " multi-process JAX world, got"
          f" jax.process_count()={jax.process_count()}."
      )
    return list(jax.devices())
  process_indices = discover_trainer_process_indices(
      trainer_replicas=trainer_replicas,
      replica=replica,
      timeout_s=timeout_s,
  )
  world = {d.process_index for d in jax.devices()}
  missing = [p for p in process_indices if p not in world]
  if missing:
    raise ValueError(
        f"Trainer process indices {missing} are outside this JAX world"
        f" {sorted(world)}."
    )
  if jax.process_index() not in process_indices:
    raise ValueError(
        f"This trainer process ({jax.process_index()}) is not in trainer"
        f" process indices {process_indices}."
    )
  return [d for p in process_indices for d in ici_transport.local_devices(p)]
