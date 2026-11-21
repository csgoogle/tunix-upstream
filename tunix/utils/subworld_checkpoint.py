# Copyright 2026 Google LLC
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

"""Sub-world Orbax checkpointing helpers for shared multi-process JAX worlds.

When trainer and rollout processes share one `jax.distributed` runtime (e.g.
for TPU ICI weight sync), only the trainer processes participate in checkpoint
save and restore. By default, Orbax sets `active_processes=None`, which causes
every barrier (`multihost.sync_global_processes` / `sync_global_devices`) to
wait on ALL processes in the JAX world, deadlocking on rollout processes that
never enter checkpointing.

This module scopes Orbax v0 (`Checkpointer`, `CheckpointManager`,
`PyTreeCheckpointHandler`, `StandardCheckpointHandler`, `ArrayHandler`) and
Orbax v1 (`Context`, `training.Checkpointer`, `_create_root_directory`)
barriers and primary-host checks strictly to the trainer's `active_processes`.
Because Orbax async checkpoint commits run on background threads that outlive
individual `save()` calls, sub-world configuration is process-lifetime once
enabled via `configure_subworld_checkpointing` (with `reset_for_testing`
provided strictly for test teardown).
"""

from __future__ import annotations

from collections.abc import Callable, Collection
import dataclasses
import inspect
import sys
import threading
import time
from typing import Any

from absl import logging
from etils import epath
import jax
from jax.experimental import multihost_utils
import orbax.checkpoint as ocp_v0
from orbax.checkpoint import checkpoint_manager as ocp_cm
from orbax.checkpoint import utils as ocp_utils
from orbax.checkpoint import v1 as ocp_v1

# Public re-export from `orbax.checkpoint` (avoids private `_src` imports).
_ocp_type_handlers = ocp_v0.type_handlers


def _loaded_orbax_module(name: str) -> Any:
  """Returns the already-imported internal Orbax module `name`.

  The public `orbax.checkpoint` packages imported above load these internals.
  Looking them up by name avoids private Orbax build dependencies.

  Args:
    name: Fully qualified module name.

  Returns:
    The loaded module.

  Raises:
    RuntimeError: If `name` is not loaded, e.g. after an Orbax refactor.
  """
  try:
    return sys.modules[name]
  except KeyError:
    raise RuntimeError(
        f"Orbax module {name!r} is not loaded; sub-world checkpointing must"
        " be updated for this Orbax version."
    ) from None


def _ocp_multihost() -> Any:
  """Returns Orbax's internal multihost module, safe against subpackage shadow.

  Importing `from orbax.checkpoint import multihost` replaces `ocp_v0.multihost`
  with the smaller public subpackage, so use the internal module instead.
  """
  return _loaded_orbax_module("orbax.checkpoint._src.multihost.multihost")


def _v1_multihost_modules() -> tuple[Any, Any]:
  """Returns the Orbax v1 modules whose `process_index` must be patched.

  `ocp_v1.multihost` re-exports the internal `process_index` by name, so both
  bindings need the patch.
  """
  return (
      _loaded_orbax_module(
          "orbax.checkpoint.experimental.v1._src.synchronization.multihost"
      ),
      ocp_v1.multihost,
  )


def _current_v1_context() -> Any:
  """Returns the ambient Orbax v1 Context, or a default one if none is set."""
  return _loaded_orbax_module(
      "orbax.checkpoint.experimental.v1._src.context.context"
  ).get_context()


_STATE_LOCK = threading.Lock()
_CONFIGURED_PROCESSES: frozenset[int] | None = None
_PATCHES_INSTALLED = False
_ORIG_PATCH_TARGETS: list[tuple[Any, str, Any]] = []
_ORIG_GLOBAL_ARRAY_PRIMARY_HOSTS: list[tuple[Any, int]] = []

# Maps JAX runtime process index (`jax.process_index()`) -> XLA
# CoordinationService distributed process id. On GKE TPU slices, JAX assigns
# runtime IDs by torus coordinates while CoordinationService indexes barrier
# `process_ids` by distributed task ID. Protected by `_MAP_LOCK`.
_MAP_LOCK = threading.Lock()
_RUNTIME_TO_DIST_ID: dict[int, int] = {}
_DIST_CLIENT_WRAPPER_INSTALLED = False
_ORIG_GET_JAX_DISTRIBUTED_CLIENT: Any = None


def jax_distributed_client() -> Any:
  """Returns the underlying JAX distributed client, or None if uninitialized."""
  mh = _ocp_multihost()
  if not mh.is_jax_distributed_client_initialized():
    return None
  with _STATE_LOCK:
    getter = (
        _ORIG_GET_JAX_DISTRIBUTED_CLIENT
        if _ORIG_GET_JAX_DISTRIBUTED_CLIENT is not None
        else mh.get_jax_distributed_client
    )
  return getter()


class _DistributedIdBarrierClient:
  """Proxy over JAX's distributed client translating runtime -> dist IDs."""

  def __init__(self, raw_client: Any):
    """Initializes the barrier client proxy around `raw_client`."""
    self._raw = raw_client

  def wait_at_barrier(
      self,
      barrier_id: str,
      timeout_in_ms: int,
      process_ids: Any = None,
  ) -> Any:
    """Waits at an XLA barrier, translating runtime process IDs to dist IDs."""
    if process_ids is None:
      return self._raw.wait_at_barrier(barrier_id, timeout_in_ms)
    if _ocp_multihost().use_experimental_distributed_process_id():
      raise NotImplementedError(
          "Sub-world checkpointing does not yet support"
          " use_experimental_distributed_process_id()=True."
      )
    requested = [int(p) for p in process_ids]
    requested_set = set(requested)
    if requested_set == set(range(jax.process_count())):
      return self._raw.wait_at_barrier(barrier_id, timeout_in_ms)
    with _MAP_LOCK:
      id_map = dict(_RUNTIME_TO_DIST_ID)
    missing = [p for p in requested if p not in id_map]
    if missing:
      raise RuntimeError(
          f"Missing runtime->distributed process ID mapping for {missing}"
          f" (requested={requested}, known={id_map})."
      )
    translated = [id_map[p] for p in requested]
    if len(set(translated)) != len(requested_set):
      raise RuntimeError(
          "Non-bijective runtime->distributed process ID mapping:"
          f" requested={requested}, translated={translated}, map={id_map}."
      )
    return self._raw.wait_at_barrier(
        barrier_id, timeout_in_ms, process_ids=translated
    )

  def __getattr__(self, name: str) -> Any:
    return getattr(self._raw, name)


def _own_distributed_process_id() -> int:
  """Returns this process's XLA CoordinationService task id."""
  # JAX has no public API for this id: `jax.process_index()` is the runtime
  # index, which differs from the CoordinationService task id on GKE TPU
  # slices. Orbax's `multihost.process_index` reads the same private state.
  state = jax._src.distributed.global_state  # pylint: disable=protected-access
  return int(state.process_id)


def publish_kv_idempotent(
    raw_client: Any, key: str, value: str, timeout_ms: int
) -> None:
  """Publishes `key=value` to the JAX KV store, verifying equality on reuse."""
  try:
    raw_client.key_value_set(key, value)
  except (RuntimeError, ValueError) as exc:
    readback_ms = max(1, min(int(timeout_ms), 5000))
    existing = raw_client.blocking_key_value_get(key, readback_ms)
    if str(existing) != value:
      raise RuntimeError(
          f"Conflicting KV value for {key!r}: existing={existing!r},"
          f" expected={value!r}."
      ) from exc


def _ensure_runtime_to_dist_map(active_processes: Collection[int]) -> None:
  """Publishes and resolves runtime -> dist task IDs for `active_processes`.

  Uses only per-process KV writes and blocking KV gets among `active_processes`
  so non-participating rollout processes never need to enter a global barrier.
  """
  mh = _ocp_multihost()
  if not mh.is_jax_distributed_client_initialized():
    return
  own_dist_id = _own_distributed_process_id()
  raw_client = jax_distributed_client()

  own_runtime_id = int(jax.process_index())
  sorted_procs = sorted({int(p) for p in active_processes})
  prefix = "tunix/subworld_ckpt/runtime_to_dist/"
  timeout_s = float(mh.coordination_timeout())
  deadline = time.monotonic() + timeout_s
  timeout_ms = max(1, int(timeout_s * 1000))

  with _MAP_LOCK:
    need_publish = own_runtime_id not in _RUNTIME_TO_DIST_ID
  if need_publish:
    publish_kv_idempotent(
        raw_client, f"{prefix}{own_runtime_id}", str(own_dist_id), timeout_ms
    )
    with _MAP_LOCK:
      _RUNTIME_TO_DIST_ID[own_runtime_id] = own_dist_id

  fetched: dict[int, int] = {}
  with _MAP_LOCK:
    missing_procs = [p for p in sorted_procs if p not in _RUNTIME_TO_DIST_ID]
  for proc in missing_procs:
    remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
    raw_val = raw_client.blocking_key_value_get(f"{prefix}{proc}", remaining_ms)
    fetched[proc] = int(raw_val)

  with _MAP_LOCK:
    _RUNTIME_TO_DIST_ID.update(fetched)
    resolved = {p: _RUNTIME_TO_DIST_ID[p] for p in sorted_procs}
  if len(set(resolved.values())) != len(resolved):
    raise RuntimeError(
        f"Non-bijective runtime->distributed id map for {sorted_procs}:"
        f" {resolved}."
    )
  logging.info(
      "Sub-world checkpoint runtime->distributed id map for %s: %s",
      sorted_procs,
      resolved,
  )


def _clear_signaling_client_cache() -> None:
  """Clears Orbax's cached signaling client if its module is already loaded."""
  sc_mod = sys.modules.get("orbax.checkpoint._src.futures.signaling_client")
  if sc_mod is None:
    return
  get_sc = getattr(sc_mod, "get_signaling_client", None)
  cache_clear = getattr(get_sc, "cache_clear", None)
  if callable(cache_clear):
    cache_clear()


def _install_distributed_id_barrier_translator_locked() -> None:
  """Installs a persistent proxy on `get_jax_distributed_client`."""
  global _DIST_CLIENT_WRAPPER_INSTALLED, _ORIG_GET_JAX_DISTRIBUTED_CLIENT
  if _DIST_CLIENT_WRAPPER_INSTALLED:
    return
  mh = _ocp_multihost()
  orig_getter = mh.get_jax_distributed_client
  _ORIG_GET_JAX_DISTRIBUTED_CLIENT = orig_getter

  def _wrapped_get_jax_distributed_client() -> Any:
    raw = orig_getter()
    if isinstance(raw, _DistributedIdBarrierClient):
      return raw
    return _DistributedIdBarrierClient(raw)

  mh.get_jax_distributed_client = _wrapped_get_jax_distributed_client
  _clear_signaling_client_cache()
  _DIST_CLIENT_WRAPPER_INSTALLED = True


def mesh_process_indices(mesh: jax.sharding.Mesh | None) -> frozenset[int]:
  """Returns the JAX process indices owning at least one device in `mesh`.

  Args:
    mesh: Target JAX mesh, or None when no mesh is available.

  Returns:
    Frozenset of integer process indices (empty if `mesh` is None).
  """
  if mesh is None:
    return frozenset()
  return frozenset(int(d.process_index) for d in mesh.devices.flat)


def state_process_indices(state: Any) -> frozenset[int]:
  """Returns the JAX process indices owning the `jax.Array` leaves of `state`.

  Args:
    state: PyTree such as `nnx.state(model)`; non-array leaves are ignored.

  Returns:
    Frozenset of integer process indices owning the arrays in `state`.
  """
  return frozenset(
      int(d.process_index)
      for leaf in jax.tree_util.tree_leaves(state)
      if isinstance(leaf, jax.Array)
      for d in leaf.sharding.device_set
  )


def is_subworld(active_processes: Collection[int] | None) -> bool:
  """Returns whether `active_processes` is a strict subset of this JAX world.

  Args:
    active_processes: Participating process indices, or None for the full world.

  Returns:
    True iff `jax.process_count() > 1` and `active_processes` is a non-empty
    strict subset of the JAX world.
  """
  if not active_processes:
    return False
  world_size = jax.process_count()
  if world_size <= 1:
    return False
  procs = frozenset(int(p) for p in active_processes)
  return len(procs) < world_size


def v0_multiprocessing_options(
    active_processes: Collection[int],
    base: ocp_v0.options.MultiprocessingOptions | None = None,
) -> ocp_v0.options.MultiprocessingOptions:
  """Builds Orbax v0 `MultiprocessingOptions` scoped to `active_processes`.

  Args:
    active_processes: Non-empty collection of participating process indices.
    base: Optional existing `MultiprocessingOptions` whose extra settings (such
      as `barrier_sync_key_prefix`) are preserved.

  Returns:
    An Orbax v0 `MultiprocessingOptions` with `primary_host=min(procs)` and
    `active_processes=set(procs)`.

  Raises:
    ValueError: If `active_processes` is empty.
  """
  procs = {int(p) for p in active_processes}
  if not procs:
    raise ValueError("active_processes must be non-empty.")
  primary_host = min(procs)
  if base is not None:
    return dataclasses.replace(
        base, primary_host=primary_host, active_processes=procs
    )
  return ocp_v0.options.MultiprocessingOptions(
      primary_host=primary_host,
      active_processes=procs,
  )


def configure_v1_context(
    context: ocp_v1.Context,
    active_processes: Collection[int] | None,
) -> ocp_v1.Context:
  """Returns a v1 `Context` scoped to `active_processes` when in a sub-world.

  Args:
    context: Base Orbax v1 `Context`.
    active_processes: Participating process indices, or None for the full world.

  Returns:
    `context` unchanged when not in a sub-world, otherwise a cloned `Context`
    with `multiprocessing_options` scoped to `active_processes`.
  """
  if active_processes is None or not is_subworld(active_processes):
    return context
  procs = {int(p) for p in active_processes}
  # The child copies every option of `context`, the barrier prefix included.
  scoped = ocp_v1.Context(context)
  scoped.multiprocessing.primary_host = min(procs)
  scoped.multiprocessing.active_processes = procs
  return scoped


def _with_subworld_cm_options(
    opts: ocp_v0.CheckpointManagerOptions | None,
    procs: frozenset[int],
) -> ocp_v0.CheckpointManagerOptions:
  """Returns `opts` with `multiprocessing_options` scoped to `procs`."""
  if opts is None:
    return ocp_v0.CheckpointManagerOptions(
        multiprocessing_options=v0_multiprocessing_options(procs)
    )
  mp = opts.multiprocessing_options
  if mp.active_processes is None:
    return dataclasses.replace(
        opts,
        multiprocessing_options=v0_multiprocessing_options(procs, base=mp),
    )
  return opts


def _registered_array_handlers() -> list[Any]:
  """Returns distinct `ArrayHandler` instances registered globally in Orbax."""
  seen: set[int] = set()
  handlers: list[Any] = []
  for typ in _ocp_type_handlers.supported_types():
    handler = _ocp_type_handlers.get_type_handler(typ)
    if isinstance(handler, _ocp_type_handlers.ArrayHandler):
      if id(handler) not in seen:
        seen.add(id(handler))
        handlers.append(handler)
  return handlers


def _set_global_array_handler_primary_host(primary_host: int) -> None:
  """Sets `_primary_host` on every globally registered `ArrayHandler`."""
  handlers = _registered_array_handlers()
  if not handlers:
    raise RuntimeError(
        "No Orbax ArrayHandler found in global type-handler registry; cannot"
        f" configure sub-world primary_host={primary_host}."
    )
  for handler in handlers:
    handler._primary_host = primary_host  # pylint: disable=protected-access


def _make_create_root_dir_wrapper(
    orig: Callable[..., Any],
) -> Callable[..., Any]:
  """Wraps `ocp_cm._create_root_directory` to support `active_processes`."""

  def _wrapped(
      directory: epath.PathLike,
      multiprocessing_options: ocp_v0.options.MultiprocessingOptions,
      file_options: ocp_v0.options.FileOptions | None = None,
  ) -> None:
    if multiprocessing_options.active_processes is None:
      orig(directory, multiprocessing_options, file_options)
      return
    path = epath.Path(directory)
    if not path.exists() and ocp_utils.is_primary_host(
        multiprocessing_options.primary_host
    ):
      path.mkdir(
          parents=True,
          exist_ok=True,
          mode=file_options.path_permission_mode if file_options else None,
      )
      logging.info("Created sub-world checkpoint directory=%s", path)
    mh = _ocp_multihost()
    mh.sync_global_processes(
        mh.unique_barrier_key(
            "CheckpointManager:create_directory",
            prefix=multiprocessing_options.barrier_sync_key_prefix,
        ),
        timeout=mh.coordination_timeout(),
        processes=multiprocessing_options.active_processes,
    )

  return _wrapped


def _make_v0_mp_init_wrapper(orig: Callable[..., Any]) -> Callable[..., Any]:
  """Wraps an Orbax v0 `__init__` to default `multiprocessing_options`."""

  def _wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
    procs = _CONFIGURED_PROCESSES
    if procs is not None:
      mp = kwargs.get("multiprocessing_options")
      if mp is None or mp.active_processes is None:
        kwargs["multiprocessing_options"] = v0_multiprocessing_options(
            procs, base=mp
        )
    return orig(self, *args, **kwargs)

  return _wrapped


def _make_v0_cm_init_wrapper(orig: Callable[..., Any]) -> Callable[..., Any]:
  """Wraps `ocp_v0.CheckpointManager.__init__` to inject sub-world options."""

  def _wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
    procs = _CONFIGURED_PROCESSES
    if procs is not None:
      if "options" not in kwargs and len(args) >= 3:
        args_list = list(args)
        args_list[2] = _with_subworld_cm_options(args_list[2], procs)
        args = tuple(args_list)
      else:
        kwargs["options"] = _with_subworld_cm_options(
            kwargs.get("options"), procs
        )
    return orig(self, *args, **kwargs)

  return _wrapped


def _make_v0_array_handler_init_wrapper(
    orig: Callable[..., Any],
) -> Callable[..., Any]:
  """Wraps `ArrayHandler.__init__` to default `primary_host=min(procs)`."""
  sig = inspect.signature(orig)

  def _wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
    procs = _CONFIGURED_PROCESSES
    if procs is not None:
      bound = sig.bind_partial(self, *args, **kwargs)
      if "primary_host" not in bound.arguments:
        kwargs["primary_host"] = min(procs)
    return orig(self, *args, **kwargs)

  return _wrapped


def _make_v1_checkpointer_init_wrapper(
    orig: Callable[..., Any],
) -> Callable[..., Any]:
  """Wraps `ocp_v1.training.Checkpointer.__init__` to scope `context`."""

  def _wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
    procs = _CONFIGURED_PROCESSES
    if procs is not None:
      ctx = kwargs.get("context")
      if ctx is None:
        ctx = _current_v1_context()
      if ctx.multiprocessing_options.active_processes is None:
        kwargs["context"] = configure_v1_context(ctx, procs)
    return orig(self, *args, **kwargs)

  return _wrapped


def _make_v1_process_index_wrapper(
    orig: Callable[[], int],
) -> Callable[[], int]:
  """Aligns Orbax v1 `process_index()` with runtime `jax.process_index()`."""

  def _wrapped() -> int:
    if _CONFIGURED_PROCESSES is not None:
      return int(jax.process_index())
    return orig()

  return _wrapped


def _make_sync_global_devices_wrapper(
    orig: Callable[[str], None],
) -> Callable[[str], None]:
  """Wraps `multihost_utils.sync_global_devices` to scope barriers."""

  def _wrapped(name: str) -> None:
    procs = _CONFIGURED_PROCESSES
    if procs is not None:
      _ocp_multihost().sync_global_processes(name, processes=set(procs))
      return
    orig(name)

  return _wrapped


def _patch_specs() -> tuple[
    tuple[Any, str, Callable[[Callable[..., Any]], Callable[..., Any]]], ...
]:
  """Declarative table of Orbax/JAX targets patched for sub-world safety."""
  # TODO(tunix-dev): Remove each entry once upstream Orbax supports sub-world
  # `create_root_directory`, global `ArrayHandler._primary_host`, consistent
  # runtime-ID v1 `process_index`, and MaxText exposes `multiprocessing_options`
  # directly.
  specs: list[
      tuple[Any, str, Callable[[Callable[..., Any]], Callable[..., Any]]]
  ] = [
      (ocp_cm, "_create_root_directory", _make_create_root_dir_wrapper),
      (ocp_v0.Checkpointer, "__init__", _make_v0_mp_init_wrapper),
      (ocp_v0.AsyncCheckpointer, "__init__", _make_v0_mp_init_wrapper),
      (ocp_v0.CheckpointManager, "__init__", _make_v0_cm_init_wrapper),
      (ocp_v0.PyTreeCheckpointHandler, "__init__", _make_v0_mp_init_wrapper),
      (ocp_v0.StandardCheckpointHandler, "__init__", _make_v0_mp_init_wrapper),
      (
          _ocp_type_handlers.ArrayHandler,
          "__init__",
          _make_v0_array_handler_init_wrapper,
      ),
      (
          ocp_v1.training.Checkpointer,
          "__init__",
          _make_v1_checkpointer_init_wrapper,
      ),
      (
          multihost_utils,
          "sync_global_devices",
          _make_sync_global_devices_wrapper,
      ),
  ]
  for v1_mh in _v1_multihost_modules():
    specs.append((v1_mh, "process_index", _make_v1_process_index_wrapper))
  return tuple(specs)


def _assert_compatible_processes_locked(procs: frozenset[int]) -> None:
  if _CONFIGURED_PROCESSES is not None and _CONFIGURED_PROCESSES != procs:
    raise ValueError(
        "Sub-world checkpointing is already configured for processes"
        f" {sorted(_CONFIGURED_PROCESSES)}; cannot reconfigure to"
        f" {sorted(procs)}."
    )


def _install_patches_locked() -> None:
  """Installs process-lifetime sub-world wrappers on Orbax and JAX."""
  global _PATCHES_INSTALLED
  if _PATCHES_INSTALLED:
    return
  _ORIG_GLOBAL_ARRAY_PRIMARY_HOSTS.clear()
  for handler in _registered_array_handlers():
    _ORIG_GLOBAL_ARRAY_PRIMARY_HOSTS.append(
        (handler, handler._primary_host)  # pylint: disable=protected-access
    )
  _ORIG_PATCH_TARGETS.clear()
  for target, attr, factory in _patch_specs():
    orig = getattr(target, attr)
    _ORIG_PATCH_TARGETS.append((target, attr, orig))
    setattr(target, attr, factory(orig))
  _PATCHES_INSTALLED = True


def configure_subworld_checkpointing(
    active_processes: Collection[int] | None,
) -> bool:
  """Configures process-lifetime sub-world Orbax checkpointing.

  No-op when `active_processes` is None or covers the entire JAX world (or in a
  single-process world). Once configured for a sub-world, patches remain active
  for the process lifetime so background Orbax async commit threads and
  teardown `close()` barriers never race against unpatching.

  Args:
    active_processes: JAX process indices participating in checkpointing, or
      None when the full JAX world participates.

  Returns:
    True if sub-world checkpointing is active for this process, False otherwise.

  Raises:
    ValueError: If `jax.process_index()` is not in `active_processes`, or if
      called again with a different `active_processes` set.
    RuntimeError: If the GKE runtime->distributed ID mapping fails or no global
      `ArrayHandler` is registered.
  """
  if active_processes is None or not is_subworld(active_processes):
    return False
  procs = frozenset(int(p) for p in active_processes)
  if jax.process_index() not in procs:
    raise ValueError(
        f"Current JAX process {jax.process_index()} is not in"
        f" active_processes={sorted(procs)}."
    )

  global _CONFIGURED_PROCESSES
  with _STATE_LOCK:
    _assert_compatible_processes_locked(procs)
    _install_distributed_id_barrier_translator_locked()

  # Resolve KV mapping outside `_STATE_LOCK` so blocking RPCs do not hold it.
  _ensure_runtime_to_dist_map(procs)

  with _STATE_LOCK:
    _assert_compatible_processes_locked(procs)
    _install_patches_locked()
    _CONFIGURED_PROCESSES = procs
    _set_global_array_handler_primary_host(min(procs))
  return True


def reset_for_testing() -> None:
  """Restores all patched Orbax/JAX globals and caches. Test-only."""
  global _CONFIGURED_PROCESSES, _PATCHES_INSTALLED
  global _DIST_CLIENT_WRAPPER_INSTALLED, _ORIG_GET_JAX_DISTRIBUTED_CLIENT
  with _STATE_LOCK:
    for target, attr, orig in reversed(_ORIG_PATCH_TARGETS):
      setattr(target, attr, orig)
    _ORIG_PATCH_TARGETS.clear()
    for handler, orig_primary in _ORIG_GLOBAL_ARRAY_PRIMARY_HOSTS:
      handler._primary_host = orig_primary  # pylint: disable=protected-access
    _ORIG_GLOBAL_ARRAY_PRIMARY_HOSTS.clear()
    _PATCHES_INSTALLED = False
    _CONFIGURED_PROCESSES = None
    if _DIST_CLIENT_WRAPPER_INSTALLED and (
        _ORIG_GET_JAX_DISTRIBUTED_CLIENT is not None
    ):
      _ocp_multihost().get_jax_distributed_client = (
          _ORIG_GET_JAX_DISTRIBUTED_CLIENT
      )
      _clear_signaling_client_cache()
    _DIST_CLIENT_WRAPPER_INSTALLED = False
    _ORIG_GET_JAX_DISTRIBUTED_CLIENT = None
  with _MAP_LOCK:
    _RUNTIME_TO_DIST_ID.clear()
