from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class DistributedRuntime:
    enabled: bool
    world_size: int
    process_index: int
    local_process_index: int
    device: Optional[Any] = None
    _state: Optional[Any] = None

    @classmethod
    def from_env(cls) -> "DistributedRuntime":
        requested_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if requested_world_size <= 1:
            device = None
            try:
                import torch
            except ImportError:
                torch = None
            if torch is not None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return cls(
                enabled=False,
                world_size=1,
                process_index=0,
                local_process_index=0,
                device=device,
            )

        try:
            from accelerate import PartialState
        except ImportError as exc:
            raise ImportError(
                "accelerate is required for multi-GPU training. Install dependencies from requirements-rl.txt."
            ) from exc

        state = PartialState()
        return cls(
            enabled=int(state.num_processes) > 1,
            world_size=int(state.num_processes),
            process_index=int(state.process_index),
            local_process_index=int(state.local_process_index),
            device=getattr(state, "device", None),
            _state=state,
        )

    @property
    def is_main_process(self) -> bool:
        return self.process_index == 0

    def wait_for_everyone(self) -> None:
        if self.enabled and self._state is not None:
            self._state.wait_for_everyone()

    def broadcast_object(self, value: Any) -> Any:
        if not self.enabled:
            return value

        try:
            import torch.distributed as dist
        except ImportError as exc:
            raise ImportError(
                "torch.distributed is required for multi-GPU training."
            ) from exc

        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("Distributed process group is not initialized.")

        objects = [value if self.is_main_process else None]
        dist.broadcast_object_list(objects, src=0)
        return objects[0]

    def all_gather_object(self, value: Any) -> List[Any]:
        if not self.enabled:
            return [value]

        try:
            import torch.distributed as dist
        except ImportError as exc:
            raise ImportError(
                "torch.distributed is required for multi-GPU training."
            ) from exc

        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("Distributed process group is not initialized.")

        objects: List[Any] = [None for _ in range(self.world_size)]
        dist.all_gather_object(objects, value)
        return objects
