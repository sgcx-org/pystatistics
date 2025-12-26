"""
Hardware detection and device management.

Provides a unified interface for detecting available compute devices
and their capabilities, regardless of the underlying framework.
"""

from dataclasses import dataclass
from typing import Literal
import platform


@dataclass(frozen=True)
class DeviceInfo:
    """
    Information about a compute device.
    
    Attributes:
        device_type: Type of device ('cpu', 'cuda', 'mps')
        device_index: Device index (None for CPU)
        name: Human-readable device name
        memory_bytes: Total device memory in bytes (None if unknown)
        compute_capability: CUDA compute capability as (major, minor), None for non-CUDA
    """
    device_type: Literal['cpu', 'cuda', 'mps']
    device_index: int | None
    name: str
    memory_bytes: int | None
    compute_capability: tuple[int, int] | None
    
    def __str__(self) -> str:
        if self.device_type == 'cpu':
            return f"CPU ({self.name})"
        mem_str = ""
        if self.memory_bytes is not None:
            mem_gb = self.memory_bytes / (1024**3)
            mem_str = f", {mem_gb:.1f}GB"
        return f"{self.device_type.upper()}:{self.device_index} ({self.name}{mem_str})"
    
    @property
    def is_gpu(self) -> bool:
        """True if this is a GPU device."""
        return self.device_type in ('cuda', 'mps')


def detect_gpu() -> DeviceInfo | None:
    """
    Detect available GPU, if any.
    
    Returns:
        DeviceInfo for the best available GPU, or None if no GPU available.
        
    Priority: CUDA > MPS (Apple Silicon)
    
    Note:
        This function imports torch lazily to avoid import overhead
        when GPU detection isn't needed.
    """
    # Try CUDA first
    try:
        import torch
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            return DeviceInfo(
                device_type='cuda',
                device_index=idx,
                name=props.name,
                memory_bytes=props.total_memory,
                compute_capability=(props.major, props.minor)
            )
    except ImportError:
        pass
    
    # Try MPS (Apple Silicon)
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return DeviceInfo(
                device_type='mps',
                device_index=0,
                name='Apple Silicon GPU',
                memory_bytes=None,  # MPS doesn't expose memory info easily
                compute_capability=None
            )
    except ImportError:
        pass
    
    return None


def get_cpu_info() -> DeviceInfo:
    """
    Get CPU device info.
    
    Returns:
        DeviceInfo for the CPU
    """
    processor = platform.processor()
    if not processor:
        processor = platform.machine() or "Unknown CPU"
    
    return DeviceInfo(
        device_type='cpu',
        device_index=None,
        name=processor,
        memory_bytes=None,
        compute_capability=None
    )


def select_device(prefer: Literal['cpu', 'gpu', 'auto'] = 'auto') -> DeviceInfo:
    """
    Select compute device based on preference and availability.
    
    Args:
        prefer: Device preference
            - 'cpu': Always use CPU
            - 'gpu': Require GPU (raises if unavailable)
            - 'auto': Use GPU if available, else CPU
            
    Returns:
        DeviceInfo for selected device
        
    Raises:
        RuntimeError: If 'gpu' requested but no GPU available
    """
    if prefer == 'cpu':
        return get_cpu_info()
    
    gpu = detect_gpu()
    
    if prefer == 'gpu':
        if gpu is None:
            raise RuntimeError(
                "GPU requested but no GPU available. "
                "Ensure PyTorch is installed with CUDA/MPS support."
            )
        return gpu
    
    # auto: prefer GPU if available
    return gpu if gpu is not None else get_cpu_info()
