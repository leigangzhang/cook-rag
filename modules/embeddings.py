"""Embedding 模型工具模块（单例）

封装 HuggingFaceEmbeddings 的初始化逻辑，支持：
- 自动检测 Apple MPS (Metal) 加速
- 使用本地 HuggingFace 缓存，避免远程拉取
- M4 芯片参数优化
- 单点维护 embedding 配置，避免多处重复定义
- 全局单例缓存，避免多次加载模型
- 强制离线模式，消除启动时的 HTTP HEAD 探测请求日志
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_huggingface import HuggingFaceEmbeddings

# 强制离线模式：模型已在本地缓存（92 MB），无需任何远程探测请求
# 同时禁用 telemetry 和进度条，从源头消除 HTTP HEAD 请求日志
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
"""嵌入模型名称（已本地缓存，约 92 MB）"""

_DEFAULT_CACHE_DIR = str(Path.home() / ".cache" / "huggingface" / "hub")
"""HuggingFace 默认缓存目录"""

_CACHED_EMBEDDING: Optional[HuggingFaceEmbeddings] = None
"""全局单例缓存，避免多次 create_embedding 重复加载模型"""


# ---------------------------------------------------------------------------
# 设备探测
# ---------------------------------------------------------------------------


def _detect_device() -> str:
    """自动选择最佳推理设备。

    Priority:
        1. mps  — Apple Silicon (M 系列)
        2. cpu  — fallback
    """
    import torch

    if torch.backends.mps.is_available():
        logger.info("检测到 Apple MPS (Metal) 加速可用，使用 device=mps")
        return "mps"
    logger.info("MPS 不可用，使用 device=cpu")
    return "cpu"


# ---------------------------------------------------------------------------
# 公共工厂函数（单例）
# ---------------------------------------------------------------------------


def create_embedding(
    model_name: str = _MODEL_NAME,
    device: Optional[str] = None,
    cache_dir: str = _DEFAULT_CACHE_DIR,
    normalize_embeddings: bool = True,
    **kwargs: Any,
) -> HuggingFaceEmbeddings:
    """创建 HuggingFace Embedding 模型实例（幂等单例）。

    默认使用本地缓存模型，配合 Mac M4 MPS 加速。
    首次调用创建并缓存实例，后续直接返回同一实例。

    Parameters
    ----------
    model_name : str
        HuggingFace 模型名称，默认 ``BAAI/bge-small-zh-v1.5``。
    device : str, optional
        推理设备，自动探测为 ``mps`` 或 ``cpu``。
    cache_dir : str
        本地模型缓存根目录。
    normalize_embeddings : bool
        是否归一化向量（余弦相似度要求）。
    **kwargs
        透传给 ``HuggingFaceEmbeddings`` 的额外参数。

    Returns
    -------
    HuggingFaceEmbeddings
    """
    global _CACHED_EMBEDDING
    if _CACHED_EMBEDDING is not None:
        return _CACHED_EMBEDDING

    if device is None:
        device = _detect_device()

    model_kwargs: Dict[str, Any] = {
        "device": device,
        "trust_remote_code": True,
    }
    model_kwargs.update(kwargs.pop("model_kwargs", {}))

    encode_kwargs: Dict[str, Any] = {
        "normalize_embeddings": normalize_embeddings,
    }
    encode_kwargs.update(kwargs.pop("encode_kwargs", {}))

    logger.info(
        "初始化 Embedding 模型 | model=%s device=%s cache=%s",
        model_name,
        device,
        cache_dir,
    )

    embedding = HuggingFaceEmbeddings(
        model_name=model_name,
        cache_folder=cache_dir,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        **kwargs,
    )

    _CACHED_EMBEDDING = embedding
    return embedding
