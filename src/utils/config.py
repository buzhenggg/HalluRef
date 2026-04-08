"""配置加载工具"""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml


def _resolve_env_vars(value: str) -> str:
    """将 ${ENV_VAR} 替换为实际环境变量值"""
    def _replace(match):
        var_name = match.group(1)
        return os.getenv(var_name, "")
    return re.sub(r'\$\{(\w+)\}', _replace, value)


def _walk_and_resolve(obj):
    """递归解析配置中的环境变量"""
    if isinstance(obj, str):
        return _resolve_env_vars(obj)
    elif isinstance(obj, dict):
        return {k: _walk_and_resolve(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_walk_and_resolve(item) for item in obj]
    return obj


def load_config(path: str | None = None) -> dict:
    """加载 YAML 配置文件, 自动解析环境变量"""
    if path is None:
        # 默认路径: 项目根目录/config/config.yaml
        root = Path(__file__).resolve().parent.parent.parent
        path = str(root / "config" / "config.yaml")

    config_path = Path(path)
    if not config_path.exists():
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return _walk_and_resolve(raw)
