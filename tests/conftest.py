"""pytest 公共配置。

处理三件在 Windows / CI 上都容易踩的事：
1. 被测 agent 会 print 中文与 emoji（如 ⚠），Windows 控制台默认 GBK 会抛
   UnicodeEncodeError —— 这里强制 stdout/stderr 用 UTF-8。
2. 被测 agent 在初始化时会把工作目录 chdir 到自己的目录（原比赛提交代码的
   遗留行为）—— 每个测试后把 cwd 还原，避免污染后续测试的相对路径。
3. 让 `import gen4_tcn_agent` 这类平铺导入可用（源码在 src/ 下）。
"""
import os
import pathlib
import sys

import pytest

# ---- 1) 强制 UTF-8 输出 ----
for _name in ("stdout", "stderr"):
    _stream = getattr(sys, _name, None)
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# ---- 2) 让 src/ 下的平铺模块可被导入 ----
SRC_DIR = pathlib.Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def restore_cwd():
    """测试结束后还原工作目录（被测代码初始化时会 chdir）。"""
    original = os.getcwd()
    yield
    try:
        os.chdir(original)
    except OSError:
        os.chdir(str(REPO_ROOT))
