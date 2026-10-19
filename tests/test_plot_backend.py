"""绘图脚本的可运行性测试。

两点设计考虑：
1. 通过 --outdir 把图片写到临时目录，不污染仓库内的 assets/，CI 上可安全执行。
2. 语法检查用 compile() 而不是 py_compile —— 后者会尝试创建 __pycache__，
   在只读文件系统的环境（如某些沙箱 / 容器）里会失败。
"""
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "plot_evolution.py"
PY = sys.executable


def _run(args, **kw):
    """运行子进程并强制 UTF-8 解码（脚本会输出中文/表格字符）。"""
    return subprocess.run(
        [PY] + args,
        capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        cwd=str(ROOT), **kw,
    )


def test_plot_script_exists():
    assert SCRIPT.exists()


def test_plot_script_is_syntactically_valid():
    """用 compile() 做语法校验，不写任何字节码文件。"""
    source = SCRIPT.read_text(encoding="utf-8")
    compile(source, str(SCRIPT), "exec")


def test_plot_script_writes_both_figures_to_outdir(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("pandas")

    result = _run([str(SCRIPT), "--outdir", str(tmp_path)])
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"

    evolution = tmp_path / "evolution.png"
    distribution = tmp_path / "score_distribution.png"
    assert evolution.exists()
    assert distribution.exists()
    assert evolution.stat().st_size > 10_000


def test_plot_script_reports_the_three_generation_means(tmp_path):
    """汇总表打印的三代得分必须与 README 中的数据一致。"""
    pytest.importorskip("matplotlib")
    pytest.importorskip("pandas")

    result = _run([str(SCRIPT), "--outdir", str(tmp_path)])
    assert result.returncode == 0, result.stderr
    for token in ("6.224", "10.862", "30.442"):
        assert token in result.stdout, f"汇总输出缺少 {token}"
