"""evaluation_results.csv 的数据完整性测试。

这三份 CSV 是 README 中「三代对比表」的唯一数据来源。
如果它们被误改、被覆盖或列名变更，README 里的结论就会失真，
所以把数据契约固化成测试。
"""
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

EXPECTED_COLUMNS = {"seed", "score", "total_reward", "avg_state_error", "safety_violations"}
EXPECTED_ROWS = 100

GENERATIONS = ["gen2_td", "gen3_history", "gen4_tcn"]


@pytest.fixture(params=GENERATIONS)
def df(request):
    pd = pytest.importorskip("pandas")
    path = RESULTS / request.param / "evaluation_results.csv"
    assert path.exists(), f"缺少评测记录：{path}"
    return pd.read_csv(path)


def test_has_expected_columns(df):
    assert EXPECTED_COLUMNS.issubset(set(df.columns)), f"列名不符：{list(df.columns)}"


def test_row_count_matches_seed_budget(df):
    """每个版本都跑满 100 个随机种子，这是"结论不是挑出来的最好结果"的依据。"""
    assert len(df) == EXPECTED_ROWS


def test_seeds_are_unique_and_zero_based(df):
    assert df["seed"].is_unique
    assert sorted(df["seed"].tolist()) == list(range(EXPECTED_ROWS))


def test_no_missing_values(df):
    assert not df.isnull().any().any()


def test_scores_are_positive_and_finite(df):
    assert (df["score"] > 0).all()
    assert (df["score"] < 100).all()


def test_state_error_is_non_negative(df):
    assert (df["avg_state_error"] >= 0).all()


def test_generations_are_ordered_as_documented():
    """README 声称的单调提升：gen2 < gen3 < gen4。若数据被替换，这里会报警。"""
    pytest.importorskip("pandas")
    import pandas as pd

    means = {}
    for gen in GENERATIONS:
        means[gen] = pd.read_csv(RESULTS / gen / "evaluation_results.csv")["score"].mean()

    assert means["gen2_td"] < means["gen3_history"] < means["gen4_tcn"], (
        f"三代得分不再单调提升：{means}"
    )
