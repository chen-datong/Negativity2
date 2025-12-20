import pandas as pd
import numpy as np

def load_results_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ["nA", "nB", "Nu", "NM"]:
        df[c] = df[c].astype(int)
    return df

def extract_arrays(df: pd.DataFrame, nA: int, NM: int):
    """
    固定 nA, NM，返回四个 numpy 数组：
    purity, thirdM, fourthM, fifthM
    """
    sub = df[(df["nA"] == nA) & (df["NM"] == NM)]
    if sub.empty:
        raise ValueError(f"No data for nA={nA}, NM={NM}")

    # # 排序只是为了让顺序稳定（不影响平均）
    # sub = sub.sort_values(["nB", "Nu"])

    purity  = sub["purity"].to_numpy(float)
    third   = sub["thirdM"].to_numpy(float)
    fourth  = sub["fourthM"].to_numpy(float)
    fifth   = sub["fifthM"].to_numpy(float)

    return purity, third, fourth, fifth

THEORY8 = {
2: 1.0000000000000009,
3: 0.7444526582608291,
4: 0.6794161346582085,
5: 0.6113730102112553,
}

THEORY = {2: 1.0000000000000013,
3: 0.7339992932054176, 
4: 0.6694761044425432, 
5: 0.5968853183347071}

def mean_abs_errors(purity, third, fourth, fifth, theory=THEORY):
    """
    返回 dict: {2: err2, 3: err3, 4: err4, 5: err5}
    """

    return {
        2: np.mean(np.abs(purity  - theory[2])),
        3: np.mean(np.abs(third   - theory[3])),
        4: np.mean(np.abs(fourth  - theory[4])),
        5: np.mean(np.abs(fifth   - theory[5])),
    }

def mean_abs_errors(purity, third, fourth, fifth, theory=THEORY):
    """
    返回 dict: {2: err2, 3: err3, 4: err4, 5: err5}
    """
    return {
        2: np.mean(np.abs(purity  - theory[2])),
        3: np.mean(np.abs(third   - theory[3])),
        4: np.mean(np.abs(fourth  - theory[4])),
        5: np.mean(np.abs(fifth   - theory[5])),
    }

def compute_errors_from_csv(csv_path: str, nA: int, NM: int):
    df = load_results_csv(csv_path)
    purity, third, fourth, fifth = extract_arrays(df, nA=nA, NM=NM)
    errors = mean_abs_errors(purity, third, fourth, fifth)
    return errors

errors = compute_errors_from_csv(
    "heisenberg_dbasis.csv",
    nA=11,
    NM=1000
)

print(errors)