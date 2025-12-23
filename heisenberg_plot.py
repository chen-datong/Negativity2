import pandas as pd
import numpy as np
from XXZ import ppt_moment, ppt_moment_depolarize
from utils import ppt3, D3, D4, D5
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.sans-serif"] = 'Times New Roman'
matplotlib.rcParams["mathtext.fontset"] = 'cm'




def load_results_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ["nA", "nB", "Nu", "NM"]:
        df[c] = df[c].astype(int)
    return df

def extract_arrays(df: pd.DataFrame, nA: int, nB: int, NM: int):
    """
    固定 nA, NM，返回四个 numpy 数组：
    purity, thirdM, fourthM, fifthM
    """
    sub = df[(df["nA"] == nA) & (df["nB"] == nB) & (df["NM"] == NM)]
    if sub.empty:
        raise ValueError(f"No data for nA={nA}, nB={nB}, NM={NM}")

    # # 排序只是为了让顺序稳定（不影响平均）
    # sub = sub.sort_values(["nB", "Nu"])

    purity  = sub["purity"].to_numpy(float)
    third   = sub["thirdM"].to_numpy(float)
    fourth  = sub["fourthM"].to_numpy(float)
    fifth   = sub["fifthM"].to_numpy(float)

    return purity, third, fourth, fifth


def extract_arrays_depol(df: pd.DataFrame, nA: int, nB: int, NM: int, dprob: float):
    """
    固定 nA, NM，返回四个 numpy 数组：
    purity, thirdM, fourthM, fifthM
    """
    sub = df[(df["nA"] == nA) & (df["nB"] == nB) & (df["NM"] == NM) & (df["dprob"] == dprob)]
    if sub.empty:
        raise ValueError(f"No data for nA={nA}, nB={nB}, NM={NM}, dprob={dprob}")

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

def mean_abs_errors(purity, third, fourth, fifth, theory=THEORY8):
    """
    返回 dict: {2: err2, 3: err3, 4: err4, 5: err5}
    """

    return {
        2: np.mean(np.abs(purity  - theory[2])),
        3: np.mean(np.abs(third   - theory[3])),
        4: np.mean(np.abs(fourth  - theory[4])),
        5: np.mean(np.abs(fifth   - theory[5])),
    }

def compute_errors_from_csv(csv_path: str, nA: int, nB: int, NM: int, theory):
    df = load_results_csv(csv_path)
    purity, third, fourth, fifth = extract_arrays(df, nA=nA, nB=nB, NM=NM)
    errors = mean_abs_errors(purity, third, fourth, fifth, theory)
    return errors

def exact_depolarize(nA, nB, save_prefix: str = None, load_prefix: str = None):
    if save_prefix:
        ps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ppt3s = []
        D3s = []
        D4s = []
        for p in ps:
            theory = ppt_moment_depolarize(nA, nB, p, 5)
            ppt3s.append(ppt3(theory[2], theory[3]))
            D3s.append(D3(theory[2], theory[3]))
            D4s.append(D4(theory[2], theory[3], theory[4]))
        
        ps_arr = np.array(ps)
        ppt3s = np.array(ppt3s)
        D3s = np.array(D3s)
        D4s = np.array(D4s)

        if save_prefix:
            np.save(save_prefix + "_ps.npy", ps_arr)
            np.save(save_prefix + "_ppt3s.npy", ppt3s)
            np.save(save_prefix + "_D3s.npy", D3s)
            np.save(save_prefix + "_D4s.npy", D4s)
    else:
        ppt3s = np.load(load_prefix + "_ppt3s.npy")
        D3s = np.load(load_prefix + "_D3s.npy")
        D4s = np.load(load_prefix + "_D4s.npy")
    ps_arr = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    plt.plot(ps_arr, ppt3s, label='PPT3')
    plt.plot(ps_arr, D3s, label='D3')
    plt.plot(ps_arr, D4s, label='D4')
    plt.hlines(0, 0.0, 0.9)
    plt.legend()
    plt.show()

    return ps_arr, ppt3s, D3s, D4s


def heisenberg_depol(nA, nB, NM, csv_path='heisenberg_dbasis_prob.csv'):
    df = load_results_csv(csv_path)
    ps = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ppt3_mean = np.array([0] * 10, dtype=float)
    ppt3_std = np.array([0] * 10, dtype=float)
    D3_mean = np.array([0] * 10, dtype=float)
    D3_std = np.array([0] * 10, dtype=float)
    D3_min = np.array([0] * 10, dtype=float)
    D3_max = np.array([0] * 10, dtype=float)
    D4_mean = np.array([0] * 10, dtype=float)
    D4_std = np.array([0] * 10, dtype=float)
    D5_mean = np.array([0] * 10, dtype=float)
    D5_std = np.array([0] * 10, dtype=float)
    for i, p in enumerate(ps):
        purity_tmp, third_tmp, fourth_tmp, fifth_tmp = extract_arrays_depol(df, nA=nA, nB=nB, NM=NM, dprob=p)
        ppt3 = purity_tmp * purity_tmp - third_tmp
        d3 = D3(purity_tmp, third_tmp)
        d4 = D4(purity_tmp, third_tmp, fourth_tmp)
        d5 = D5(purity_tmp, third_tmp, fourth_tmp, fifth_tmp)
        ppt3_mean[i] = np.mean(ppt3)
        ppt3_std[i] = np.std(ppt3)
        D3_mean[i] = np.mean(d3)
        D3_std[i] = np.std(d3)
        D3_min[i] = np.min(d3)
        D3_max[i] = np.max(d3)
        D4_mean[i] = np.mean(d4)
        D4_std[i] = np.std(d4)
        D5_mean[i] = np.mean(d5)
        D5_std[i] = np.std(d5)

    exact = np.load('heisenberg/depol_nA11_nB1_ppt3s.npy')
    plt.figure(figsize=(10, 6.18))
    # plt.plot(ps, ppt3_mean, marker='.', markersize=10, label=r'$p_2^2-p_3$')
    # # plt.plot(ps, exact, label='exact', c='r')
    # plt.fill_between(
    #     ps,
    #     ppt3_mean - ppt3_std,
    #     ppt3_mean + ppt3_std,
    #     alpha=0.3
    # )
    plt.plot(ps, D3_mean, marker='o', markersize=7, label=r'$D_3$')
    plt.fill_between(
        ps,
        D3_mean - D3_std,
        D3_mean + D3_std,
        alpha=0.3
    )
    plt.plot(ps, D4_mean, marker='^', markersize=7, label=r'$D_4$')
    plt.fill_between(
        ps,
        D4_mean - D4_std,
        D4_mean + D4_std,
        alpha=0.3
    )
    plt.plot(ps, D5_mean, marker='s', markersize=7, label=r'$D_5$')
    plt.fill_between(
        ps,
        D5_mean - D5_std,
        D5_mean + D5_std,
        alpha=0.3
    )
    plt.axhline(y=0, linestyle=(0, (6, 6)), linewidth=1.5, color='black')
    plt.xlabel('Depolarizing Probability', fontsize=20)
    # plt.ylabel(r'$p_2^2-p_3$')
    plt.xlim(left=-0.01, right=0.91)
    plt.legend(fontsize=20)
    plt.tick_params(axis='both', which='both', direction='in', labelsize=20, pad=8)
    plt.savefig('./figure/Depol_ent_detect.pdf')
    plt.show()






# errors = compute_errors_from_csv(
#     "heisenberg_dbasis.csv",
#     nA=8,
#     nB=2,
#     NM=1000,
#     theory=ppt_moment(8, 2, 5)
# )

# print(errors)

# exact_depolarize(11, 1, load_prefix='heisenberg/depol_nA11_nB1')
if __name__ == '__main__':
    heisenberg_depol(11, 1, 100, csv_path='heisenberg_dbasis_prob.csv')