import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker
from XXZ import ppt_moments_from_rho, depolarize_state
from utils import D3, D4, ppt3, B2
from Gibbs import load_results_csv, extract_arrays_gibbs

matplotlib.rcParams["font.sans-serif"] = 'Times New Roman'
matplotlib.rcParams["mathtext.fontset"] = 'cm'

def plot_nA_dependence(csv_filename='experiment_results_dbasis.csv'):
    # 读取CSV文件
    df = pd.read_csv(csv_filename)
    cmap = LinearSegmentedColormap.from_list("deep_light_blue", ["#00098B", "#4159E1D4", "#87A7EBE2", "#ADD8E6"])
    # 获取唯一的 nA 值
    nA_values = df['nA'].unique()
    # 设置画布
    plt.figure(figsize=(10, 6.18))

    # 遍历不同的 nA 值，画出每条曲线
    for nA in nA_values[2:]:
        # 筛选出当前 nA 的数据
        nA_data = df[(df['nA'] == nA) & (df['Nu'] == 1)].copy()
        color = cmap(1 - (nA - 7) / (len(nA_values) - 2))
        # 以 NM 为横轴，result 为纵轴画图
        plt.plot(nA_data['NM'], nA_data['thirdM'], label=f'$n_A = {nA}$', marker='.', markersize=10, color = color)

    # 添加图例
    plt.legend(fontsize=20, loc='upper right')
    plt.xscale('log')
    plt.yscale('log')
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    # 设置标题和标签
    # plt.xlim(90, 110000)
    # plt.ylim(0.4, 9)
    plt.xlabel(r'$N_M/d$', fontsize=20)
    # plt.ylabel(r'$\langle|\epsilon|\rangle$', fontsize=20)
    plt.ylabel(r'Statistical Error', fontsize=20)
    plt.tick_params(axis='both', which='both', direction='in', labelsize=20, pad=8)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    # 显示图形
    plt.savefig('./figure/SE_nA.pdf')

    plt.show()


def plot_Nu_dependence(filename='experiment_results_dbasis.csv'):
    df = pd.read_csv(filename)
    cmap = LinearSegmentedColormap.from_list(
        "deep_light_green",
        ["#003300", "#1b5e20", "#388e3c", "#66bb6a", "#c8e6c9"]
    )
    nu_data = df[(df['nA'] == 9)].copy()
    nu = nu_data['Nu'].unique()
    colors = np.linspace(0.2, 0.95, len(nu))[::-1]
    plt.figure(figsize=(10, 6.18))
    for i, n in enumerate(nu):
        # 以 NM 为横轴，result 为纵轴画图
        plot_data = nu_data[nu_data['Nu'] == n].copy()
        plt.plot(plot_data['NM'], plot_data['thirdM'], label=f'$N_U = {n}$', marker='.', markersize=10, color = cmap(colors[i]))
    plt.legend(fontsize=20, loc='upper right')
    plt.xscale('log')
    plt.yscale('log')
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    # 设置标题和标签
    # plt.xlim(90, 110000)
    # plt.ylim(0.4, 9)
    plt.xlabel(r'$N_M/d$', fontsize=20)
    # plt.ylabel(r'$\langle|\epsilon|\rangle$', fontsize=20)
    plt.ylabel(r'Statistical Error', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tick_params(axis='both', which='both', direction='in', labelsize=20, pad=8)
    plt.tight_layout()
    # 显示图形
    plt.savefig('./figure/SE_Nu.pdf')
    plt.show()
    

def plot_converge_val(csv_filename='experiment_results.csv'):
    # 读取CSV文件
    df = pd.read_csv(csv_filename)
    cmap = LinearSegmentedColormap.from_list("deep_light_blue", ["#00008B", "#4144E1D5", "#87A7EBE2", "#ADD8E6"])
    # 获取唯一的 nA 值
    nA_values = df['nA'].unique()
    converge = df[(df['NM'] == 100000) & (df['Nu'] == 1)]['result']

    # 设置画布
    plt.figure(figsize=(10, 6.18))

    plt.scatter(nA_values, converge, color = '#00008B')
    coeffs = np.polyfit(nA_values, np.log(converge), 1)
    poly = np.exp(np.polyval(coeffs, nA_values))
    plt.plot(nA_values, poly, linewidth=2, color = '#00008B')


    # 添加图例
    # plt.legend(fontsize=20)
    # plt.xscale('log')
    plt.yscale('log')
    ax = plt.gca()
    # ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.ticklabel_format(style='plain', axis='y')
    # 设置标题和标签
    plt.xlabel(r'$n_A$', fontsize=20)
    plt.ylabel(r'$\langle|\epsilon|\rangle$ Convergency value', fontsize=20)
    plt.xticks([5, 6, 7, 8, 9], fontsize=20)
    plt.yticks([0.1, 0.2, 0.4, 0.8], fontsize=20)
    plt.tight_layout()
    # plt.savefig('./figure/convergency_NM.pdf')
    # 显示图形
    plt.show()

def exact_depol_GHZ(nA, nB, sys="B", save_prefix: str = None, load_prefix: str = None):
    if save_prefix:
        N = nA + nB
        d = 2 ** N
        dimA = 2 ** nA
        dimB = 2 ** nB
        psi = np.zeros(d, dtype=complex)
        psi[0] = 1 / np.sqrt(2)
        psi[-1] = 1 / np.sqrt(2)
        rho = np.outer(psi, psi.conj())
        ps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ppt3s = []
        D3s = []
        D4s = []
        for p in ps:
            rho_depol = depolarize_state(rho, d, p)
            moments = ppt_moments_from_rho(rho_depol, dimA, dimB, t_max=5, sys=sys)
            p2 = moments[2]
            p3 = moments[3]
            p4 = moments[4]
            ppt3s.append(ppt3(p2, p3))
            D3s.append(D3(p2, p3))
            D4s.append(D4(p2, p3, p4))
        ppt3s = np.array(ppt3s)
        D3s = np.array(D3s)
        D4s = np.array(D4s)
        np.save(f'{save_prefix}_nA{nA}_nB{nB}_ppt3s.npy', ppt3s)
        np.save(f'{save_prefix}_nA{nA}_nB{nB}_D3s.npy', D3s)
        np.save(f'{save_prefix}_nA{nA}_nB{nB}_D4s.npy', D4s)
    else:
        ppt3s = np.load(f'{load_prefix}_nA{nA}_nB{nB}_ppt3s.npy')
        D3s = np.load(f'{load_prefix}_nA{nA}_nB{nB}_D3s.npy')
        D4s = np.load(f'{load_prefix}_nA{nA}_nB{nB}_D4s.npy')
    ps_arr = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    plt.plot(ps_arr, ppt3s, label='PPT3')
    plt.plot(ps_arr, D3s, label='D3')
    plt.plot(ps_arr, D4s, label='D4')
    plt.hlines(0, 0.0, 0.9)
    plt.legend()
    plt.show()
    
    return ps_arr, ppt3s, D3s, D4s

def gibbs_ent_detect(nA, nB, NM, csv_path='gibbs_dbasis.csv'):
    df = load_results_csv(csv_path)
    betas = np.array([0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5])
    # betas = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    ppt3_mean = np.array([0] * 10, dtype=float)
    ppt3_std = np.array([0] * 10, dtype=float)
    B2_mean = np.array([0] * 10, dtype=float)
    B2_std = np.array([0] * 10, dtype=float)
    for i, beta in enumerate(betas):
        purity_tmp, third_tmp, fourth_tmp, fifth_tmp = extract_arrays_gibbs(df, nA=nA, nB=nB, NM=NM, beta=beta)
        PPT3 = ppt3(purity_tmp, third_tmp)
        b2 = B2(purity_tmp, third_tmp, fourth_tmp, fifth_tmp)
        ppt3_mean[i] = np.mean(PPT3)
        ppt3_std[i] = np.std(PPT3)
        B2_mean[i] = np.mean(b2)*10
        B2_std[i] = np.std(b2)*10
    print(B2_mean)

    plt.figure(figsize=(10, 6.18))
    plt.plot(betas, ppt3_mean, marker='.', markersize=10, label=r'$-|B_1|$')
    # plt.plot(ps, exact, label='exact', c='r')
    plt.fill_between(
        betas,
        ppt3_mean - ppt3_std,
        ppt3_mean + ppt3_std,
        alpha=0.3
    )
    plt.plot(betas, B2_mean, marker='o', markersize=7, label=r'$-10|B_2|$')
    plt.fill_between(
        betas,
        B2_mean - B2_std,
        B2_mean + B2_std,
        alpha=0.3
    )
    plt.axhline(y=0, linestyle=(0, (6, 6)), linewidth=1.5, color='black')
    plt.xlabel(r'$\beta$', fontsize=20)
    # plt.ylabel(r'$p_2^2-p_3$')
    # plt.xlim(left=-0.01, right=0.91)
    plt.legend(fontsize=20)
    plt.tick_params(axis='both', which='both', direction='in', labelsize=20, pad=8)
    plt.savefig('./figure/Gibbs_ent_detect.pdf')
    plt.show()


if __name__ == '__main__':
    # 调用函数画图
    # plot_nA_dependence()
    # plot_converge_val()
    # plot_Nu_dependence()
    gibbs_ent_detect(nA=11, nB=1, NM=1000)