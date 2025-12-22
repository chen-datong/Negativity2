import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker

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
        plt.plot(nA_data['NM'], nA_data['result'], label=f'$n_A = {nA}$', marker='.', markersize=10, color = color)

    # 添加图例
    plt.legend(fontsize=20)
    plt.xscale('log')
    plt.yscale('log')
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    # 设置标题和标签
    # plt.xlim(90, 110000)
    # plt.ylim(0.4, 9)
    plt.xlabel(r'$N_M$', fontsize=20)
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
    nu_data = df[(df['nA'] == 10)].copy()
    nu = nu_data['Nu'].unique()
    colors = np.linspace(0.2, 0.95, len(nu))[::-1]
    plt.figure(figsize=(10, 6.18))
    for i, n in enumerate(nu):
        # 以 NM 为横轴，result 为纵轴画图
        plot_data = nu_data[nu_data['Nu'] == n].copy()
        plt.plot(plot_data['NM'], plot_data['result'], label=f'$N_U = {n}$', marker='.', markersize=10, color = cmap(colors[i]))
    plt.legend(fontsize=20)
    plt.xscale('log')
    plt.yscale('log')
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    # 设置标题和标签
    # plt.xlim(90, 110000)
    # plt.ylim(0.4, 9)
    plt.xlabel(r'$N_M$', fontsize=20)
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



# 调用函数画图
plot_nA_dependence()
# plot_converge_val()
plot_Nu_dependence()