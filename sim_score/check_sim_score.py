import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def check_sim_score():
    for mode in ["train", "eval", "all"]:
        # 读取CSV文件
        file_path = f'/mnt/ssd1/yangxinyao/FuxiCTR/sim_score/sim_score_Yelp_{mode}.csv'  # 替换成你的CSV文件路径
        df = pd.read_csv(file_path, names=[f"column {c}" for c in range(100)])

        # 将DataFrame转换为numpy数组
        data = df.to_numpy()

        # 筛选出非零数据
        non_zero_data = data[data != 0]

        # 打印基本统计信息
        print(f"数据总数: {data.shape}")
        print(f"非零数据总数: {len(non_zero_data)}")
        print(f"非零数据占比: {len(non_zero_data) / (data.shape[0] * data.shape[1]) * 100:.2f}%")
        print(f"最小值: {np.min(non_zero_data)}")
        print(f"最大值: {np.max(non_zero_data)}")
        print(f"平均值: {np.mean(non_zero_data)}")
        print(f"标准差: {np.std(non_zero_data)}")

        # 绘制非零数据的分布图
        bucket_len = 0.004
        bucket_bins = np.arange(0.7, 1 + bucket_len, bucket_len)
        plt.hist(non_zero_data, bins=bucket_bins, edgecolor='black')
        plt.title(f'Non-Zero sim_score Distribution of {mode} Dataset')
        plt.xlabel('value')
        plt.ylabel('frequency')

        plt.text(0.95, 0.95, f'Max: {np.max(non_zero_data):.6f}', transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right')
        plt.text(0.95, 0.90, f'Min: {np.min(non_zero_data):.6f}', transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right')
        plt.text(0.95, 0.85, f'Mean: {np.mean(non_zero_data):.6f}', transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right')
        plt.text(0.95, 0.80, f'Standard deviation: {np.std(non_zero_data):.6f}', transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right')
        plt.savefig(f'/mnt/ssd1/yangxinyao/FuxiCTR/sim_score/distribution_histogram_Yelp_full_{mode}_v2.png')
        plt.close()

if __name__ == "__main__":
    check_sim_score()