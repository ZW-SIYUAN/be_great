import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv("electricity_synthetic.csv")
df1 = pd.read_csv("electricity.csv")
# 只取前2000条
df = df.iloc[:2000]
df1 = df1.iloc[:2000]
# 转换标签
df["class"] = df["class"].map({"UP": 1, "DOWN": 0})
df1["class"] = df1["class"].map({"UP": 1, "DOWN": 0})
# 时间索引
df["t"] = range(len(df))
df1["t"] = range(len(df1))
# 滑动窗口
window = 200

rolling_mean = df["class"].rolling(window=window).mean()
rolling_mean1 = df1["class"].rolling(window=window).mean()
plt.figure()
plt.plot(df["t"], rolling_mean, label="Synthetic Data")
plt.plot(df1["t"], rolling_mean1, label="Original Data")
plt.xlabel("Time (0-2000)")
plt.ylabel("P(UP)")
plt.title("Concept Drift (First 2000 Samples)")
plt.legend()
plt.show()