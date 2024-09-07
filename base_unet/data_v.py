import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# Load the data from the Excel file
path = r"savefile/CGNet_train1.xlsx"
savef = r"CGNet_train1.png"

# 加载 Excel 文件数据
df = pd.read_excel(path)

# 绘制损失值随 Epoch 变化的图
plt.figure(figsize=(10, 6))
sns.lineplot(x='epoch', y='loss', data=df, marker='o')
plt.title('Training loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # 保证横轴标签为整数
plt.grid(True)
plt.show()

# 绘制训练准确率随 Epoch 变化的图
plt.figure(figsize=(10, 6))
sns.lineplot(x='epoch', y='train_acc', data=df, marker='o')
plt.title('Tr accuracy')
plt.xlabel('Epoch')
plt.ylabel('Tr accuracy')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.grid(True)
plt.show()

# 绘制测试准确率随 Epoch 变化的图
plt.figure(figsize=(10, 6))
sns.lineplot(x='epoch', y='test_acc', data=df, marker='o')
plt.title('Te accuracy')
plt.xlabel('Epoch')
plt.ylabel('Te accuracy')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.grid(True)
plt.show()

# 绘制训练时间随 Epoch 变化的图
plt.figure(figsize=(10, 6))
sns.lineplot(x='epoch', y='time', data=df, marker='o')
plt.title('Time')
plt.xlabel('Epoch')
plt.ylabel('Time(s)')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.grid(True)
plt.show()

# 将多个指标合并在一张图中
plt.figure(figsize=(12, 8))
sns.lineplot(x='epoch', y='loss', data=df, marker='o', label='Loss')
sns.lineplot(x='epoch', y='train_acc', data=df, marker='o', label='Training Accuracy')
sns.lineplot(x='epoch', y='test_acc', data=df, marker='o', label='Test Accuracy')
plt.title(fr'Multi values : {savef.replace(".png", "")}')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.grid(True)
plt.savefig('data_v/'+savef)
plt.show()
