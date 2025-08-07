# 步骤 0: 安装必要的库
# 如果您尚未安装 'datasets' 和 'pandas' 库，请先取消下面的注释并运行它

from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    """
    主函数，演示如何加载和使用 fathys/financial_news_sentiment 数据集
    """
    print("--- 步骤 1: 加载数据集 ---")
    # 使用 load_dataset 函数从 Hugging Face Hub 下载并加载数据集
    # 第一次运行时，它会自动下载数据并缓存到您的本地机器上
    try:
        dataset = load_dataset("fathys/financial_news_sentiment")
        print("数据集加载成功！")
    except Exception as e:
        print(f"数据集加载失败，请检查您的网络连接。错误: {e}")
        return

    print("\n--- 步骤 2: 查看数据集结构 ---")
    # 打印数据集对象，了解其结构
    # 通常它是一个 DatasetDict，包含不同的数据分割（如 'train', 'test'）
    print(dataset)

    # 这个特定的数据集只有一个 'train' 分割
    train_dataset = dataset['train']
    print(f"\n我们有 '{list(dataset.keys())[0]}' 数据分割，包含 {len(train_dataset)} 条数据。")


    print("\n--- 步骤 3: 检查单条数据样本 ---")
    # 查看第一条数据，了解其字段和内容
    sample = train_dataset[0]
    print("第一条数据样本:")
    print(sample)
    # 标签 '0' 代表负面, '1' 代表中性, '2' 代表正面
    label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    print(f"样本标签 '{sample['label']}' 对应的情感是: '{label_map[sample['label']]}'")


    print("\n--- 步骤 4: 将数据集转换为 Pandas DataFrame ---")
    # 为了更方便地进行数据操作和分析，通常会将其转换为 Pandas DataFrame
    df = train_dataset.to_pandas()
    print("已成功转换为 Pandas DataFrame。预览前5行数据:")
    print(df.head())


    print("\n--- 步骤 5: 进行简单的探索性数据分析 (EDA) ---")
    # 替换数字标签为可读的文本标签
    df['sentiment_label'] = df['label'].map(label_map)

    # 计算并打印各类情感标签的数量
    print("\n情感标签分布统计:")
    sentiment_counts = df['sentiment_label'].value_counts()
    print(sentiment_counts)

    # 使用图表可视化情感分布
    try:
        # 设置中文字体，防止绘图时乱码
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        plt.figure(figsize=(8, 5))
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis')
        plt.title('新闻情感标签分布')
        plt.xlabel('情感类别')
        plt.ylabel('新闻数量')
        print("\n正在生成情感分布图...")
        plt.show()
        print("图表已显示。")
    except Exception as e:
        print(f"\n无法生成图表，可能是缺少中文字体或绘图库问题。错误: {e}")
        print("不过，数据已经加载并处理完毕，可以用于后续步骤。")


if __name__ == '__main__':
    main()

