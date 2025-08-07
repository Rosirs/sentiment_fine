# 新浪财经7×24实时情感分析系统

本项目是一个基于FinBERT的中文财经新闻实时情感分析系统，能够自动抓取新浪财经7×24小时新闻，进行实时情感分析，并提供可视化看板展示市场情绪变化。

## 🌟 主要功能

- **实时新闻抓取**: 自动抓取新浪财经7×24小时新闻数据
- **智能情感分析**: 使用FinBERT模型进行专业财经文本情感分析
- **商品分类**: 自动识别新闻关联的商品期货品种
- **实时看板**: 提供交互式可视化看板，展示情绪趋势和品种分布
- **词云生成**: 支持按品种和情感生成词云分析
- **后台监控**: 支持后台定时抓取和实时更新

## 📁 项目结构

```
sentiment_run/
├── main.py                    # 主程序入口
├── dashboard.py               # 可视化看板
├── bot/
│   ├── fetch.py              # 新闻抓取模块
│   ├── nlp.py                # 情感分析模块
│   ├── items_classify.py     # 商品分类模块
│   └── translate.py          # 翻译模块
├── data/                     # 数据目录
│   ├── sina_7x24_latest.xlsx # 最新抓取的新闻数据
│   └── ...                   # 其他数据文件
├── model/                    # 模型文件
├── train.py                  # 模型训练脚本
├── predict.py                # 预测脚本
└── requirements.txt          # 依赖包列表
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd sentiment_run
activate env(如有)

# 安装依赖
pip install -r requirements.txt
```

### 2. 基础使用

#### 抓取新闻并进行情感分析
```bash
# 抓取100条最新新闻并生成csv文件
python main.py -l 100

# 同时生成交互式图表
python main.py -l 100 --plot
```

#### 启动实时看板
```bash
# 启动可视化看板（包含词云和实时更新）
python main.py -l 100 --dashboard
```

### 3. 高级功能

#### 模型训练
```bash
# 使用自定义参数训练模型
python train.py --batch_size 8 --epochs 50 --lr 1e-3
```

#### 批量预测
```bash
# 对CSV文件进行批量情感分析
python predict.py
```

## 📊 功能详解

### 新闻抓取 (bot/fetch.py)
- 自动抓取新浪财经7×24小时新闻API
- 支持多页数据获取
- 自动解析新闻标题和内容
- 实时保存为CSV格式

### 情感分析 (bot/nlp.py)
- 使用FinBERT专业财经情感分析模型
- 支持批量文本处理
- 输出情感分数和标签（positive/neutral/negative）
- 自动处理文本长度限制

### 商品分类 (bot/items_classify.py)
- 覆盖主要商品期货品种：
  - 黑色系：螺纹钢、铁矿石、焦炭等
  - 能源化工：原油、PTA、甲醇等
  - 有色金属：铜、铝、锌等
  - 农产品：豆粕、白糖、棉花等
  - 贵金属：黄金、白银

### 可视化看板 (dashboard.py)
- 实时情绪时间序列图
- 品种提及次数统计
- 交互式词云生成
- 支持按品种和情感筛选
- 自动定时刷新

## 🔧 配置说明

### 环境变量
```bash
# 可选：设置代理（如果需要）
export HTTP_PROXY=http://proxy:port
export HTTPS_PROXY=http://proxy:port
```

### 参数配置
- `--limit`: 抓取新闻条数（默认100）
- `--plot`: 生成交互式图表
- `--dashboard`: 启动实时看板
- `--batch_size`: 训练批次大小
- `--epochs`: 训练轮数
- `--lr`: 学习率

## 📈 数据格式

### 输入数据格式
```csv
dt,title,raw,commodity_type
2024-01-01 10:00:00,标题,内容,螺纹钢
```

### 输出数据格式示例
```json
  {
    "dt": "2025-08-06 12:48:31",
    "text": "。印度央行行长：将确保满足经济的生产性需求。",
    "sentiment": "positive",
    "confidence": 0.85,
    "impact_horizon": "medium",
    "reason": "新闻提到印度央行行长将确保满足经济的生产性需求，这表明央行对经济的支持和积极态度，对经济有正面影响。"
  },
```

## 🛠️ 技术栈

- **Python 3.8+**
- **Transformers**: 汉化版的FinBERT模型（有点大，跑更轻量的）
- **PyTorch**: 深度学习框架（cpu版本）
- **Pandas**: 数据处理
- **Plotly**: 交互式可视化
- **Dash**: Web应用框架
- **httpx**: 异步HTTP请求
- **jieba**: 中文分词

## 📝 使用示例

### 1. 实时监控市场情绪

获取帮助
```bash
python main.py --help
```

抓取最新的100条新闻
```bash
# 抓取1000条新闻，结束（fetch_only）
python main.py --l 1000 -f
```

### 2. ai_check.py

调用kimi api直接分析文本
使用前将"./env/sk-xxx"替换为你的api key
batch_size 建议设置成10或者以下，(20报错)

```bash
python ai_check.py
```

### 3. dashboard.py

看板代码
```bash
streamlit run dashboard.py
```

## 🔍 故障排除

### 常见问题

1. **模型下载失败**
   ```bash
   # 设置镜像源
   pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
   ```

2. **内存不足**
   ```bash
   # 减少批次大小
   ```

3. **网络连接问题**
   ```bash
   # 检查网络连接
   ping zhibo.sina.com.cn
   ```
     