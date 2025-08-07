import torch, pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "hfl/chinese-bert-wwm"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
model      = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)  # 修改为3分类

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

print("正在加载模型权重...")
try:
    model.load_state_dict(torch.load("./best.pt", map_location=device))
    print("模型权重加载成功")
except Exception as e:
    print(f"模型加载失败: {e}")
    exit(1)

print("正在将模型移至设备...")
model.to(device).eval()
print("模型准备完成")

def predict_list(texts, batch_size=4):
    preds = []
    from tqdm import tqdm
    for i in tqdm(range(0, len(texts), batch_size), desc="预测进度"):
        batch = texts[i:i+batch_size]
        enc = tokenizer(
            batch,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
            preds.extend(logits.argmax(-1).cpu().numpy())
    return preds

# 预测 data.csv
print("正在读取数据文件...")
df = pd.read_csv('./data/data.csv')
print(f"读取到 {len(df)} 条数据")
print(f"数据列名: {df.columns.tolist()}")

df['text'] = df['title'] + '。' + df['raw']
print("开始预测...")
df['predict_sentiment'] = predict_list(df['text'].astype(str).tolist())  # 列名建议改为 predict_sentiment
print("正在保存结果...")
df.to_csv('./data/data_pred.csv', index=False, encoding='utf-8-sig')
print("已生成 data_pred.csv，含 predict_sentiment 列（0=消极，1=中性，2=积极）")