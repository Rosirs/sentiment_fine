import torch, pandas as pd, numpy as np
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)
from torch.optim import AdamW
import traceback
from tqdm import tqdm
import sys
import gc
import argparse
import os

# 命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser(description='情感分析模型训练')
    parser.add_argument('--model_name', type=str, default='hfl/chinese-bert-wwm', 
                       help='预训练模型名称')
    parser.add_argument('--batch_size', type=int, default=4, 
                       help='批次大小')
    parser.add_argument('--max_len', type=int, default=128, 
                       help='最大序列长度')
    parser.add_argument('--lr', type=float, default=1e-2, 
                       help='学习率')
    parser.add_argument('--epochs', type=int, default=200, 
                       help='训练轮数')
    parser.add_argument('--patience', type=int, default=3, 
                       help='早停耐心值')
    parser.add_argument('--min_delta', type=float, default=0.001, 
                       help='最小提升阈值')
    parser.add_argument('--device', type=str, default='cpu', 
                       help='训练设备 (cpu/cuda)')
    return parser.parse_args()

args = parse_args()
print('程序启动')
# 使用命令行参数
MODEL_NAME = args.model_name

# 查看模型参数大小
try:
    model_tmp = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    total_params = sum(p.numel() for p in model_tmp.parameters())
    print(f"模型参数总数: {total_params:,}")
    del model_tmp
    gc.collect()
except Exception as e:
    print("加载模型时出错:", e)
    traceback.print_exc()
    sys.exit(1)

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Tokenizer加载成功")
except Exception as e:
    print("Tokenizer加载失败:", e)
    traceback.print_exc()
    sys.exit(1)

# --------------------------------------------------
# 1) 读取数据
try:
    print("开始读取数据...")
    df_train = pd.read_csv('./data/cleaned_data.csv')
    print(f"原始数据形状: {df_train.shape}")
    print(f"原始列名: {list(df_train.columns)}")
    
    df_train['train_text'] = df_train['title'] + '。' + df_train['content']
    frac = 1/3
    df_train = df_train.sample(frac=frac, random_state=42).reset_index(drop=True)  

    # 只保留需要的列
    df_train = df_train[['train_text', 'sentiment']].dropna()
    print(f"处理后数据形状: {df_train.shape}")
    
    # 先去除 sentiment 列的空格，并转为 int
    df_train['sentiment'] = df_train['sentiment'].astype(str).str.strip().astype(int)
    # 再做映射
    sentiment_mapping = {-1: 0, 0: 1, 1: 2}
    df_train['sentiment'] = df_train['sentiment'].map(sentiment_mapping)
    # 检查是否有未映射的值
    if df_train['sentiment'].isnull().any():
        print('存在未映射的sentiment值:', df_train[df_train['sentiment'].isnull()])
        df_train = df_train.dropna(subset=['sentiment'])
    df_train['sentiment'] = df_train['sentiment'].astype(int)
    print("数据处理完成")
    
except Exception as e:
    print("数据处理时出错:", e)
    traceback.print_exc()
    sys.exit(1)

# 2) 划分训练/验证
try:
    from sklearn.model_selection import train_test_split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df_train['train_text'].tolist(),
        df_train['sentiment'].tolist(),
        test_size=0.15,
        random_state=42,
        stratify=df_train['sentiment']
    )
    print(f"训练集大小: {len(train_texts)}, 验证集大小: {len(val_texts)}")
except Exception as e:
    print("数据划分时出错:", e)
    traceback.print_exc()
    sys.exit(1)

# --------------------------------------------------
class FinDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts, self.labels = texts, labels
        self.tok, self.max_len  = tokenizer, max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        try:
            enc = self.tok(
                self.texts[idx],
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            item = {k: v.squeeze(0) for k, v in enc.items()}
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item
        except Exception as e:
            print(f"处理样本 {idx} 时出错: {e}")
            print(f"文本内容: {self.texts[idx][:100]}...")
            raise e

def build_loader(texts, labels, batch_size=8):
    try:
        ds = FinDataset(texts, labels, tokenizer, max_len=args.max_len)
        return DataLoader(ds, batch_size=batch_size, shuffle=bool(labels))
    except Exception as e:
        print("构建DataLoader时出错:", e)
        traceback.print_exc()
        raise e

try:
    train_loader = build_loader(train_texts, train_labels, args.batch_size)
    val_loader   = build_loader(val_texts,   val_labels,   args.batch_size)
    print("DataLoader构建完成")
except Exception as e:
    print("构建DataLoader失败:", e)
    traceback.print_exc()
    sys.exit(1)

# --------------------------------------------------
try:
    # 设置设备
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"使用设备: {device}")
    
    # 修改为3分类模型 (0: 消极, 1: 中性, 2: 积极)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    model.to(device)
    print("模型加载完成")
    
    # 加载已有权重继续训练
    epoch_ckpt = "model/epoch_5.pt"
    if os.path.exists(epoch_ckpt):
        model.load_state_dict(torch.load(epoch_ckpt, map_location=device))
        print(f"已加载断点权重: {epoch_ckpt}")
        
except Exception as e:
    print("模型加载失败:", e)
    traceback.print_exc()
    sys.exit(1)

# 使用命令行参数
EPOCHS = args.epochs
PATIENCE = args.patience
MIN_DELTA = args.min_delta

try:
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader)*EPOCHS
    )
    print("优化器和调度器初始化完成")
except Exception as e:
    print("优化器初始化失败:", e)
    traceback.print_exc()
    sys.exit(1)

def train_epoch(loader, epoch_num):
    try:
        total_batches = len(loader)
        print(f"Epoch {epoch_num}/{EPOCHS} - 总batch数: {total_batches}")
        model.train()
        losses = []
        batch_count = 0
        
        for batch in tqdm(loader, desc=f"Epoch {epoch_num}/{EPOCHS}"):
            try:
                batch_count += 1
                if batch_count % 50 == 0:  # 每50个batch打印一次进度
                    print(f"Epoch {epoch_num}/{EPOCHS} - Batch {batch_count}/{total_batches}")
                
                batch = {k:v.to(device) for k,v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
                losses.append(loss.item())
                
                if batch_count % 50 == 0:
                    print(f"Epoch {epoch_num}/{EPOCHS} - Batch {batch_count}/{total_batches} - Loss: {loss.item():.4f}")
                
                # 清理内存
                del outputs, loss
                
            except Exception as e:
                print(f"Epoch {epoch_num}/{EPOCHS} - Batch {batch_count}/{total_batches} 训练出错: {e}")
                traceback.print_exc()
                continue
                
        return np.mean(losses) if losses else 0
    except Exception as e:
        print(f"Epoch {epoch_num}/{EPOCHS} 整体训练出错: {e}")
        traceback.print_exc()
        return 0

def eval_epoch(loader, epoch_num):
    try:
        model.eval()
        preds, golds = [], []
        with torch.no_grad():
            for batch in loader:
                try:
                    batch = {k:v.to(device) for k,v in batch.items()}
                    logits = model(**batch).logits
                    preds.extend(logits.argmax(-1).cpu().numpy())
                    golds.extend(batch['labels'].cpu().numpy())
                    
                    # 清理内存
                    del logits
                        
                except Exception as e:
                    print(f"Epoch {epoch_num}/{EPOCHS} 验证batch出错: {e}")
                    traceback.print_exc()
                    continue
        return accuracy_score(golds, preds), f1_score(golds, preds, average='weighted')
    except Exception as e:
        print(f"Epoch {epoch_num}/{EPOCHS} 验证出错: {e}")
        traceback.print_exc()
        return 0.0, 0.0

try:
    best_f1 = 0
    patience_counter = 0
    print(f"训练集样本数: {len(df_train)}")
    print(f"情感分布: {df_train['sentiment'].value_counts().to_dict()}")
    print(f"训练参数:")
    print(f"  - 模型: {MODEL_NAME}")
    print(f"  - 批次大小: {args.batch_size}")
    print(f"  - 最大长度: {args.max_len}")
    print(f"  - 学习率: {args.lr}")
    print(f"  - 训练轮数: {EPOCHS}")
    print(f"  - 早停耐心值: {PATIENCE}")
    print(f"  - 最小提升阈值: {MIN_DELTA}")
    print(f"  - 设备: {device}")
    print(f"  - 情感映射: -1(消极)->0, 0(中性)->1, 1(积极)->2")

    # 新增：确保model文件夹存在
    os.makedirs('model', exist_ok=True)

    for epoch in range(EPOCHS):
        try:
            epoch_num = epoch + 1
            print(f"\n{'='*50}")
            print(f"开始训练 Epoch {epoch_num}/{EPOCHS}")
            print(f"{'='*50}")
            
            train_loss = train_epoch(train_loader, epoch_num)
            acc, f1 = eval_epoch(val_loader, epoch_num)
            
            print(f"\nEpoch {epoch_num}/{EPOCHS} 结果:")
            print(f"  - 训练损失: {train_loss:.4f}")
            print(f"  - 验证准确率: {acc:.4f}")
            print(f"  - 验证F1: {f1:.4f}")
            
            # 每10个epoch存储一次模型
            if epoch_num % 5 == 0:
                model_path = f"model/epoch_{epoch_num}.pt"
                torch.save(model.state_dict(), model_path)
                print(f"  - 已保存模型到: {model_path}")

            # 早停检查
            if f1 > best_f1 + MIN_DELTA:
                best_f1 = f1
                patience_counter = 0
                torch.save(model.state_dict(), "./best.pt")
                print(f"  - 保存最佳模型，F1={f1:.4f}")
            else:
                patience_counter += 1
                print(f"  - F1未提升，patience_counter={patience_counter}/{PATIENCE}")
            
            # 早停
            if patience_counter >= PATIENCE:
                print(f"\n早停触发！连续{PATIENCE}个epoch F1未提升，停止训练")
                break
                
        except Exception as e:
            print(f"Epoch {epoch_num}/{EPOCHS} 训练出错: {e}")
            traceback.print_exc()
            continue

    print(f"\n{'='*50}")
    print(f"训练完成！")
    print(f"最佳F1: {best_f1:.4f}")
    print(f"最佳模型已保存为: best.pt")
    print(f"{'='*50}")
    
except Exception as e:
    print("训练过程中出现严重错误:", e)
    traceback.print_exc()
    sys.exit(1) 