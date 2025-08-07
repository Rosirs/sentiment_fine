import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

# 全局变量存储模型和tokenizer
_model = None
_tokenizer = None

def load_finbert_model():
    """加载FinBERT模型和tokenizer"""
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        try:
            logging.info("正在加载FinBERT模型...")
            _tokenizer = AutoTokenizer.from_pretrained("bardsai/finance-sentiment-zh-base")
            _model = AutoModelForSequenceClassification.from_pretrained("bardsai/finance-sentiment-zh-base")
            logging.info("FinBERT模型加载完成")
        except Exception as e:
            logging.error(f"加载FinBERT模型失败: {e}")
            raise RuntimeError(f"加载FinBERT模型失败: {e}")
    return _model, _tokenizer

def analyze_sentiment_finbert(text, max_length=512):
    """
    使用FinBERT分析文本情感
    :param text: 输入文本
    :param max_length: 最大文本长度
    :return: 情感分数 (-1到1之间，负值表示负面，正值表示正面)
    """
    if not text or pd.isna(text):
        return 0.0
    
    model, tokenizer = load_finbert_model()
    
    try:
        # 对文本进行编码
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            max_length=max_length, 
            truncation=True, 
            padding=True
        )
        
        # 进行预测
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
        
        # FinBERT的标签: 0=negative, 1=neutral, 2=positive
        # 计算情感分数: (positive_prob - negative_prob) / (positive_prob + negative_prob)
        negative_prob = probabilities[0][0].item()
        neutral_prob = probabilities[0][1].item()
        positive_prob = probabilities[0][2].item()
        
        # 计算加权情感分数
        sentiment_score = (positive_prob - negative_prob) / (positive_prob + negative_prob + neutral_prob)
        
        return sentiment_score
        
    except Exception as e:
        logging.error(f"FinBERT情感分析失败: {e}")
        return 0.0

def analyze_sentiment_batch_finbert(texts, max_length=512):
    """
    批量使用FinBERT分析文本情感
    :param texts: 文本列表
    :param max_length: 最大文本长度
    :return: 情感分数列表
    """
    logging.info(f"[DEBUG] analyze_sentiment_batch_finbert: 输入文本数量: {len(texts)}")
    if not texts:
        logging.info("[DEBUG] analyze_sentiment_batch_finbert: 输入为空，返回空列表")
        return []
    
    model, tokenizer = load_finbert_model()
    logging.info("[DEBUG] analyze_sentiment_batch_finbert: 模型加载完成")
    
    try:
        # 过滤空文本
        valid_texts = [str(text) for text in texts if text and not pd.isna(text)]
        logging.info(f"[DEBUG] analyze_sentiment_batch_finbert: 有效文本数量: {len(valid_texts)}")
        if not valid_texts:
            logging.info("[DEBUG] analyze_sentiment_batch_finbert: 有效文本为空，返回全0")
            return [0.0] * len(texts)
        
        # 批量编码
        inputs = tokenizer(
            valid_texts,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        )
        logging.info("[DEBUG] analyze_sentiment_batch_finbert: 编码完成")
        
        # 批量预测
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
        logging.info(f"[DEBUG] analyze_sentiment_batch_finbert: 模型输出维度: {probabilities.shape}")
        logging.info(f"[DEBUG] analyze_sentiment_batch_finbert: 前3条概率: {probabilities[:3]}")
        
        # 计算每个文本的情感分数
        sentiment_scores = []
        for i in range(len(valid_texts)):
            try:
                if probabilities.shape[1] == 3:
                    negative_prob = probabilities[i][0].item()
                    neutral_prob = probabilities[i][1].item()
                    positive_prob = probabilities[i][2].item()
                    sentiment_score = (positive_prob - negative_prob) / (positive_prob + negative_prob + neutral_prob)
                elif probabilities.shape[1] == 2:
                    negative_prob = probabilities[i][0].item()
                    positive_prob = probabilities[i][1].item()
                    sentiment_score = (positive_prob - negative_prob) / (positive_prob + negative_prob)
                else:
                    sentiment_score = probabilities[i][0].item() - 0.5
                sentiment_scores.append(round(sentiment_score, 4))
                logging.info(f"[DEBUG] 第{i}条: neg={negative_prob:.4f}, neu={neutral_prob if probabilities.shape[1]==3 else 'N/A'}, pos={positive_prob:.4f}, score={sentiment_scores[-1]}")
            except Exception as e:
                logging.error(f"处理第{i}个文本时出错: {e}")
                sentiment_scores.append(0.0)
        logging.info(f"[DEBUG] analyze_sentiment_batch_finbert: 情感分数计算完成")
        
        # 为原始列表中的空值填充0
        result = []
        valid_idx = 0
        for text in texts:
            if text and not pd.isna(text):
                result.append(sentiment_scores[valid_idx])
                valid_idx += 1
            else:
                result.append(0.0)
        logging.info(f"[DEBUG] analyze_sentiment_batch_finbert: 返回分数长度: {len(result)}")
        return result
        
    except Exception as e:
        logging.error(f"FinBERT批量情感分析失败: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return [0.0] * len(texts)


def analyze(df, text_column='text', max_length=512):
    """
    对DataFrame进行FinBERT情感分析
    :param df: 输入的DataFrame
    :param text_column: 文本列名
    :param max_length: 最大文本长度
    :return: 添加了情感分数的DataFrame
    """
    logging.info(f"[DEBUG] analyze: DataFrame shape: {df.shape}")
    if text_column not in df.columns:
        logging.warning(f"列 '{text_column}' 不存在于DataFrame中")
        return df
    
    logging.info(f"开始使用FinBERT分析 {len(df)} 条文本的情感...")
    
    # 获取文本列表
    texts = df[text_column].fillna('').astype(str).tolist()
    logging.info(f"[DEBUG] analyze: 文本列表长度: {len(texts)}")
    
    # 批量分析情感
    sentiment_scores = analyze_sentiment_batch_finbert(texts, max_length)
    logging.info(f"[DEBUG] analyze: 批量情感分析完成，分数样例: {sentiment_scores[:5]}")
    
    # 添加情感分数列
    df['finbert_sentiment_score'] = sentiment_scores
    
    # 添加情感标签
    df['finbert_sentiment_label'] = df['finbert_sentiment_score'].apply(
        lambda x: 'positive' if x > 0.1 else ('negative' if x < -0.1 else 'neutral')
    )
    logging.info(f"[DEBUG] analyze: 标签分布: {df['finbert_sentiment_label'].value_counts().to_dict()}")
    
    logging.info(f"FinBERT情感分析完成，平均分数: {np.mean(sentiment_scores):.3f}")
    
    return df

def get_sentiment_summary(df, sentiment_column='finbert_sentiment_score'):
    """
    获取情感分析摘要统计
    :param df: DataFrame
    :param sentiment_column: 情感分数列名
    :return: 摘要字典
    """
    if sentiment_column not in df.columns:
        return {}
    
    scores = df[sentiment_column].dropna()
    if len(scores) == 0:
        return {}
    
    summary = {
        'total_count': len(df),
        'valid_count': len(scores),
        'mean_score': float(scores.mean()),
        'std_score': float(scores.std()),
        'min_score': float(scores.min()),
        'max_score': float(scores.max()),
        'positive_count': len(scores[scores > 0.1]),
        'negative_count': len(scores[scores < -0.1]),
        'neutral_count': len(scores[(scores >= -0.1) & (scores <= 0.1)])
    }
    
    return summary

