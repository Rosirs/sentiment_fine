import pandas as pd
import jieba
import re
import os

def load_sentiment_dict(dictionary_path):
    try:
        # 读取CSV格式的情感词典
        sentiment_df = pd.read_csv(dictionary_path)
        positive_words = set(sentiment_df[sentiment_df['Sentiment'] == 'Positive']['Term'])
        negative_words = set(sentiment_df[sentiment_df['Sentiment'] == 'Negative']['Term'])
        return positive_words, negative_words
    except Exception as e:
        raise RuntimeError(f"加载情感词典失败: {e}")

negation_words = set([
    '不', '没', '并非', '不是', '毫无', '未', '无'
])
degree_adverbs = {
    '非常': 2.0, '极其': 2.0, '极度': 2.0, '最': 2.0, '显著': 2.0, '重大': 2.0,
    '很': 1.5, '颇': 1.5, '相当': 1.5,
    '比较': 1.2, '较为': 1.2, '还算': 1.2,
    '有点': 0.8, '稍微': 0.8, '略微': 0.8,
    '不太': 0.5, '不大': 0.5, '不够': 0.5,
    '极其': 2.0
}

def calculate_sentiment_score(words, positive_words, negative_words):
    score = 0
    for i, word in enumerate(words):
        if word in positive_words or word in negative_words:
            base_score = 1 if word in positive_words else -1
            weight = 1
            if i > 0:
                prev_word = words[i-1]
                if prev_word in degree_adverbs:
                    weight *= degree_adverbs[prev_word]
                elif prev_word in negation_words:
                    weight *= -1
            if i > 1:
                prev_word_2 = words[i-2]
                if words[i-1] in degree_adverbs and prev_word_2 in negation_words:
                    weight *= -1
            score += base_score * weight
    return score

def sentiment_analysis_from_csv(csv_path, output_path=None, dict_path=None):
    """
    读取csv文件，计算情感分数，输出Excel。
    :param csv_path: 输入csv路径
    :param output_path: 输出Excel路径，默认与输入同名
    :param dict_path: 情感词典路径，默认'../data/sentiment.csv'
    :return: DataFrame
    """
    if dict_path is None:
        dict_path = os.path.join(os.path.dirname(__file__), '../data/sentiment.csv')
    positive_words, negative_words = load_sentiment_dict(dict_path)
    df = pd.read_csv(csv_path)
    if 'dt' in df.columns:
        df['dt'] = pd.to_datetime(df['dt'], errors='coerce')
        df.dropna(subset=['dt'], inplace=True)
    if 'title' in df.columns:
        df['title'] = df['title'].fillna('')
    if 'raw' in df.columns:
        df['raw'] = df['raw'].fillna('')
    df['text'] = df.get('title', '') + '。' + df.get('raw', '')
    jieba.initialize()
    df['cut_words'] = df['text'].astype(str).apply(jieba.lcut)
    df['sentiment_score'] = df['cut_words'].apply(lambda ws: calculate_sentiment_score(ws, positive_words, negative_words))
    if output_path is None:
        output_path = os.path.splitext(csv_path)[0] + '_sentiment.xlsx'
    # df.to_excel(output_path, index=False)
    return df

# if __name__ == '__main__':
#     # 示例用法
#     csv_path = '../data/sina_7x24_20250722_162900.csv'  # 可根据实际情况修改
#     sentiment_analysis_from_csv(csv_path)