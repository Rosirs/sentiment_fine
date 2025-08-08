import pandas as pd
import jieba
import re
import math
import os

def load_sentiment_dict(dictionary_path):
    """
    Loads the sentiment dictionary from a CSV file, handling potential encoding issues.
    """
    try:
        # First, try reading with 'utf-8-sig', which handles BOM characters often added by Windows.
        sentiment_df = pd.read_csv(dictionary_path, encoding='utf-8-sig')
        positive_words = set(sentiment_df[sentiment_df['Sentiment'] == 'Positive']['Term'])
        negative_words = set(sentiment_df[sentiment_df['Sentiment'] == 'Negative']['Term'])
        print(positive_words)
        return positive_words, negative_words
    except UnicodeDecodeError:
        try:
            # If 'utf-8-sig' fails, try 'gbk' as a fallback, common for Chinese text on Windows.
            sentiment_df = pd.read_csv(dictionary_path, encoding='gbk')
            positive_words = set(sentiment_df[sentiment_df['Sentiment'] == 'Positive']['Term'])
            negative_words = set(sentiment_df[sentiment_df['Sentiment'] == 'Negative']['Term'])
            # print(positive_words)
            return positive_words, negative_words
        except Exception as e:
            raise RuntimeError(f"Failed to load sentiment dictionary with both utf-8-sig and gbk encodings: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load sentiment dictionary: {e}")

def calculate_sentiment_score(text, positive_words, negative_words):
    """
    Calculates the sentiment score by counting keywords directly in the text.
    The final score is normalized to the [-1, 1] range using the tanh function.
    This method does not require any external NLP libraries.
    """
    score = 0
    # Count occurrences of each positive and negative term in the text
    for word in positive_words:
        score += text.count(word)
    for word in negative_words:
        score -= text.count(word)
    
    # Normalize the raw score to the (-1, 1) range using the hyperbolic tangent function.
    # This handles cases with many keywords gracefully, preventing scores from growing indefinitely.
    normalized_score = math.tanh(score)
    return normalized_score

def analyze_sentiment_from_csv(csv_path, output_path=None, dict_path='../data/sentiment.csv'):
    """
    Reads a news CSV file, performs sentiment analysis, and saves the results.
    """
    if output_path is None:
        base, _ = os.path.splitext(csv_path)
        output_path = f"{base}_result.csv"

    # 1. Load the sentiment dictionary
    positive_words, negative_words = load_sentiment_dict(dict_path)
    
    # 2. Read the news data
    df = pd.read_csv(csv_path)

    # 3. Clean and prepare the text data
    df['dt'] = pd.to_datetime(df.get('dt'), errors='coerce').dt.date
    df['title'] = df.get('title', '').fillna('')
    df['raw'] = df.get('raw', '').fillna('')
    df['text'] = df['title'] + '。' + df['raw']

    # 4. Calculate sentiment score for each news item
    df['sentiment_score'] = df['text'].apply(
        lambda text: calculate_sentiment_score(text, positive_words, negative_words)
    )

    # 5. Determine the sentiment label based on the score
    def get_sentiment_label(score, threshold=0.15):
        """Classifies score into Positive, Negative, or Neutral."""
        if score > threshold:
            return 'Positive'
        elif score < -threshold:
            return 'Negative'
        else:
            return 'Neutral'

    df['sentiment_label'] = df['sentiment_score'].apply(get_sentiment_label)

    # 6. Prepare and save the final output
    output_df = df[['dt', 'title', 'sentiment_score', 'sentiment_label']]
    output_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\nAnalysis complete. Results saved to: {output_path}")
    return output_df


if __name__ == '__main__':
    # Generate the necessary data files first
    # Run the analysis on the generated sample file
    print("\n--- Starting Analysis on Sample News File ---")
    result_df = analyze_sentiment_from_csv('../data/sina_raw.csv')

    # Display the results in a formatted way
    print("\n--- Analysis Results Preview ---")
    result_df['sentiment_score'] = result_df['sentiment_score'].apply(lambda x: f"{x:.6f}")
    # print(result_df.to_string(index=False))