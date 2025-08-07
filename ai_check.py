import os, json, re, html
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
client = OpenAI(
    api_key=os.getenv("MOONSHOT_API_KEY"),
    base_url="https://api.moonshot.cn/v1"
)

# -------------------- 1. 读数据 --------------------
df = pd.read_csv("./data/sina_raw.csv")
df['text'] = df['title'].fillna('') + '。' + df['raw'].fillna('')
df = df[['dt', 'text']]               # 只保留时间+合并后文本

texts = df['text'].astype(str).tolist()
# print(texts)

# -------------------- 2. Prompt --------------------
SYSTEM_PROMPT = (
    "你是量化金融分析师，分析新闻给出情绪判断。"
)

USER_PROMPT_TPL = (
    '对下列新闻做情感分析，返回JSON格式，每行一个对象：\n'
    '[{{"sentiment":"positive|negative|neutral","confidence":0.85,"impact_horizon":"short|medium|long","reason":""}}, ...]\n'
    '注意：confidence字段需要输出0.0-1.0之间的实际置信度数值，表示你对情感判断的确信程度。\n'
    '新闻列表（一行一条）：\n{}'
)

# -------------------- 3. 批量分析 --------------------
def analyze_sentiment_batch(texts: list[str], batch_size: 20) -> list[dict]:
    """
    分批调用 Kimi API，返回与 texts 顺序一致的 dict list
    """
    results = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    with tqdm(total=len(texts), desc="分析新闻情感", unit="条") as pbar:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            # 更新进度条描述
            pbar.set_description(f"处理第 {batch_num}/{total_batches} 批次")
            
            # 每一条占一行，避免模型串行
            prompt_text = [f"{idx}. {t}" for idx, t in enumerate(batch, 1)]

            try:
                resp = client.chat.completions.create(
                    model="moonshot-v1-8k",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": USER_PROMPT_TPL.format(prompt_text)}
                    ],
                    temperature=0.0,
                    max_tokens=1024
                )
                
                # 获取AI返回的内容
                ai_response = resp.choices[0].message.content.strip()
                print(f"AI返回内容: {ai_response[:200]}...")  # 打印前200个字符用于调试
                
                # 尝试解析JSON
                try:
                    batch_results = json.loads(ai_response)
                except json.JSONDecodeError as json_error:
                    print(f"JSON解析错误: {json_error}")
                    print(f"完整AI响应: {ai_response}")
                    # 如果JSON解析失败，尝试提取JSON部分
                    import re
                    json_match = re.search(r'\[.*\]', ai_response, re.DOTALL)
                    if json_match:
                        try:
                            batch_results = json.loads(json_match.group())
                        except:
                            raise Exception(f"无法解析JSON: {ai_response}")
                    else:
                        raise Exception(f"未找到JSON数组: {ai_response}")
                
                results.extend(batch_results)
                pbar.update(len(batch))  # 更新进度条
                
            except Exception as e:
                # 该批次全部置 error
                print(f"第 {batch_num} 批次出错，跳过：{e}")
                results.extend(
                    [{"sentiment": "error", "confidence": 0.0,
                      "impact_horizon": "", "reason": str(e)}] * len(batch)
                )
                pbar.update(len(batch))  # 即使出错也要更新进度条
    return results

# -------------------- 4. 调用并整合 --------------------
print(f"开始分析 {len(texts)} 条新闻...")
results = analyze_sentiment_batch(texts, batch_size=10)   # 可调 batch_size
print(f"分析完成，共处理 {len(results)} 条结果")

# 直接转成 DataFrame
senti_df = pd.DataFrame(results)

# 与原表横向合并
final_df = pd.concat([df.reset_index(drop=True), senti_df], axis=1)

# -------------------- 5. 保存 --------------------
print("正在保存结果...")

# 将DataFrame转换为JSON格式
results_json = final_df.to_dict('records')

# 保存为JSON文件
with open("./data/data_sentiment_kimi.json", "w", encoding="utf-8") as f:
    json.dump(results_json, f, ensure_ascii=False, indent=2)

print("已保存 → ./data/data_sentiment_kimi.json")
