import asyncio, pandas as pd, datetime, argparse, os
from bot.fetch import fetch_7x24
from bot.nlp import analyze
import logging, sys
import threading

# 让 print 立即刷新，防止“卡住”
sys.stdout.reconfigure(line_buffering=True)

# 全局日志 + 时间戳
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

FETCH_INTERVAL = 300  # 5分钟

def start_background_fetch(csv_path, limit=100, fetch_only=False):
    import time
    from dashboard import CSV_FILE
    def fetch_loop():
        while True:
            try:
                print("[后台] 正在抓取最新新闻...")
                items, new_csv_path = asyncio.run(fetch_7x24(limit))
                if not items:
                    print("[后台] 未获取到新闻。")
                    time.sleep(FETCH_INTERVAL)
                    continue
                
                if fetch_only:
                    # 仅抓取模式：fetch_7x24已经保存了文件，直接追加到主文件
                    df_new = pd.read_csv(new_csv_path)
                    df_new.to_csv(csv_path, mode='a', header=False, index=False)
                    print(f"[后台] 已追加 {len(df_new)} 条原始新闻。")
                else:
                    # 读取新抓取的CSV文件并使用FinBERT进行情感分析
                    df_new = pd.read_csv(new_csv_path)
                    df_new = analyze(df_new, text_column='text', max_length=512)
                    
                    # 合并文本列（title + raw）
                    df_new['text'] = df_new['title'].fillna('') + '。' + df_new['raw'].fillna('')
                    
                    # 追加到主文件
                    df_new.to_csv(csv_path, mode='a', header=False, index=False)
                    print(f"[后台] 已追加 {len(df_new)} 条新闻。")
            except Exception as e:
                print(f"[后台] 抓取异常: {e}")
            time.sleep(FETCH_INTERVAL)
    t = threading.Thread(target=fetch_loop, daemon=True)
    t.start()

async def run(limit=100, plot=False, fetch_only=False):
    items, csv_path = await fetch_7x24(limit)
    logging.info(f"实际抓取到 {len(items)} 条新闻")
    if not items:
        logging.warning("抓到的列表为空，请检查 fetch.py！")
        return # 如果没有抓取到数据，则提前退出
    
    if fetch_only:
        # 只抓取新闻，不进行情感分析
        logging.info("仅抓取模式：跳过情感分析")
        print(f"✅ 新闻数据已保存到 {csv_path} 共 {len(items)} 条")
        return
    
    # 读取抓取的CSV文件并使用FinBERT进行情感分析
    df = pd.read_csv(csv_path)
    
    # 合并文本列（title + raw）用于情感分析
    df['text'] = df['title'].fillna('') + '。' + df['raw'].fillna('')
    df.drop(columns=['title', 'raw'], inplace=True)
    
    # 使用FinBERT进行情感分析
    try:
        logging.info("开始FinBERT情感分析...")
        df = analyze(df, text_column='text', max_length=512)
        logging.info("FinBERT情感分析完成")
    except Exception as e:
        logging.error(f"FinBERT情感分析失败: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return
    
    # 保存带有情感分析结果的文件（覆盖原文件）
    data_path = os.path.join(data_dir, "sina_7x24_latest.xlsx")
    df.to_excel(data_path, index=False)
    print(f"✅ 已更新 {data_path} 共 {len(df)} 条")

    if plot:
        try:
            import plotly.express as px
            fig = px.scatter(
                df,
                x="dt",
                y="finbert_sentiment_score",
                color="commodity_type",
                hover_data=["title", "raw"],
                title="新浪财经 7×24 FinBERT情绪分布",
                opacity=0.8,
                color_discrete_sequence=px.colors.qualitative.Set2,
                size_max=12
            )
            fig.update_traces(marker=dict(size=12, line=dict(width=1, color='white')))
            fig.update_layout(
                title_font_size=22,
                plot_bgcolor="white",
                paper_bgcolor="white",
                legend_title_text="品种",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=40, t=60, b=40)
            )
            html_path = os.path.join(data_dir, "sina_7x24_latest.html")
            fig.write_html(html_path)
            print(f"📊 图表已更新 {html_path}")
        except ImportError:
            logging.warning("缺少 plotly 库，无法生成图表。请运行 'pip install plotly' 安装。")
        except Exception as e:
            logging.error(f"生成图表时出错: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--limit", type=int, default=100, help="抓取条数")
    parser.add_argument("--plot", action="store_true", help="生成交互图表")
    parser.add_argument("--dashboard", action="store_true", help="启动可视化看板（含词云）")
    parser.add_argument("-f", "--fetch-only", action="store_true", help="仅抓取新闻，不进行情感分析")
    args = parser.parse_args()
    if args.dashboard:
        from dashboard import run_dashboard
        # 先抓取数据再传递给 dashboard
        async def fetch_and_dashboard():
            items, csv_path = await fetch_7x24(args.limit)
            logging.info(f"实际抓取到 {len(items)} 条新闻")
            if not items:
                logging.warning("抓到的列表为空，请检查 fetch.py！")
                return
            
            if args.fetch_only:
                # 仅抓取模式：跳过情感分析
                logging.info("仅抓取模式：跳过情感分析")
                print(f"✅ 新闻数据已保存到 {csv_path} 共 {len(items)} 条")
                # 启动后台定时抓取（仅抓取模式）
                start_background_fetch(csv_path, args.limit, args.fetch_only)
                run_dashboard(csv_path)
                return
            
            # 读取抓取的CSV文件并使用FinBERT进行情感分析
            df = pd.read_csv(csv_path)
            
            # 合并文本列（title + raw）用于情感分析
            df['text'] = df['title'].fillna('') + '。' + df['raw'].fillna('')
            df.drop(columns=['title', 'raw'], inplace=True)
            
            # 使用FinBERT进行情感分析
            try:
                logging.info("开始FinBERT情感分析...")
                df = analyze(df, text_column='text', max_length=512)
                logging.info("FinBERT情感分析完成")
            except Exception as e:
                logging.error(f"FinBERT情感分析失败: {e}")
                import traceback
                logging.error(traceback.format_exc())
                return
            
            data_path = os.path.join(data_dir, "sina_7x24_latest.xlsx")
            df.to_excel(data_path, index=False)
            print(f"✅ 已更新 {data_path} 共 {len(df)} 条")
            # 启动后台定时抓取
            start_background_fetch(data_path, args.limit, args.fetch_only)
            run_dashboard(data_path)
        asyncio.run(fetch_and_dashboard())
    else:
        asyncio.run(run(args.limit, args.plot, args.fetch_only))
    # 调用情感分析包示例
    # sentiment_analysis_from_csv(csv_path)
