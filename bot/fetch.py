import httpx
import datetime
import asyncio
import csv
import os
import re
from bot.items_classify import get_commodity

# The URL is the API endpoint for the live feed.
API_URL = "https://zhibo.sina.com.cn/api/zhibo/feed"

async def fetch_7x24(limit=100):
    """
    Fetches the latest news flashes from Sina Finance 7x24 using its API.
    It can now fetch more than one page of results to meet the limit.

    Args:
        limit (int): The total number of news items to fetch across multiple pages.

    Returns:
        list: A list of dictionaries, where each dictionary represents a news item.
              Returns an empty list if an error occurs.
    """
    items = []
    page = 1
    # Set a fixed page size for each request.
    page_size = 50

    # Use httpx.AsyncClient for asynchronous HTTP requests.
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://finance.sina.com.cn/7x24/?tag=5"
    }

    async with httpx.AsyncClient(timeout=15, headers=headers) as client:
        # --- Loop to fetch multiple pages ---
        # Continue fetching as long as we haven't reached the desired limit.
        while len(items) < limit:
            params = {
                "zhibo_id": "152",
                "page": page,
                "page_size": page_size,
                "type": 0,
            }
            
            try:
                # Send a GET request to the API endpoint for the current page.
                print(f"Fetching page {page}...")
                r = await client.get(API_URL, params=params)
                r.raise_for_status()
                data = r.json()
            except (httpx.RequestError, ValueError) as e:
                print(f"An error occurred while fetching page {page}: {e}")
                # Stop trying if a request fails.
                break

            try:
                news_list = data["result"]["data"]["feed"]["list"]
                # If the API returns an empty list, there's no more news to fetch.
                if not news_list:
                    print("No more news to fetch.")
                    break 
            except KeyError:
                print("Could not find news list in API response. Structure may have changed.")
                break

            for item in news_list:
                try:
                    title = item.get("rich_text", "No title available")
                    time_str = item["create_time"]
                    dt = datetime.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')

                    # Extract title and raw according to the new rule
                    match = re.match(r"[【\[]([^】\]]+)[】\]](.*)", title)
                    if match:
                        extracted_title = match.group(1).strip()
                        raw = match.group(2).strip()
                    else:
                        extracted_title = ""
                        raw = title.strip()

                    commodity_type = get_commodity(raw)
                    items.append({"dt": dt, "title": extracted_title, "raw": raw, "commodity_type": commodity_type})
                except (KeyError, ValueError, TypeError) as e:
                    print(f"Skipping item due to parsing error: {e}. Item data: {item}")
                    continue
            
            # Increment the page number for the next request.
            page += 1

    return items[:limit], save_news_to_csv(items)

def save_news_to_csv(news_items, filename=None):
    """
    保存新闻数据到data目录下的csv文件。
    Args:
        news_items (list): 由fetch_7x24返回的新闻字典列表。
        filename (str): 可选，文件名，默认为'sina_7x24_<日期时间>.csv'
    Returns:
        str: 保存的文件路径，如果保存失败则返回None
    """
    try:
        # 获取当前文件所在目录的上级目录（项目根目录）
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        data_dir = os.path.join(project_root, "data")
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        if filename is None:
            # now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sina_raw.csv"
        
        filepath = os.path.join(data_dir, filename)
        
        with open(filepath, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["dt", "title", "raw", "commodity_type"])
            for item in news_items:
                try:
                    writer.writerow([
                        item["dt"].strftime("%Y-%m-%d %H:%M:%S") if isinstance(item["dt"], datetime.datetime) else item["dt"],
                        item["title"],
                        item["raw"],
                        item["commodity_type"]
                    ])
                except (KeyError, TypeError) as e:
                    print(f"跳过无效的新闻项: {e}")
                    continue
        
        print(f"已保存 {len(news_items)} 条新闻到 {filepath}")
        return filepath
        
    except FileNotFoundError as e:
        print(f"文件路径错误: {e}")
        return None
    except PermissionError as e:
        print(f"权限错误，无法写入文件: {e}")
        return None
    except Exception as e:
        print(f"保存文件时发生未知错误: {e}")
        return None


def test_save_news_to_csv():
    """
    测试save_news_to_csv函数的功能
    """
    print("开始测试 save_news_to_csv 函数...")
    
    # 测试数据
    test_news_items = [
        {
            "dt": datetime.datetime.now(),
            "title": "测试标题1",
            "raw": "测试原始内容1",
            "commodity_type": "黄金"
        },
        {
            "dt": datetime.datetime.now(),
            "title": "测试标题2", 
            "raw": "测试原始内容2",
            "commodity_type": "原油"
        }
    ]
    
    # 测试1: 正常保存
    print("\n测试1: 正常保存")
    result = save_news_to_csv(test_news_items, "test_news.csv")
    if result:
        print(f"✓ 测试1成功，文件保存到: {result}")
    else:
        print("✗ 测试1失败")
    
    # 测试2: 使用默认文件名
    print("\n测试2: 使用默认文件名")
    result = save_news_to_csv(test_news_items)
    if result:
        print(f"✓ 测试2成功，文件保存到: {result}")
    else:
        print("✗ 测试2失败")
    
    # 测试3: 空数据列表
    print("\n测试3: 空数据列表")
    result = save_news_to_csv([])
    if result:
        print(f"✓ 测试3成功，文件保存到: {result}")
    else:
        print("✗ 测试3失败")
    
    # 测试4: 无效数据
    print("\n测试4: 无效数据")
    invalid_news_items = [
        {
            "dt": "invalid_date",
            "title": "测试标题",
            "raw": "测试内容",
            "commodity_type": "黄金"
        }
    ]
    result = save_news_to_csv(invalid_news_items, "test_invalid.csv")
    if result:
        print(f"✓ 测试4成功，文件保存到: {result}")
    else:
        print("✗ 测试4失败")
    
    print("\n测试完成！")


# # --- Test Example Usage ---
# async def main():
#     # Now we can fetch a larger number of items, e.g., 120.
#     number_to_fetch = 1200
#     print(f"Fetching latest {number_to_fetch} news items from Sina Finance...")
#     news_items = await fetch_7x24(limit=number_to_fetch)
    
#     if news_items:
#         save_news_to_csv(news_items)
#     else:
#         print("Failed to fetch news.")

# if __name__ == "__main__":
#     # 运行测试
#     test_save_news_to_csv()
#     # asyncio.run(main())
