import akshare as ak
import matplotlib.pyplot as plt

futures_index_ccidx_df = ak.futures_index_ccidx(symbol="中证商品期货指数")

plt.figure(figsize=(10, 5))
plt.plot(futures_index_ccidx_df['日期'], futures_index_ccidx_df['收盘点位'], label='close price')
plt.xlabel('date')
plt.ylabel('close price')
plt.title('ICC Index Close Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()