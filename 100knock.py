# ここからもらいました
# https://github.com/noguhiro2002/100knocks-preprocess_ForColab-AzureNotebook/blob/master/preprocess_knock_Python_Colab.ipynb

# %%
import os
import pandas as pd
from pandas.core.reshape.concat import concat
import numpy as np
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import math
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

# %%
df_customer = pd.read_csv(
    'https://raw.githubusercontent.com/The-Japan-DataScientist-Society/100knocks-preprocess/master/docker/work/data/customer.csv')
df_category = pd.read_csv(
    'https://raw.githubusercontent.com/The-Japan-DataScientist-Society/100knocks-preprocess/master/docker/work/data/category.csv')
df_product = pd.read_csv(
    'https://raw.githubusercontent.com/The-Japan-DataScientist-Society/100knocks-preprocess/master/docker/work/data/product.csv')
df_receipt = pd.read_csv(
    'https://raw.githubusercontent.com/The-Japan-DataScientist-Society/100knocks-preprocess/master/docker/work/data/receipt.csv')
df_store = pd.read_csv(
    'https://raw.githubusercontent.com/The-Japan-DataScientist-Society/100knocks-preprocess/master/docker/work/data/store.csv')
df_geocode = pd.read_csv(
    'https://raw.githubusercontent.com/noguhiro2002/100knocks-preprocess_ForColab-AzureNotebook/master/data/geocode.csv')

# %%
# P-001: レシート明細のデータフレーム（df_receipt）から全項目の先頭10件を表示し、どのようなデータを保有しているか目視で確認せよ。
df_receipt.head(10)

# %%
# P-002: レシート明細のデータフレーム（df_receipt）から売上日（sales_ymd）、顧客ID（customer_id）、商品コード（product_cd）、売上金額（amount）の順に列を指定し、10件表示させよ。

listP002 = ["sales_ymd", "customer_id", "product_cd", "amount"]
df_receipt[listP002].head(10)

# %%
# P-003: レシート明細のデータフレーム（df_receipt）から売上日（sales_ymd）、顧客ID（customer_id）、商品コード（product_cd）、売上金額（amount）の順に列を指定し、10件表示させよ。ただし、sales_ymdはsales_dateに項目名を変更しながら抽出すること。
df_receipt = df_receipt\
    .rename(columns={"sales_ymd": "sales_date"})
listP002 = ["sales_date", "customer_id", "product_cd", "amount"]

df_receipt[listP002].head(10)

# %%
# P-004: レシート明細のデータフレーム（df_receipt）から売上日（sales_ymd）、顧客ID（customer_id）、商品コード（product_cd）、売上金額（amount）の順に列を指定し、以下の条件を満たすデータを抽出せよ。
df_receipt = df_receipt.rename(columns={"sales_date": "sales_ymd"})
listP002 = ["sales_ymd", "customer_id", "product_cd", "amount"]

#df_receipt[df_receipt.customer_id == "CS018205000001"][listP002]

df_receipt[listP002].query('customer_id == "CS018205000001"')
# %%
#P-005: レシート明細のデータフレーム（df_receipt）から売上日（sales_ymd）、顧客ID（customer_id）、商品コード（product_cd）、売上金額（amount）の順に列を指定し、以下の条件を満たすデータを抽出せよ。
df_receipt[listP002].\
    query('customer_id == "CS018205000001" and amount >= 1000')

# %%
#P-006: レシート明細データフレーム「df_receipt」から売上日（sales_ymd）、顧客ID（customer_id）、商品コード（product_cd）、売上数量（quantity）、売上金額（amount）の順に列を指定し、以下の条件を満たすデータを抽出せよ。
#顧客ID（customer_id）が"CS018205000001"
#売上金額（amount）が1, 000以上または売上数量（quantity）が5以上

listP006 = ["sales_ymd", "customer_id", "product_cd","quantity","amount"]

df_receipt[listP006].\
    query('customer_id == "CS018205000001" and (amount >= 1000 or quantity >= 5)')

# %%
#P-007: レシート明細のデータフレーム（df_receipt）から売上日（sales_ymd）、顧客ID（customer_id）、商品コード（product_cd）、売上金額（amount）の順に列を指定し、以下の条件を満たすデータを抽出せよ。
#顧客ID（customer_id）が"CS018205000001"
#売上金額（amount）が1, 000以上2, 000以下
df_receipt[listP002]\
    .query('customer_id == "CS018205000001"')\
        .query('amount >= 1000 and amount <= 2000')
# %%
#P-008: レシート明細のデータフレーム（df_receipt）から売上日（sales_ymd）、顧客ID（customer_id）、商品コード（product_cd）、売上金額（amount）の順に列を指定し、以下の条件を満たすデータを抽出せよ。
#顧客ID（customer_id）が"CS018205000001"
#商品コード（product_cd）が"P071401019"以外
df_receipt[listP002].\
    query('customer_id == "CS018205000001"')\
        .query('product_cd != "P071401019"')

# %%
#P-009: 以下の処理において、出力結果を変えずにORをANDに書き換えよ。
#df_store.query('not(prefecture_cd == "13" | floor_area > 900)')
df_store.query('not(prefecture_cd == "13") and not(floor_area > 900)')

# %%
#解き直し2021年5月28日
#P-010: 店舗データフレーム（df_store）から、店舗コード（store_cd）が"S14"で始まるものだけ全項目抽出し、10件だけ表示せよ。
df_store.query('store_cd.str.startswith("S14")',engine='python').head(10)

# %%
#P-011:
df_customer.query('customer_id.str.endswith("1")',engine='python').head(10)
# %%
#P-012:
df_store.query('address.str.contains("横浜市")', engine='python')

#%%
#解き直し5月29日
#P-013:
df_customer.query("status_cd.str.contains('^[A-F]')", engine='python').head(10)
#%%
#解き直し5月29日
#P-014:
df_customer.query('status_cd.str.contains("[1-9]$")',engine='python').head(10)

# %%
#解き直し5月29日
#P-015:
#df_customer.query('status_cd.str.contains("^[A-F]") and status_cd.str.contains("[1-9]$")',engine='python').head(10)
df_customer.query(
    'status_cd.str.contains("^[A-F].*[1-9]$")', engine='python').head(10)

# %%
#P-016:
df_store.query('tel_no.str.contains("^...[-]...[-]....$")',engine='python')

# %%
#P-017:
df_customer.sort_values('birth_day').head(10)
# %%
#P-018:
df_customer.sort_values('birth_day',ascending= False).head(10)
# %%
#解き直し2021年5月28日
#P-019:
ans = df_receipt[["customer_id", "amount"]].sort_values('amount',ascending=False)

rank = pd.RangeIndex(start=1, stop=len(ans) + 1, step=1)
ans["rank"]=rank

print(ans.head(10))

"""公式回答
df_tmp = pd.concat([df_receipt[['customer_id', 'amount']],\
    df_receipt['amount'].rank(method='min', ascending=False)], axis=1)

df_tmp.columns = ['customer_id', 'amount', 'ranking']
df_tmp.sort_values("ranking", ascending=True).head(10)
"""
#%%
#P-020:
ans = df_receipt[["customer_id", "amount"]].sort_values('amount', ascending=False)

rank = pd.RangeIndex(start=1, stop=len(ans) + 1, step=1)
ans["rank"] = rank

print(ans.head(10))

"""公式回答
df_tmp = pd.concat([df_receipt[['customer_id', 'amount']],\
    df_receipt['amount'].rank(method='first', ascending=False)], axis=1)

df_tmp.columns = ['customer_id', 'amount', 'ranking']
df_tmp.sort_values("ranking", ascending=True).head(10)
"""
# %%
#P-021:
print(len(df_receipt))

# %%
#P-022:
print(len(df_receipt['customer_id'].unique()))

# %%
#P-023:
ans = df_receipt.groupby("store_cd")
print(ans[["amount","quantity"]].sum())

# %%
#解き直し2021年5月28日
#P-024:
ans = df_receipt.groupby("customer_id")
print(ans.sales_ymd.max().head(10))

#模範回答
#df_receipt.groupby('customer_id').sales_ymd.max().reset_index().head(10)

# %%
#P-025:
df_receipt.groupby("customer_id").sales_ymd.min().head(10)

# %%
#P-026:
atarashi = df_receipt.groupby("customer_id").sales_ymd.max()
furui = df_receipt.groupby("customer_id").sales_ymd.min()

df_tmp = pd.concat([atarashi,furui],axis=1)
df_tmp.columns = ['latest', 'oldest']

df_tmp[df_tmp.latest != df_tmp.oldest].head(10)

# %%
#P-027:
df_receipt.groupby("store_cd").amount.mean()\
    .sort_values(ascending=False).head(5)

# %%
#P-028:
df_receipt.groupby("store_cd").amount.median().sort_values(ascending=False).head(5)
# %%
#P-029:
df_receipt.groupby("store_cd")['product_cd'].apply(lambda x:x.mode())

# %%
#5月30日解き直し
#P-030:
df_receipt.groupby("store_cd").amount.var(ddof=0).sort_values(ascending=False).head(5)

# %%
#5月30日解き直し
#P-031:
df_receipt.groupby("store_cd").amount.std(ddof=0).sort_values(ascending=False).head(5)
# %%
#P-032:
df_receipt.amount.quantile([0,0.25,0.5,0.75,1])

# %%
#P-033:
df_receipt.groupby("store_cd").amount.mean()\
    [df_receipt.groupby("store_cd").amount.mean() >= 330]

#df_receipt.groupby("store_cd").amount.mean().reset_index().query('amount>= 330')

# %%
#5月30日解き直し
#P-034:
a = df_receipt.query('customer_id.str.contains("^[A-Y]")', engine='python')
b = sum(a["amount"])
c = len(a["customer_id"].unique())

print(b/c)
# %%
