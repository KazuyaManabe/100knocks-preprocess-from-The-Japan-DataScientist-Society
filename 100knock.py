#%%
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
#from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler


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
# 2021年6月2日解きなおし
# P-003: レシート明細のデータフレーム（df_receipt）から売上日（sales_ymd）、顧客ID（customer_id）、商品コード（product_cd）、売上金額（amount）の順に列を指定し、10件表示させよ。ただし、sales_ymdはsales_dateに項目名を変更しながら抽出すること。
df_receipt = df_receipt.rename(columns={"sales_ymd": "sales_date"})
listP002 = ["sales_date", "customer_id", "product_cd", "amount"]

df_receipt[listP002].head(10)

# %%
# 2021年6月2日解きなおし
# P-004: レシート明細のデータフレーム（df_receipt）から売上日（sales_ymd）、顧客ID（customer_id）、商品コード（product_cd）、売上金額（amount）の順に列を指定し、以下の条件を満たすデータを抽出せよ。
df_receipt = df_receipt.rename(columns={"sales_date": "sales_ymd"})
listP002 = ["sales_ymd", "customer_id", "product_cd", "amount"]

#df_receipt[df_receipt.customer_id == "CS018205000001"][listP002]

df_receipt[listP002].query('customer_id == "CS018205000001"')
# %%
# P-005: レシート明細のデータフレーム（df_receipt）から売上日（sales_ymd）、顧客ID（customer_id）、商品コード（product_cd）、売上金額（amount）の順に列を指定し、以下の条件を満たすデータを抽出せよ。
df_receipt[listP002].query(
    'customer_id == "CS018205000001" and amount >= 1000')

# %%
# P-006: レシート明細データフレーム「df_receipt」から売上日（sales_ymd）、顧客ID（customer_id）、商品コード（product_cd）、売上数量（quantity）、売上金額（amount）の順に列を指定し、以下の条件を満たすデータを抽出せよ。
# 顧客ID（customer_id）が"CS018205000001"
# 売上金額（amount）が1, 000以上または売上数量（quantity）が5以上

listP006 = ["sales_ymd", "customer_id", "product_cd", "quantity", "amount"]

df_receipt[listP006].\
    query('customer_id == "CS018205000001" and (amount >= 1000 or quantity >= 5)')

# %%
# P-007: レシート明細のデータフレーム（df_receipt）から売上日（sales_ymd）、顧客ID（customer_id）、商品コード（product_cd）、売上金額（amount）の順に列を指定し、以下の条件を満たすデータを抽出せよ。
# 顧客ID（customer_id）が"CS018205000001"
# 売上金額（amount）が1, 000以上2, 000以下
df_receipt[listP002]\
    .query('customer_id == "CS018205000001"')\
    .query('amount >= 1000 and amount <= 2000')
# %%
# P-008: レシート明細のデータフレーム（df_receipt）から売上日（sales_ymd）、顧客ID（customer_id）、商品コード（product_cd）、売上金額（amount）の順に列を指定し、以下の条件を満たすデータを抽出せよ。
# 顧客ID（customer_id）が"CS018205000001"
# 商品コード（product_cd）が"P071401019"以外
df_receipt[listP002].\
    query('customer_id == "CS018205000001"')\
    .query('product_cd != "P071401019"')

# %%
# P-009: 以下の処理において、出力結果を変えずにORをANDに書き換えよ。
#df_store.query('not(prefecture_cd == "13" | floor_area > 900)')
df_store.query('not(prefecture_cd == "13") and not(floor_area > 900)')

# %%
# 解き直し2021年5月28日
# P-010: 店舗データフレーム（df_store）から、店舗コード（store_cd）が"S14"で始まるものだけ全項目抽出し、10件だけ表示せよ。
df_store.query('store_cd.str.startswith("S14")', engine='python').head(10)

# %%
# P-011:
df_customer.query('customer_id.str.endswith("1")', engine='python').head(10)
# %%
# P-012:
df_store.query('address.str.contains("横浜市")', engine='python')

# %%
# 解き直し5月29日
# P-013:
df_customer.query("status_cd.str.contains('^[A-F]')", engine='python').head(10)
# %%
# 解き直し5月29日
# P-014:
df_customer.query('status_cd.str.contains("[1-9]$")', engine='python').head(10)

# %%
# 解き直し5月29日
# P-015:
#df_customer.query('status_cd.str.contains("^[A-F]") and status_cd.str.contains("[1-9]$")',engine='python').head(10)
df_customer.query(
    'status_cd.str.contains("^[A-F].*[1-9]$")', engine='python').head(10)

# %%
# P-016:
df_store.query('tel_no.str.contains("^...[-]...[-]....$")', engine='python')

# %%
# P-017:
df_customer.sort_values('birth_day').head(10)
# %%
# P-018:
df_customer.sort_values('birth_day', ascending=False).head(10)
# %%
# 解き直し2021年5月28日
# P-019:
ans = df_receipt[["customer_id", "amount"]
                 ].sort_values('amount', ascending=False)

rank = pd.RangeIndex(start=1, stop=len(ans) + 1, step=1)
ans["rank"] = rank

print(ans.head(10))

"""公式回答
df_tmp = pd.concat([df_receipt[['customer_id', 'amount']],\
    df_receipt['amount'].rank(method='min', ascending=False)], axis=1)

df_tmp.columns = ['customer_id', 'amount', 'ranking']
df_tmp.sort_values("ranking", ascending=True).head(10)
"""
# %%
# P-020:
ans = df_receipt[["customer_id", "amount"]
                 ].sort_values('amount', ascending=False)

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
# P-021:
print(len(df_receipt))

# %%
# P-022:
print(len(df_receipt['customer_id'].unique()))

# %%
# P-023:
ans = df_receipt.groupby("store_cd")
print(ans[["amount", "quantity"]].sum())

# %%
# 解き直し2021年5月28日
# P-024:
ans = df_receipt.groupby("customer_id")
print(ans.sales_ymd.max().head(10))

# 模範回答
# df_receipt.groupby('customer_id').sales_ymd.max().reset_index().head(10)

# %%
# P-025:
df_receipt.groupby("customer_id").sales_ymd.min().head(10)

# %%
# P-026:
atarashi = df_receipt.groupby("customer_id").sales_ymd.max()
furui = df_receipt.groupby("customer_id").sales_ymd.min()

df_tmp = pd.concat([atarashi, furui], axis=1)
df_tmp.columns = ['latest', 'oldest']

df_tmp[df_tmp.latest != df_tmp.oldest].head(10)

# %%
# P-027:
df_receipt.groupby("store_cd").amount.mean()\
    .sort_values(ascending=False).head(5)

# %%
# P-028:
df_receipt.groupby("store_cd").amount.median(
).sort_values(ascending=False).head(5)
# %%
# P-029:
df_receipt.groupby("store_cd")['product_cd'].apply(lambda x: x.mode())

# %%
# 5月30日解き直し
# P-030:
df_receipt.groupby("store_cd").amount.var(
    ddof=0).sort_values(ascending=False).head(5)

# %%
# 5月30日解き直し
# P-031:
df_receipt.groupby("store_cd").amount.std(
    ddof=0).sort_values(ascending=False).head(5)
# %%
# P-032:
df_receipt.amount.quantile([0, 0.25, 0.5, 0.75, 1])

# %%
# P-033:
df_receipt.groupby("store_cd").amount.mean()[
    df_receipt.groupby("store_cd").amount.mean() >= 330]

#df_receipt.groupby("store_cd").amount.mean().reset_index().query('amount>= 330')

# %%
# 5月30日解き直し
# P-034:
a = df_receipt.query('customer_id.str.contains("^[A-Y]")', engine='python')
b = sum(a["amount"])
c = len(a["customer_id"].unique())

print(b/c)

# 公式回答
#df_receipt.query('not customer_id.str.startswith("Z")', engine='python').groupby("customer_id").amount.sum().mean()

# %%
# 5月30日解き直し
# P-035:
a = df_receipt[~ df_receipt["customer_id"].str.startswith(
    "Z")].groupby("customer_id").amount.sum()

b = a.mean()
c = a.reset_index()

c[c["amount"] >= b].head(10)

"""
notZ_list = df_receipt.query('not customer_id.str.startswith("Z")',
                             engine="python").groupby("customer_id").sum("amount").reset_index()
notZ_list_mean = notZ_list["amount"].mean()

notZ_list[notZ_list["amount"]>=notZ_list_mean].head(10)
"""

# %%
# P-036:
pd.merge(df_receipt, df_store[["store_cd", "store_name"]],
         how="inner", on="store_cd").head(10)
# %%
# P-037:
columns_list = []
for i in df_product.columns:
    columns_list.append(i)

columns_list.append("category_small_name")

ans = pd.concat([df_product, df_category], axis=1, join="inner")
ans[columns_list].head()
# %%
df_receipt
# %%
# P-038:
amo = df_receipt.groupby("customer_id").amount.sum().reset_index()
cus = df_customer.query('gender_cd == 1 and not customer_id.str.startswith("Z")',
                        engine="python").reset_index(drop=True)

pd.merge(cus["customer_id"], amo, how="left",
         on="customer_id").fillna(0).head(10)

# %%
# P-039
ans1 = df_receipt[~df_receipt.duplicated(subset=["customer_id", "sales_ymd"])].reset_index().query(
    'not customer_id.str.startswith("Z")', engine="python").groupby("customer_id").sales_ymd.count().reset_index().sort_values("sales_ymd", ascending=False).head(20)

ans2 = df_receipt.query('not customer_id.str.startswith("Z")', engine="python").groupby(
    "customer_id").amount.sum().reset_index().sort_values("amount", ascending=False).head(20)

pd.merge(ans2, ans1, how="outer", on="customer_id")
# %%
# P-40
len(df_store["store_cd"].unique())*len(df_product["product_cd"].unique())
# %%
# P-41
df_receipt.groupby('sales_ymd').amount.sum().reset_index().diff().head(10)

# %%
# P-42
df_receipt.groupby("sales_ymd").amount.sum().reset_index()

# pd.merge(day,day1)
# %%
# P-042
SA = df_receipt.groupby("sales_ymd").amount.sum().reset_index()
ans = pd.concat([SA, SA.shift(1)], axis=1)
for i in range(2, 4):
    ans = pd.concat([ans, SA.shift(i)], axis=1)
ans.columns = ["sales_ymd", "amount", "lag_ymd_1", "lag_amount_1", "lag_ymd_2",
               "lag_amount_2", "lag_ymd_3", "lag_amount_3"]

ans.dropna().head(10)
# %%
# 6月6日一応やり直し
# P-043
Money = df_receipt.groupby("customer_id").amount.sum().reset_index()
Money
# %%
Cus = pd.concat([df_customer[["customer_id", "gender_cd"]],
                df_customer["age"]//10 * 10], axis=1).reset_index(drop=True)
Cus
# %%
ans = pd.merge(Cus, Money, how="left", on="customer_id").reset_index(drop=True)
ans
# %%
df_sales_summary = pd.pivot_table(
    ans, index="age", columns="gender_cd", values="amount", aggfunc='sum')
print(df_sales_summary)

# %%
# ６月7日題意読み解きやり直し
# P-044
ans.iloc[ans["gender_cd"] == 0, 1] = "00"
ans.iloc[ans["gender_cd"] == 1, 1] = "01"
ans.iloc[ans["gender_cd"] == 9, 1] = "99"
ans

pd.pivot_table(ans, index=["age", "gender_cd"],
               values="amount", aggfunc="sum").reset_index()
# %%
# ６月7日日付への変換方法確認
# P-045
# df_customer.dtypes
pd.to_datetime(df_customer['birth_day']).dt.strftime('%Y%m%d')

# %%
# P-046
time = pd.to_datetime(df_customer["application_date"].astype("str"))
pd.concat([df_customer["customer_id"], time], axis=1).head(10)

# %%
# P-047
# 6月8日やり直し
# df_receipt.dtypes
# df_receipt
time = pd.to_datetime(df_receipt["sales_ymd"].astype("str"))
pd.concat([df_receipt[["receipt_no", "receipt_sub_no"]], time], axis=1).head(10)

# %%
# P-048
# 6月8日やり直し
# df_receipt.dtypes
time = pd.to_datetime(df_receipt["sales_epoch"], unit="s")
pd.concat([df_receipt[["receipt_no", "receipt_sub_no"]], time], axis=1).head(10)

# %%
# P-049
time = pd.to_datetime(df_receipt["sales_epoch"], unit="s").dt.strftime('%Y')
pd.concat([df_receipt[["receipt_no", "receipt_sub_no"]], time], axis=1).head(10)

# %%
# P-050:
time = pd.to_datetime(df_receipt["sales_epoch"], unit="s").dt.strftime('%m')
pd.concat([df_receipt[["receipt_no", "receipt_sub_no"]], time], axis=1).head(10)

# %%
# P-051
time = pd.to_datetime(df_receipt["sales_epoch"], unit="s").dt.strftime('%d')
pd.concat([df_receipt[["receipt_no", "receipt_sub_no"]], time], axis=1).head(10)

# %%
# P-052
df_tmp = df_receipt.query('not customer_id.str.startswith("Z")', engine="python").groupby(
    "customer_id").amount.sum().reset_index()


df_tmp.iloc[df_tmp["amount"] <= 2000, 1] = 0
df_tmp.iloc[df_tmp["amount"] > 2000, 1] = 1

pd.concat([df_receipt.query('not customer_id.str.startswith("Z")', engine="python").groupby(
    "customer_id").amount.sum().reset_index(), df_tmp["amount"]], axis=1).head(10)

# %%
# 053
df_tmp = df_customer[["customer_id", "postal_cd"]].copy()
df_tmp["postal_flg"] = df_tmp["postal_cd"].apply(
    lambda x: 1 if 100 <= int(x[0:3]) <= 209 else 0)

pd.merge(df_tmp, df_receipt, how="inner",
         on="customer_id").groupby("postal_flg").agg({"customer_id": "nunique"})

# %%
# 054
tmp = df_customer[["customer_id", "address"]].copy()


def func_address_flg(x):
    if x[0:2] == "東京":
        return 13
    elif x[0:2] == "千葉":
        return 12
    elif x[0:2] == "埼玉":
        return 11
    else:
        return 14


tmp["address_flg"] = df_customer["address"].apply(func_address_flg)
tmp

# 模範回答
tmp["address_flg"] = df_customer["address"].str[0:2].map(
    {"埼玉": "11", "千葉": "12", "東京": "13", "神奈": "14"})
tmp

# %%
# 055
ans = df_receipt.groupby("customer_id").amount.sum().reset_index()
points = ans.quantile([0.25, 0.5, 0.75])


def func_25points(x):
    if x < points.iloc[0][0]:
        return "1"
    elif x < points.iloc[1][0]:
        return "2"
    elif x < points.iloc[2][0]:
        return "3"
    else:
        return "4"


ans["25%_flg"] = ans["amount"].apply(func_25points)
ans.head(10)

# %%
# 056
ans = df_customer[["customer_id", "birth_day", "age"]]
ans["age_layer"] = ans["age"].apply(lambda x: x // 10*10 if x < 60 else 60)
ans.head(10)
# %%
# 057
ans = pd.merge(ans, df_customer[["customer_id", "gender_cd"]],
               how="left", on="customer_id")
ans["AL*10+GC"] = ans["age_layer"].astype("str") + \
    ans["gender_cd"].astype("str")
ans.head(10)
# %%
# 058
df_customer[["0", "1", "9"]] = pd.get_dummies(df_customer["gender_cd"])
df_customer[["customer_id", "0", "1", "9"]].head(10)


# %%
#解き直し
# 59
ans = df_receipt.query('not customer_id.str.startswith("Z")', engine="python").groupby(
    "customer_id").amount.sum().reset_index()

ans["amount_ss"] = preprocessing.scale(ans["amount"])
ans.head(10)

# %%
#解き直し6月19日
# 60
ans = df_receipt.query('not customer_id.str.startswith("Z")', engine="python").groupby(
    "customer_id").amount.sum().reset_index()
ans["amount_mm"] = preprocessing.minmax_scale(ans["amount"])
ans.head(10)
# %%
#解き直し6月19日
# 61
ans = df_receipt.query('not customer_id.str.startswith("Z")', engine="python").groupby(
    "customer_id").amount.sum().reset_index()
ans['amount_log10'] = np.log10(ans['amount']+1)
ans.head(10)

# %%
#解き直し6月19日
#62
ans = df_receipt.query('not customer_id.str.startswith("Z")', engine="python").groupby(
    "customer_id").amount.sum().reset_index()
ans['amount_log'] = np.log(ans['amount']+1)
ans.head(10)

# %%
#解き直し6月19日
#63
tmp = df_product.dropna()
tmp["unit_profit"] = tmp["unit_price"]-tmp["unit_cost"]
tmp.head(10)
#ans = df_product.
#ans
# %%
#64
tmp = df_product.copy()
tmp["unit_profit_rate"]= (tmp["unit_price"]-tmp["unit_cost"])/tmp["unit_price"]
tmp["unit_profit_rate"].mean(skipna=True)
tmp
# %%
#65
#P-065: 商品データフレーム（df_product）の各商品について、利益率が30%となる新たな単価を求めよ。
# ただし、1円未満は切り捨てること。そして結果を10件表示させ、利益率がおよそ30％付近であることを確認せよ。ただし、単価（unit_price）と原価（unit_cost）にはNULLが存在することに注意せよ。
tmp = df_product.copy()
tmp["new_unit_price"] = tmp["unit_cost"].apply(lambda x: np.floor(x/0.7))
tmp["new_profit_rate"]=(tmp["new_unit_price"]-tmp["unit_cost"])/tmp["new_unit_price"]
tmp[:10]

#%% 
# 66
tmp = df_product.copy().dropna()
tmp["new_unit_price"] = tmp["unit_cost"].apply(lambda x: round(x/0.7))
tmp["new_profit_rate"]=(tmp["new_unit_price"]-tmp["unit_cost"])/tmp["new_unit_price"]
tmp[:10]

# %%
# # P-067: 商品データフレーム（df_product）の各商品について、利益率が30%となる新たな単価を求めよ。
# 今回は、1円未満を切り上げること。
# そして結果を10件表示させ、利益率がおよそ30％付近であることを確認せよ。
# ただし、単価（unit_price）と原価（unit_cost）にはNULLが存在することに注意せよ。

tmp = df_product.copy().dropna()
tmp["new_unit_price"] = tmp["unit_cost"].apply(lambda x: math.ceil(x/0.7))
tmp["new_profit_rate"]=(tmp["new_unit_price"]-tmp["unit_cost"])/tmp["new_unit_price"]
tmp[:10]
# %%
# P-068: 商品データフレーム（df_product）の各商品について、消費税率10%の税込み金額を求めよ。 
# 1円未満の端数は切り捨てとし、結果は10件表示すれば良い。
# ただし、単価（unit_price）にはNULLが存在することに注意せよ。

tmp = df_product.copy().dropna()
tmp["tax_unit_price"] = tmp["unit_price"].apply(lambda x: np.floor(x*1.1))
tmp[:10]
# %%
# P-069: 
# レシート明細データフレーム（df_receipt）と商品データフレーム（df_product）を結合し、
# 顧客毎に全商品の売上金額合計と、
# カテゴリ大区分（category_major_cd）が"07"（瓶詰缶詰）の売上金額合計を計算の上、
# 両者の比率を求めよ。
# 抽出対象はカテゴリ大区分"07"（瓶詰缶詰）の購入実績がある顧客のみとし、
# 結果は10件表示させればよい。
df_receipt

#%%
df_product

#%%
df_tmp_1=pd.merge(df_receipt,df_product,how="inner",on="product_cd").groupby("customer_id").aggregate({"amount":"sum"}).reset_index()
df_tmp_1

#%%
df_tmp_1 = df_tmp_1.rename(columns={"amount": "all_amount"})

#%%
df_product["category_major_cd"].unique()
df_tmp_1
#%%
df_tmp_2=pd.merge(df_receipt,df_product.query("category_major_cd==7"),how="inner",on="product_cd").groupby("customer_id").aggregate({"amount":"sum"}).reset_index()
df_tmp_2
#%%
df_tmp_2 = df_tmp_2.rename(columns={"amount": "7_amount"})
df_tmp_2
# %%
df_tmp_3=pd.merge(df_tmp_1,df_tmp_2,how="outer",on="customer_id")
df_tmp_3
# %%
df_tmp_3["ratio"]=df_tmp_3["7_amount"]/df_tmp_3["all_amount"]
df_tmp_3[:10]
# %%
# P-070: レシート明細データフレーム（df_receipt）の売上日（sales_ymd）に対し、
# 顧客データフレーム（df_customer）の会員申込日（application_date）からの経過日数を計算し、
# 顧客ID（customer_id）、売上日、会員申込日とともに表示せよ。
# 結果は10件表示させれば良い
# （なお、sales_ymdは数値、application_dateは文字列でデータを保持している点に注意）。

df_receipt["sales_ymd"]

# %%
df_customer["application_date"]
# %%
df_tmp = pd.merge(df_receipt,df_customer,how="inner",on="customer_id")
df_tmp
#%%
#df_tmp[["sales_ymd","application_date"]]
# %%
df_tmp = pd.merge(df_receipt[['customer_id', 'sales_ymd']], df_customer[['customer_id', 'application_date']],
                 how='inner', on='customer_id')
df_tmp
# %%
df_tmp = df_tmp.drop_duplicates()
df_tmp
# %%
df_tmp['sales_ymd'] = pd.to_datetime(df_tmp['sales_ymd'].astype('str'))
df_tmp

#%%
df_tmp['application_date'] = pd.to_datetime(df_tmp['application_date'])
df_tmp

#%%
df_tmp['elapsed_date'] = df_tmp['sales_ymd'] - df_tmp['application_date']
df_tmp.head(10)
# %%
