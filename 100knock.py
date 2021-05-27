# ここからもらいました
# https://github.com/noguhiro2002/100knocks-preprocess_ForColab-AzureNotebook/blob/master/preprocess_knock_Python_Colab.ipynb

# %%
import os
import pandas as pd
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
df_receipt = df_receipt.rename(columns={"sales_ymd": "sales_date"})
listP002 = ["sales_date", "customer_id", "product_cd", "amount"]

df_receipt[listP002].head(10)

# %%
# P-004: レシート明細のデータフレーム（df_receipt）から売上日（sales_ymd）、顧客ID（customer_id）、商品コード（product_cd）、売上金額（amount）の順に列を指定し、以下の条件を満たすデータを抽出せよ。
df_receipt = df_receipt.rename(columns={"sales_date": "sales_ymd"})
listP002 = ["sales_ymd", "customer_id", "product_cd", "amount"]

#df_receipt[df_receipt.customer_id == "CS018205000001"][listP002]

df_receipt[listP002].query('customer_id == "CS018205000001"')
# %%
