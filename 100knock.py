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

# P-001: レシート明細のデータフレーム（df_receipt）から全項目の先頭10件を表示し、どのようなデータを保有しているか目視で確認せよ。

# %%
# P-001: レシート明細のデータフレーム（df_receipt）から全項目の先頭10件を表示し、どのようなデータを保有しているか目視で確認せよ。
print(df_receipt.head(10))

# %%
