# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 15:10:51 2022

@author: ELECTROBOT
"""
import os
os.chdir(r'C:\Users\ELECTROBOT\Desktop\kaggle\Song_popularity')

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
color_pal = plt.rcParams['axes.prop_cycle'].by_key()['color']

import warnings
warnings.filterwarnings("ignore")
plt.style.use('ggplot')


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
ss = pd.read_csv("sample_submission.csv")

train["isTrain"] = True
test["isTrain"] = False

tt = pd.concat([train, test]).reset_index(drop=True).copy()


FEATURES = [
    "song_duration_ms",
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "key",
    "liveness",
    "loudness",
    "audio_mode",
    "speechiness",
    "tempo",
    "time_signature",
    "audio_valence",
]

#What are the counts of missing values in train vs. test?

ncounts = pd.DataFrame([train.isna().mean(), test.isna().mean()]).T
ncounts = ncounts.rename(columns={0: "train_missing", 1: "test_missing"})

ncounts.query("train_missing > 0").plot(
    kind="barh", figsize=(8, 5), title="% of Values Missing"
)
plt.show()



nacols = [
    "song_duration_ms",
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "key",
    "liveness",
    "loudness",
]


tt["n_missing"]=tt[nacols].isna().sum(axis=1)

tt['n_missing'].value_counts().plot(kind='bar', title= "Number of missing in observations")

tt.query("n_missing==6")


#Do we see an imbalance in missing values when splitting by other features?

tt.groupby("audio_mode")["n_missing"].mean()

tt.groupby("time_signature")["n_missing"].agg(['mean', 'count'])



#create columns for missing or not
tt_missing_tag_df = tt[nacols].isna()
tt_missing_tag_df.columns = [f"{i}_missing" for i in tt_missing_tag_df.columns]

tt = pd.concat([tt, tt_missing_tag_df], axis=1)


# =============================================================================
# Protip:
# Try to predict the target using only missing value indicators as features
# =============================================================================
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score

lr = LogisticRegressionCV(scoring="accuracy")

X = tt.query("isTrain")[
    [
        "song_duration_ms_missing",
        "acousticness_missing",
        "danceability_missing",
        "energy_missing",
        "instrumentalness_missing",
        "key_missing",
        "liveness_missing",
        "loudness_missing",
    ]
]

y = tt.query("isTrain")["song_popularity"]

lr.fit(X, y)
lr.score(X, y)

preds = lr.predict_proba(X)[:, 0]

roc_auc_score(y, preds)

#.49



# =============================================================================
# #one --do noth9ng
# =============================================================================

import lightgbm as lgb
lgbm_params = {
    'objective': 'regression',
    'metric': 'auc',
    'verbose': -1,
    'boost_from_average': False,
    'min_data': 1,
    'num_leaves': 2,
    'learning_rate': .1,
    'min_data_in_bin': 1,
#     'use_missing': False,
#     'zero_as_missing': True
}

model = lgb.LGBMClassifier(**lgbm_params)
#model = lgb.LGBMClassifier()


model.fit(train[FEATURES], train["song_popularity"])
model.score(train[FEATURES], train["song_popularity"])

preds = model.predict_proba(train[FEATURES])[:, 0]

roc_auc_score(y, preds)








# =============================================================================
# #drop al na
# =============================================================================

tt.dropna(axis=0) #drop all rows with any NA
tt.dropna(axis=1) # drop all columns with any NA





# =============================================================================
# Pandas imputation
# =============================================================================









