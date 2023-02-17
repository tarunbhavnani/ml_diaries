#!/usr/bin/env python3
"""
make_dataset.py
"""

import lightgbm as lgb
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import cross_validate


def main() -> None:
    df = pd.read_csv("train.csv", index_col="id")
    tdf = pd.read_csv("test.csv")

    for i in range(1, 11):
        print(i)
        imputer = KNNImputer(n_neighbors=i)

        train = pd.DataFrame(
            imputer.fit_transform(
                df.drop(columns="song_popularity"), y=df["song_popularity"]
            ),
            columns=df.columns[:-1],
        )
        
        train['song_popularity']= df['song_popularity']
        
        test = pd.DataFrame(
            imputer.transform(tdf.drop(columns="id")), columns=tdf.columns[1:]
        )
        
        
        
        train.to_csv(f"train-knn{i}.csv", index=False)
        test.to_csv(f"test-knn{i}.csv", index=False)

        cv = cross_validate(
            lgb.LGBMClassifier(),
            X=train.drop(columns="song_popularity"),
            y=df["song_popularity"],
            return_train_score=True,
            scoring="roc_auc",
            verbose=True,
            cv=5,
            fit_params={"eval_metric": "auc"},
        )
        print(cv)


if __name__ == "__main__":
    main()