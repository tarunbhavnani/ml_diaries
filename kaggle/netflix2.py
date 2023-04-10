import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD


data= pd.read_csv(r"C:\Users\ELECTROBOT\Desktop\kaggle\netflix\netflix_0.csv")

    
#reduce the data
hj=data.groupby("Movie_Id")["Rating"].count().reset_index()

hj.describe()
mv=list(hj[(hj.Rating>200) & (hj.Rating<2000)]["Movie_Id"])

df=data[data.Movie_Id.isin(mv)]

df.Rating.value_counts()
df.Movie_Id.value_counts()



#create ui matrix
rdf=df.pivot(index="Cust_Id", columns="Movie_Id", values="Rating").fillna(0)
matrix= rdf.values


num_components = 10

#u, s, v = svds(matrix, k=6)
u, s, v = svds(matrix, k=2)
u.shape, s.shape, v.shape

plt.scatter(u[:,0],u[:,1])
plt.scatter(v.T[:,0],v.T[:,1])
#marix_recreated=np.dot(np.dot(u,np.diag(s)), v)


svd = TruncatedSVD(n_components=6)
features = svd.fit_transform(matrix)

plt.scatter(features[:,0],features[:,1])

#very similar to u , just the scale differs




from sklearn.metrics.pairwise import cosine_similarity

#find all users similar to custid 1

cust=np.array(rdf.iloc[0]).reshape(1,862)

cust= svd.transform(cust)

scores=cosine_similarity(features, cust)
scores={num:i[0] for num,i in enumerate(scores)}

scores=sorted(scores.items(), key= lambda x: x[1])[::-1]
top_10_closest=[i[0] for i in scores[0:11]]

rdf_= rdf.reset_index(drop=False)
#customers=list(rdf_[rdf_.index.isin(top_10_closest)]["Cust_Id"])

customers=rdf_[rdf_.index.isin(top_10_closest)]

#okay the results are very sparse and i dont understand how the top 10 colesest to our user are actually similar!!






from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

reader = Reader()

# get just top 100K rows for faster run time
data = Dataset.load_from_df(data[['Cust_Id', 'Movie_Id', 'Rating']][:], reader)

svd = SVD()
%%time
cross_validate(svd, data, measures=['RMSE', 'MAE'])




df_785314 = df[(df['Cust_Id'] == 785314) & (df['Rating'] == 5)]
df_785314 = df_785314.set_index('Movie_Id')
df_785314 = df_785314.join(df_title)['Name']
print(df_785314)



user_785314 = df_title.copy()
user_785314 = user_785314.reset_index()
user_785314 = user_785314[~user_785314['Movie_Id'].isin(drop_movie_list)]

# getting full dataset
data = Dataset.load_from_df(df[['Cust_Id', 'Movie_Id', 'Rating']], reader)

trainset = data.build_full_trainset()
svd.fit(trainset)

user_785314['Estimate_Score'] = user_785314['Movie_Id'].apply(lambda x: svd.predict(785314, x).est)

user_785314 = user_785314.drop('Movie_Id', axis = 1)

user_785314 = user_785314.sort_values('Estimate_Score', ascending=False)
print(user_785314.head(10))



















