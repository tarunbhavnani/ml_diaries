# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 12:00:23 2023

@author: ELECTROBOT
"""

import pandas as pd
import numpy as np
import scipy
import pandas as pd
import math
import random
import sklearn
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt



articles_df = pd.read_csv(r"D:\kaggle\Deskdrop\shared_articles.csv")
articles_df = articles_df[articles_df['eventType'] == 'CONTENT SHARED']
articles_df.head(5)
list(articles_df)
interactions_df = pd.read_csv(r"D:\kaggle\Deskdrop\users_interactions.csv")
interactions_df.head(10)
list(interactions_df)

# sum(interactions_df.contentId.isin(articles_df.contentId))/len(interactions_df)
# sum(articles_df.contentId.isin(interactions_df.contentId))/len(articles_df)

#assign severity
event_type_strength = {
   'VIEW': 1.0,
   'LIKE': 2.0, 
   'BOOKMARK': 2.5, 
   'FOLLOW': 3.0,
   'COMMENT CREATED': 4.0,  
}


interactions_df['eventStrength'] = interactions_df["eventType"].apply(lambda x: event_type_strength[x])

#jk=interactions_df.groupby(['contentId', 'personId'])['eventStrength'].apply(lambda x: sum(x)).reset_index()
#jk['eventStrength']=[smooth_user_preference(i) for i in jk['eventStrength']]
list(interactions_df)

#get personid connects, and remove less then 5
#interactions_df.groupby("personId").size() # we dont use this as same content is removed

users_interactions_count_df = interactions_df.groupby(['personId', 'contentId']).size().groupby('personId').size()
print('# users: %d' % len(users_interactions_count_df))
users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 5].reset_index()[['personId']]
print('# users with at least 5 interactions: %d' % len(users_with_enough_interactions_df))



interactions_from_selected_users_df= interactions_df.merge(users_with_enough_interactions_df, left_on="personId", right_on="personId", how="right")

# interactions_from_selected_users_df["strength"]=interactions_from_selected_users_df.groupby(['personId', 'contentId'])["eventStrength"].transform("sum")


#log transformation to smooth the distribution of weighted average of strength per user.
def smooth_user_preference(x):
    return math.log(1+x, 2)
    
interactions_full_df = interactions_from_selected_users_df \
                    .groupby(['personId', 'contentId'])['eventStrength'].sum() \
                    .apply(smooth_user_preference).reset_index()
print('# of unique user/item interactions: %d' % len(interactions_full_df))
interactions_full_df.head(10)

#user-item matrix
#ui= interactions_full_df.pivot(index="personId", columns="contentId", values="eventStrength").fillna(0)



#popularity model

interactions_full_df["content_strength"]=interactions_full_df.groupby("contentId").transform("sum").values[:,1]


#interactions_full_df["person_content_strength"]=interactions_full_df.groupby("personId").transform("sum").values[:,2]

popularity_data= interactions_full_df[["contentId", "content_strength"]].drop_duplicates().sort_values(["content_strength"], ascending=False)

#suggest to some personId
pid=-9223121837663643404

person_interacted_items= list(set(interactions_full_df[interactions_full_df["personId"]==pid]["contentId"]))


#suggestions=popularity_data[[True if i not in person_interacted_items else False for i in popularity_data.contentId ]][0:10]
#suggest the top rated which are not in already interacted

suggestions= popularity_data[~popularity_data["contentId"].isin(person_interacted_items)][0:10]

#get the full data
#articles_df
suggestions= articles_df.merge(suggestions,right_on="contentId",left_on="contentId", how="right")



# =============================================================================
# #item based, here we have movies so we will get the item similarity on text provided using tfidf
# =============================================================================

#take all movies for the user
#create vectorozer
#take all the rest of the movies
#now calculate the score for each of the related movies with each of the new moviw and add them to the bnew movi score
#sort teh score
#give top 1000




list(articles_df)
articles_df.text

vectorizer = TfidfVectorizer(analyzer='word',
                     ngram_range=(1, 2),
                     min_df=0.003,
                     max_df=0.5,
                     max_features=5000,
                     stop_words="english")

vectorizer.fit(articles_df.text.values)

X= vectorizer.transform(articles_df.text.values)

#lets check for pid pid=-9223121837663643404

related_content=list(set(interactions_full_df[interactions_full_df["personId"]==-1479311724257856983]["contentId"]))


related_text=articles_df[articles_df["contentId"].isin(related_content)]["text"].reset_index()

#lets find similar stuff
#rated_text=pd.DataFrame([0]*len(articles_df),columns=["score"])
rated_text=articles_df.copy()
rated_text["score"]=0
for num in range(len(related_text)):
    #break
    text=related_text.text.iloc[num]
    text_vec= vectorizer.transform([text])
    scores= cosine_similarity(X,text_vec)
    scores=[i[0] for i in scores]
    rated_text["score"]+=scores

#remove retalted text
rated_text=rated_text[~rated_text.contentId.isin(related_content)]
rated_text=rated_text.sort_values(by=["score"], ascending=False)[0:1000]
rated_text["text"]


# =============================================================================
# #collaborative filtering
# =============================================================================

ui= interactions_full_df.pivot(index="personId", columns="contentId", values="eventStrength").fillna(0)

#singular decompose
matrix= ui.values
u, s, v = svds(matrix)
u.shape, v.shape, s.shape

matrix_reconstructed= np.dot(np.dot(u, np.diag(s)), v)
#normalize
matrix_reconstructed_norm = (matrix_reconstructed - matrix_reconstructed.min()) / (matrix_reconstructed.max() - matrix_reconstructed.min())

ui_reconstructed=pd.DataFrame(matrix_reconstructed_norm)
ui_reconstructed=ui_reconstructed.set_index(ui.index)
ui_reconstructed.columns= ui.columns


#get recommendations for 9223121837663643404

cont=ui_reconstructed.loc[-1479311724257856983].reset_index()
cont.columns=["contentId", "score"]

related_content=list(set(interactions_full_df[interactions_full_df["personId"]==-1479311724257856983]["contentId"]))

cont=cont[~cont.contentId.isin(related_content)]
cont= cont.sort_values(by="score", ascending=False)[0:1000]
cont= cont.merge(articles_df, left_on="contentId", right_on="contentId", how="left")
cont.text


#hybrid model


hybrid= cont.merge(rated_text, left_on= "contentId", right_on="contentId")


hybrid["final_score_x"]= hybrid["score_x"].apply(lambda x:(x-max(hybrid["score_x"]))/(max(hybrid["score_x"])-min(hybrid["score_x"])))
hybrid["final_score_y"]= hybrid["score_y"].apply(lambda x:(x-max(hybrid["score_y"]))/(max(hybrid["score_y"])-min(hybrid["score_y"])))

#lets give severity of .3 to cf and .7 to tfidf
hybrid["final_score"]= [.3*i+.7*j for i,j in zip(hybrid["final_score_x"],hybrid["final_score_y"])]

hybrid= hybrid.sort_values(by="final_score", ascending=False)




#we have seenn popularity, item based, user based and hybrid of item and user
#next is model based.


# =============================================================================
# #svd surprise to predict!!
# =============================================================================

from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

reader = Reader()

# get just top 100K rows for faster run time


data = Dataset.load_from_df(interactions_full_df[['personId', 'contentId', 'eventStrength']][:], reader)

svd = SVD()

cross_validate(svd, data, measures=['RMSE', 'MAE'])


#id=1479311724257856983

df_785314=interactions_full_df.copy()

df_785314['Estimate_Score']=[svd.predict(-1479311724257856983,i).est for i in df_785314['contentId']]

related_content=list(set(interactions_full_df[interactions_full_df["personId"]==-1479311724257856983]["contentId"]))

df_785314=df_785314[~df_785314.contentId.isin(related_content)]
df_785314=df_785314[["contentId","Estimate_Score"] ].drop_duplicates()
df_785314=df_785314[["contentId","Estimate_Score"] ].drop_duplicates()
df_785314 = df_785314.merge(articles_df, left_on="contentId", right_on="contentId", how="left")

df_785314=df_785314.sort_values(by="Estimate_Score", ascending=False)[0:1000]

print(df_785314.text)



# =============================================================================
# 
# =============================================================================
trainset = data.build_full_trainset()
svd.fit(trainset)
svd.predict(uid=-1479311724257856983, iid=-8949113594875411859 )
df=articles_df.copy()
df["pred"]=0
for i in range(len(df.contentId)):
    
    df["pred"].iloc[i]=svd.predict(uid=-1479311724257856983, iid=df.contentId.iloc[i] ).est

df=df.sort_values(by= "pred", ascending=False)

df=df[~df.contentId.isin(related_content)]

df.text


# =============================================================================
# 
# =============================================================================


interactions_train_df, interactions_test_df = train_test_split(interactions_full_df,
                                   stratify=interactions_full_df['personId'], 
                                   test_size=0.20,
                                   random_state=42)

print('# interactions on Train set: %d' % len(interactions_train_df))
print('# interactions on Test set: %d' % len(interactions_test_df))

#Indexing by personId to speed up the searches during evaluation
interactions_full_indexed_df = interactions_full_df.set_index('personId')
interactions_train_indexed_df = interactions_train_df.set_index('personId')
interactions_test_indexed_df = interactions_test_df.set_index('personId')


def get_items_interacted(person_id, interactions_df):
    # Get the user's data and merge in the movie information.
    interacted_items = interactions_df.loc[person_id]['contentId']
    return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])



#Top-N accuracy metrics consts
EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 100

class ModelEvaluator:


    def get_not_interacted_items_sample(self, person_id, sample_size, seed=42):
        interacted_items = get_items_interacted(person_id, interactions_full_indexed_df)
        all_items = set(articles_df['contentId'])
        non_interacted_items = all_items - interacted_items

        random.seed(seed)
        non_interacted_items_sample = random.sample(non_interacted_items, sample_size)
        return set(non_interacted_items_sample)

    def _verify_hit_top_n(self, item_id, recommended_items, topn):        
            try:
                index = next(i for i, c in enumerate(recommended_items) if c == item_id)
            except:
                index = -1
            hit = int(index in range(0, topn))
            return hit, index

    def evaluate_model_for_user(self, model, person_id):
        #model, person_id=popularity_model,-830175562779396891
        
        #Getting the items in test set
        interacted_values_testset = interactions_test_indexed_df.loc[person_id]
        if type(interacted_values_testset['contentId']) == pd.Series:
            person_interacted_items_testset = set(interacted_values_testset['contentId'])
        else:
            person_interacted_items_testset = set([int(interacted_values_testset['contentId'])])  
        interacted_items_count_testset = len(person_interacted_items_testset) 

        #Getting a ranked recommendation list from a model for a given user
        person_recs_df = model.recommend_items(person_id, 
                                               items_to_ignore=get_items_interacted(person_id, 
                                                                                    interactions_train_indexed_df), 
                                               topn=10000000000)
        hits_at_5_count = 0
        hits_at_10_count = 0
        #For each item the user has interacted in test set
        for item_id in person_interacted_items_testset:
            #Getting a random sample (100) items the user has not interacted 
            #(to represent items that are assumed to be no relevant to the user)
            non_interacted_items_sample = self.get_not_interacted_items_sample(person_id, 
                                                                          sample_size=EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS, 
                                                                          seed=item_id%(2**32))

            #Combining the current interacted item with the 100 random items
            items_to_filter_recs = non_interacted_items_sample.union(set([item_id]))

            #Filtering only recommendations that are either the interacted item or from a random sample of 100 non-interacted items
            valid_recs_df = person_recs_df[person_recs_df['contentId'].isin(items_to_filter_recs)]                    
            valid_recs = valid_recs_df['contentId'].values
            #Verifying if the current interacted item is among the Top-N recommended items
            hit_at_5, index_at_5 = self._verify_hit_top_n(item_id, valid_recs, 5)
            hits_at_5_count += hit_at_5
            hit_at_10, index_at_10 = self._verify_hit_top_n(item_id, valid_recs, 10)
            hits_at_10_count += hit_at_10

        #Recall is the rate of the interacted items that are ranked among the Top-N recommended items, 
        #when mixed with a set of non-relevant items
        recall_at_5 = hits_at_5_count / float(interacted_items_count_testset)
        recall_at_10 = hits_at_10_count / float(interacted_items_count_testset)

        person_metrics = {'hits@5_count':hits_at_5_count, 
                          'hits@10_count':hits_at_10_count, 
                          'interacted_count': interacted_items_count_testset,
                          'recall@5': recall_at_5,
                          'recall@10': recall_at_10}
        return person_metrics

    def evaluate_model(self, model):
        #model=popularity_model
        #print('Running evaluation for users')
        people_metrics = []
        for idx, person_id in enumerate(list(interactions_test_indexed_df.index.unique().values)):
            #break
            #if idx % 100 == 0 and idx > 0:
            #    print('%d users processed' % idx)
            person_metrics = self.evaluate_model_for_user(model, person_id)  
            person_metrics['_person_id'] = person_id
            people_metrics.append(person_metrics)
        print('%d users processed' % idx)

        detailed_results_df = pd.DataFrame(people_metrics) \
                            .sort_values('interacted_count', ascending=False)
        
        global_recall_at_5 = detailed_results_df['hits@5_count'].sum() / float(detailed_results_df['interacted_count'].sum())
        global_recall_at_10 = detailed_results_df['hits@10_count'].sum() / float(detailed_results_df['interacted_count'].sum())
        
        global_metrics = {'modelName': model.get_model_name(),
                          'recall@5': global_recall_at_5,
                          'recall@10': global_recall_at_10}    
        return global_metrics, detailed_results_df
    
model_evaluator = ModelEvaluator()    

#popularity model
#Computes the most popular items
item_popularity_df = interactions_full_df.groupby('contentId')['eventStrength'].sum().sort_values(ascending=False).reset_index()
item_popularity_df.head(10)


class PopularityRecommender:
    
    MODEL_NAME = 'Popularity'
    
    def __init__(self, popularity_df, items_df=None):
        self.popularity_df = popularity_df#popularity_df=item_popularity_df
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Recommend the more popular items that the user hasn't seen yet.
        recommendations_df = self.popularity_df[~self.popularity_df['contentId'].isin(items_to_ignore)] \
                               .sort_values('eventStrength', ascending = False) \
                               .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'contentId', 
                                                          right_on = 'contentId')[['eventStrength', 'contentId', 'title', 'url', 'lang']]


        return recommendations_df
    
popularity_model = PopularityRecommender(item_popularity_df, articles_df)




print('Evaluating Popularity recommendation model...')
pop_global_metrics, pop_detailed_results_df = model_evaluator.evaluate_model(popularity_model)
print('\nGlobal metrics:\n%s' % pop_global_metrics)
pop_detailed_results_df.head(10)