#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
from math import sqrt
from tqdm import tqdm_notebook as tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# In[2]:


path = 'D:\다운로드\BX-CSV-Dump'
ratings_df = pd.read_csv(os.path.join(path, 'BX-Book-Ratings_updated.csv'),delimiter=";", encoding='cp949',  error_bad_lines=False, names=['Dummy'])
# 세미콜론 기준으로 데이터 나눔
print(ratings_df.shape)
print(ratings_df.head())


# In[3]:


ratings_df['UserID'] = ratings_df['Dummy'].str.split(';').str[0]
ratings_df['ISBN']= ratings_df['Dummy'].str.split(';').str[1]
ratings_df['Book-Rating']= ratings_df['Dummy'].str.split(';').str[2]
ratings_df = ratings_df.drop(columns=['Dummy'], axis=1)
ratings_df = ratings_df.drop(index=0, axis= 0)
# 새 컬럼에 적용 후 기존 컬럼 삭제, 불필요한 헤더 부분 삭제
ratings_df['Book-Rating'] = ratings_df['Book-Rating'].str.replace('"','')
ratings_df['Book-Rating'] = ratings_df['Book-Rating'].str.replace(',','')
ratings_df['ISBN'] = ratings_df['ISBN'].str.replace('"','')
# 큰따옴표, 콤마 등 삭제
ratings_df.head()


# In[56]:


ratings_df.hist()


# In[5]:


ratings_df['UserID'] = ratings_df['UserID'].apply(pd.to_numeric)
ratings_df['Book-Rating'] = ratings_df['Book-Rating'].apply(pd.to_numeric)
ratings_df.dtypes 
# rating 타입 숫자형으로 변경 


# In[6]:


user_count = ratings_df.groupby('UserID').agg({'Book-Rating' : [np.size, np.mean, np.std]})
book_count = ratings_df.groupby('ISBN').agg({'Book-Rating' : [np.size, np.mean, np.std]})

user_count.loc[user_count[('Book-Rating', 'size')] > 2, 'heavy'] = 1
user_filtered = user_count[user_count['heavy']==1]
user_filtered.describe()


# In[40]:


ratings = ratings_df.loc[ratings_df['UserID'].isin(user_filtered.index)]
#ratings_df.to_csv(os.path.join(path, 'BX-Book-Ratings_updated2.csv'), index=False)

ratings = ratings[:10000]


# In[41]:


train_df, test_df = train_test_split(ratings, test_size=0.3, random_state=1234)

print(train_df.shape)
print(test_df.shape)


# In[42]:


#ValueError: Unstacked DataFrame is too big, causing int32 overflow 문제 발생

sparse_matrix = train_df.groupby('ISBN').apply(lambda x: pd.Series(x['Book-Rating'].values, index=x['UserID'])).unstack()
sparse_matrix.index.name = 'ISBN'

sparse_matrix


# In[43]:


from sklearn.metrics.pairwise import cosine_similarity

def cossim_matrix(a, b):
    cossim_values = cosine_similarity(a.values, b.values)
    cossim_df = pd.DataFrame(data=cossim_values, columns = a.index.values, index=a.index)

    return cossim_df


# In[44]:


user_sparse_matrix = sparse_matrix.fillna(0).transpose()
user_sparse_matrix


# In[45]:


user_cossim_df = cossim_matrix(user_sparse_matrix, user_sparse_matrix)
#user_cossim_df.describe()
user_cossim_df


# In[46]:


isbn_grouped = train_df.groupby('ISBN')
user_prediction_result_df = pd.DataFrame(index=list(isbn_grouped.indices.keys()), columns=user_sparse_matrix.index)
user_prediction_result_df


# In[47]:


for isbn, group in tqdm(isbn_grouped):
    user_sim = user_cossim_df.loc[group['UserID']]
    user_rating = group['Book-Rating']
    sim_sum = user_sim.sum(axis=0)

    pred_ratings = np.matmul(user_sim.T.to_numpy(), user_rating) / (sim_sum+1)
    user_prediction_result_df.loc[isbn] = pred_ratings


# In[48]:


user_prediction_result_df = user_prediction_result_df.transpose()
user_prediction_result_df


# In[49]:


test_df.head()


# In[50]:


def evaluate(test_df, prediction_result_df):
    groups_with_isbn_ids = test_df.groupby(by='ISBN')
    groups_with_user_ids = test_df.groupby(by='UserID')
    intersection_isbn_ids = sorted(list(set(list(prediction_result_df.columns)).intersection(set(list(groups_with_isbn_ids.indices.keys())))))
    intersection_user_ids = sorted(list(set(list(prediction_result_df.index)).intersection(set(list(groups_with_user_ids.indices.keys())))))

    print(len(intersection_isbn_ids))
    print(len(intersection_user_ids))

    compressed_prediction_df = prediction_result_df.loc[intersection_user_ids][intersection_isbn_ids]

  # test_df에 대해서 RMSE 계산
    grouped = test_df.groupby(by='UserID')

    result_df = pd.DataFrame(columns=['rmse'])
    for userId, group in tqdm(grouped):
        if userId in intersection_user_ids:
            pred_ratings = compressed_prediction_df.loc[userId][compressed_prediction_df.loc[userId].index.intersection(list(group['ISBN'].values))]
            pred_ratings = pred_ratings.to_frame(name='Book-Rating').reset_index().rename(columns={'index':'ISBN','Book-Rating':'pred_rating'})
        
            actual_ratings = group[['Book-Rating', 'ISBN']].rename(columns={'Book-Rating':'actual_rating'})
          
            final_df = pd.merge(actual_ratings, pred_ratings, on='ISBN', how='inner')
            final_df = final_df.round(4) # 반올림

          #if not final_df.empty:
              # rmse = sqrt(mean_squared_error(final_df['actual_rating'], final_df['pred_rating']))
               #result_df.loc[userId] = rmse
               #print(userId, rmse)
  
    return final_df
  


# In[51]:


#evaluate(test_df, user_prediction_result_df)


# In[52]:


book_name = pd.read_csv(os.path.join(path, 'Books.csv'),delimiter=";", encoding='cp949',  error_bad_lines=False, names=['BookName'])

book_name['ISBN'] = book_name['BookName'].str.split(';').str[0]
book_name['title']= book_name['BookName'].str.split(';').str[1]
book_name = book_name.drop(columns=['BookName'], axis=1)
book_name['title'] = book_name['title'].str.replace('"','')
book_name.head()


# In[53]:


result_df = evaluate(test_df, user_prediction_result_df)
print(result_df)
print(f"RMSE: {sqrt(mean_squared_error(result_df['actual_rating'].values, result_df['pred_rating'].values))}")


# In[54]:


result_df_with_title = pd.merge(result_df, book_name, how='inner', on='ISBN')
result_df_with_title


# In[ ]:





# In[ ]:




