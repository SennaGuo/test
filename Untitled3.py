
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from xgboost.sklearn import XGBRegressor
from xgboost import plot_importance
from sklearn import metrics

data = pd.read_csv("F:/MyDownloads/data_kesi/train_kpi.csv",sep=',')
predict_data = pd.read_csv("F:/MyDownloads/data_kesi/test_kpi.csv",sep=',')




data_train = data.loc[data['d'] != '2018-04-29',:]
data_test = data.loc[data['d']=='2018-04-29',:]
data_train = data.fillna(value=0)
data_test = data.fillna(value=0)



data_train['d'] = pd.to_datetime(data_train['d'])
data_train['year'] = data_train['d'].dt.year
data_train['month'] = data_train['d'].dt.month
data_train['day'] = data_train['d'].dt.day
data_train['week'] = data_train['d'].dt.weekday
data_train['weekday'] = data_train['week'].apply(lambda x: 1 if x>=5 else 0)




data_test['d'] = pd.to_datetime(data_test['d'])
data_test['year'] = data_test['d'].dt.year
data_test['month'] = data_test['d'].dt.month
data_test['day'] = data_test['d'].dt.day
data_test['week'] = data_test['d'].dt.weekday
data_test['weekday'] = data_test['week'].apply(lambda x: 1 if x>=5 else 0)




data_train1 = data_train.drop(['masterhotelid','d','quantity_dc_45days'],axis=1)
data_train_label = data_train['quantity_dc_45days']

data_test1 = data_test.drop(['masterhotelid','d','quantity_dc_45days'],axis=1)
data_test_label = data_test['quantity_dc_45days']



model = XGBRegressor(seed=1000,base_score=0.5,
            colsample_bylevel=0.7,
            colsample_bytree=0.9,
            gamma=0,
            learning_rate=0.1,
            max_delta_step=0,
            max_depth=5,
            missing=None,
            nthread=-1,
            objective='reg:linear',
            reg_alpha=0,
            reg_lambda=1,
            silent=True,
            subsample=0.9)


model.fit(data_train1,data_train_label)
data_predict = model.predict(test_data11)

r2_score = metrics.r2_score(data_test_label.values,data_predict)
mse = metrics.mean_squared_error(data_test_label.values,data_predict)




print('R^2分数为: %.4f' % r2_score)
print('均方根误差为: %.4f' % mse)

