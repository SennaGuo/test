#! /usr/bin/python

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import xgboost as xgb
print("XGBoost Version: ",xgb.__version__)
from xgboost import XGBRegressor as xgb
from sklearn.metrics import mean_squared_error as mse,median_absolute_error as mae
from sklearn.externals import joblib

####################Step1.读数据####################
data = pd.read_csv('/home/hotelbi/stan/room/Train.csv',sep=',')
print(data.shape)
print(list(data.columns))
print(dict(data.dtypes))
data.head(5)
####################Step2.数据清洗####################
##查看各列缺失值##
Nan_number = []
Nan_ratio = []
for name in data.columns:
    Nan_number.append(data[name].isnull().sum())    
    Nan_ratio.append((data[name].isnull().sum()) / len(data))
Nan_info = pd.concat([pd.Series(data.columns),pd.Series(Nan_number),pd.Series(Nan_ratio)],axis=1)
Nan_info.columns = ['Column_name','Nan_number','Nan_ratio']
Nan_info.sort_values(by='Nan_ratio',ascending=False).loc[Nan_info['Nan_number']>0].reset_index(drop=True)

##缺失值处理##
data.dropna(inplace=True)

##转换数据类型##
price_features = data.columns[data.columns.str.contains('price')]
data[price_features] = data[price_features].astype('int64')

##判断异常值标准，其中K=1.5表示中度异常，K=3表示极度异常##
# Q1 - K(Q3 - Q1)
# Q3 + K(Q3 - Q1)
Q1 = np.percentile(data['price_label_30'],25)
Q3 = np.percentile(data['price_label_30'],75)
#K=1.5，中度异常
Lower_value_low = round(Q1 - 1.5*(Q3 - Q1))
High_value_low = round(Q3 + 1.5*(Q3 - Q1))
#K=3，极度异常
Lower_value_high = round(Q1 - 3*(Q3 - Q1))
High_value_high = round(Q3 + 3*(Q3 - Q1))
print("中度异常值下限和上限分别为：L: {}, H: {}".format(Lower_value_low,High_value_low))
print("极度异常值下限和上限分别为：L: {}, H: {}".format(Lower_value_high,High_value_high))
print('=====================================================================================')
print("房价高于中度异常上限的记录数占比为：",len(data[data['price_label_30'] >= High_value_low]) / len(data))
print("房价高于极度异常上限的记录数占比为：",len(data[data['price_label_30'] >= High_value_high]) / len(data))
print('=====================================================================================')
print("房价高于中度异常上限的记录数为：",len(data[data['price_label_30'] >= High_value_low]))
print("房价高于极度异常上限的记录数为：",len(data[data['price_label_30'] >= High_value_high]))
print("房价高于中度异常上限的母基记录数为：",len(data[data['price_label_30'] >= High_value_low]['masterroomid'].unique()))
print("房价高于极度异常上限的母基记录数为：",len(data[data['price_label_30'] >= High_value_high]['masterroomid'].unique()))

##查看母酒店物理属性缺失情况##
np.sum(data['goldstar'].isnull()     | data['operation_length'].isnull()   | data['customereval'].isnull()   |
       data['recommend'].isnull()    | data['room_score'].isnull()         | data['isfamilystay'].isnull()   |
       data['hlevel'].isnull()       | data['novoters'].isnull()           | data['novoters_level'].isnull() | 
       data['roomquantity'].isnull() | data['roomquantity_level'].isnull() | data['star'].isnull()           |
       data['star_level'].isnull())
	   
##查看入住日记录##
data['arrival'].value_counts(normalize=True)
####################Step3.划分数据集####################
##划分数据集：训练集&验证集&测试集##
d_train = data[data.arrival<='2018-11-07']
d_val = data[(data.arrival>='2018-11-08') & (data.arrival<='2018-11-12')]
d_test = data[(data.arrival>='2018-11-13') & (data.arrival<='2018-11-15')]
print("训练集大小为： {}\n".format(d_train.shape))
print("验证集大小为： {}\n".format(d_val.shape))
print("测试集大小为： {}".format(d_test.shape))

##模型特征和标签##
features = ['masterroomid',
 'masterhotelid',
   'goldstar',
   'operation_length',
  'customereval',
  'recommend',
  'room_score',
  'isfamilystay',
  'hlevel',
  'novoters',
  'novoters_level',
  'roomquantity',
#  'roomquantity_level',
  'star',
#  'star_level',
 'arrival',
 'ord_sum_7d',
 'ord_sum_14d',
 'ord_sum_30d',
 'ord_sum_60d',
 'ord_sum_90d',
 'ord_sum_180d',
 'price_avg_7d',
 'price_avg_14d',
 'price_avg_30d',
 'price_avg_60d',
 'price_avg_90d',
 'price_avg_180d',
 'price_max_7d',
 'price_max_14d',
 'price_max_30d',
 'price_max_60d',
 'price_max_90d',
 'price_max_180d',
 'price_min_7d',
 'price_min_14d',
 'price_min_30d',
 'price_min_60d',
 'price_min_90d',
 'price_min_180d',
 'price_percentile_25_7d',
 'price_percentile_25_14d',
 'price_percentile_25_30d',
 'price_percentile_25_60d',
 'price_percentile_25_90d',
 'price_percentile_25_180d',
 'price_percentile_50_7d',
 'price_percentile_50_14d',
 'price_percentile_50_30d',
 'price_percentile_50_60d',
 'price_percentile_50_90d',
 'price_percentile_50_180d',
 'price_percentile_75_7d',
 'price_percentile_75_14d',
 'price_percentile_75_30d',
 'price_percentile_75_60d',
 'price_percentile_75_90d',
 'price_percentile_75_180d'
#  'holiday_n30_future_days',
#  'is_contain_fholiday',
#  'holiday_n30_past_days',
#  'is_contain_pholiday'  
]
##预测未来30天均值##
label = [
    'price_label_30'
# ,'price_label_60',
#  'price_label_90'
]

##训练集:##
dtrain_low = d_train.loc[d_train['price_label_30'] < High_value_low, features]
ltrain_low = d_train.loc[d_train['price_label_30'] < High_value_low, label]
print('训练数据维度：特征{}, 标签{}'.format(dtrain_low.shape,ltrain_low.shape))
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
##验证集:##
dval_low = d_val.loc[d_val['price_label_30'] < High_value_low, features]
lval_low = d_val.loc[d_val['price_label_30'] < High_value_low, label]
print('验证数据维度：特征{}, 标签{}'.format(dval_low.shape,lval_low.shape))
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
##测试集:##
dtest_low = d_test.loc[d_test['price_label_30'] < High_value_low, features]
ltest_low = d_test.loc[d_test['price_label_30'] < High_value_low, label]
print('测试数据维度：特征{}, 标签{}'.format(dtest_low.shape,ltest_low.shape))

####################Step4.模型训练&预测####################
##第一版模型：GPU训练##
XGB_model_low = xgb(seed=1234,nthread=-1,n_estimators=200 ,max_depth=6,min_child_weight=1,subsample=0.95,silent=False)
import time
print("训练开始======================================================")
print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
XGB_model_low.fit(dtrain_low.drop(['masterroomid','arrival','masterhotelid'],axis=1),ltrain_low,eval_set=[(dtrain_low.drop(['masterroomid','arrival','masterhotelid'],axis=1),ltrain_low)],eval_metric='mae')
##print("训练结束======================================================")
##print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))

##保存训练模型##
joblib.dump(XGB_model_low,'/home/hotelbi/stan/room/model_room_alpha.m')
print(dval_low.shape)