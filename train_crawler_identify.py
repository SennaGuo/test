
# coding: utf-8



from sklearn.metrics import mean_squared_error
from sklearn import metrics
import xgboost as xgb
import lightgbm as lgb
import pandas as pd
import numpy as np
import os
import gc


data=pd.read_csv('./data/crawler_train.txt',sep='\t')


feature=['clientippart0',
         'clientippart1',
         'clientippart2',
         'clientippart3',
         'islandingpage',
         'pageid',
         'paramlength',
         'referer',
         'servicecodehash',
         'sessionid',
         'updatetimehour',
         'useragenthash',
         'clientiphash',
         'hasgethotelrooms',
         'notnullpageid5min', 
         'notnullpageid5minrate',
         'pageid2minutes',
         'servicecode2minutes',
         'sessionid2minutes', 
         'timelag5minutes',
         'frequency2minutes',
         'hasfilter',
         'isfirstrequest', 
         'servicecoderepeat',
         'haslogin',
         'hasbookingcheckdata',
         'minreqinterval2min', 
         'minreqinterval5min',
         'maxreqinterval2min', 
         'maxreqinterval5min',
         'stddevpopreqinterval2min',
         'stddevpopreqinterval5min',
         'reqinterval25percentile2min',
         'reqinterval25percentile5min',
         'reqinterval50percentile2min',
         'reqinterval50percentile5min',
         'reqinterval75percentile2min',
         'reqinterval75percentile5min',
         'maxpercent', 
         'maxservicecodehash']



print(len(data[data.label==2]),len(data[data.label==0]),len(data[data.label==1]))

train = data[(data.d<='2019-08-03')]
val =data[(data.d=='2019-08-04')]
test =data[(data.d=='2019-08-05')]


train["y"]=None
train["y"][train.label==2]=np.zeros(len(train[train.label==2]))
train["y"][train.label!=2]=train[train.label!=2].label

val["y"]=None
val["y"][val.label==2]=np.zeros(len(val[val.label==2]))
val["y"][val.label!=2]=val[val.label!=2].label




#爬虫类型抽样策略##
#50%抽样
#train_crawler=pd.concat([train[train.label==2][np.random.rand(len(train[train.label==2]))<0.15],train[train.label==0][np.random.rand(len(train[train.label==0]))<0.5]])
#按原比例抽样
#train_crawler=train[train.label!=1][np.random.rand(len(train[train.label==2])+len(train[train.label==0]))<0.97]
#train2=pd.concat([train[train.label==1],train_crawler])
#train2["y"]=None
#train2["y"][train2.label==2]=np.zeros(len(train2[train2.label==2]))
#train2["y"][train2.label!=2]=train2[train2.label!=2].label

##smote抽样##
oversampler = SMOTE(ratio=0.2, random_state=np.random.randint(100), kind='regular', n_jobs=-1)
os_X_train, os_y_train = oversampler.fit_sample(X_train.fillna(0),y_train)

##ADASYN 运行起来很慢###
X_resampled_adasyn, y_resampled_adasyn = ADASYN(sampling_strategy=0.2,n_jobs =-1 ).fit_sample(train.loc[:,feature].fillna(0).values,train["y"].values.astype('int'))



###删除边界的一些噪声点###
from imblearn.under_sampling import EditedNearestNeighbours
enn = EditedNearestNeighbours(random_state=0)
X_resampled, y_resampled = enn.fit_sample(X, y)



dtrain=xgb.DMatrix(data=train.loc[:,feature].astype('float'), label=train['y'].astype('int'))
dval=xgb.DMatrix(data=val.loc[:,feature].astype('float'), label=val['y'].astype('int'))
train.loc[:,feature].info(null_counts=True)





params={
    'booster':'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth':6,
    'subsample':0.8,
    'colsample_bytree':0.8,
    'min_child_weight':2,
    'eta': 0.1,
    'seed':7,
    'nthread':-1,
    'silent':1
    }
watchlist = [(dtrain,'train'), (dval, 'val')]
gc.collect()

model = xgb.train( params,
                   dtrain,
                   num_boost_round=200,
                   early_stopping_rounds=10,
                   evals=watchlist
                 )
print("best best_ntree_limit", model.best_ntree_limit)


feature_importance_gain=pd.DataFrame(pd.Series(model.get_score(fmap='',importance_type='gain')),columns=['score']).sort_values(by='score',ascending=False)
feature_importance_weight=pd.DataFrame(pd.Series(model.get_score(fmap='',importance_type='weight')),columns=['score']).sort_values(by='score',ascending=False)
print(feature_importance_gain)
model.save_model("spider_v0805.model")
preds2 = model.predict(xgb.DMatrix(data=val.loc[:,feature].astype('float')))


##pr 曲线 ##
from sklearn.metrics import precision_recall_curve
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
plt.rcParams["figure.figsize"] = [12,6]
precision,recall,threshold = precision_recall_curve(val["label"],preds,pos_label=1)
prlist = {}
prlist['precision'] = precision
prlist['recall'] = recall
prlist['threshold'] = np.append(threshold,1)
prlist = pd.DataFrame(prlist,columns=['threshold','precision','recall'])
#plot the precision-recall
plt.figure(figsize=(4,4), dpi=120)
plt.plot(recall,precision)
plt.grid(True,ls = '--',which = 'both')
plt.xlabel("Recall")
plt.ylabel("Percision")
plt.show()

##结果##
recall_list=[1,3,5,10,20,30,40,50,60,70,80,90]
thres1=[]
thres2=[]
precision_list1=[]
precision_list2=[]
for recall in recall_list:
    t1=prlist.loc[np.round(prlist['recall']*100)==recall].max().loc['threshold']
    p1=prlist.loc[np.round(prlist['recall']*100)==recall].max().loc['precision']
      
    precision_list1.append(p1)

    thres1.append(t1)
recall_precision_df=pd.concat([pd.DataFrame({'thres1':thres1})
  ,pd.DataFrame({'recall':recall_list})
  ,pd.DataFrame({'precision_test':precision_list1})],axis=1)
print(recall_precision_df)
fpr, tpr, thresholds = metrics.roc_curve(val["y"].astype("int").values, preds)
auc=metrics.auc(fpr, tpr)




##heat matrix##
import seaborn as sns
sns.set(style="white")
corr = train.loc[:,feature].fillna(0).corr()
mask = np.zeros_like(corr, dtype=np.bool)
f, ax = plt.subplots(figsize=(20, 11))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}
           )


##feature selection## lasso 
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
X, y = train.loc[:,feature].fillna(0),  train.loc[:,[label]]
X.shape

lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X)
X_new.shape


##筛选结果##
feature_selections=pd.DataFrame({"feature":feature,"is_selected":model.get_support()})
feature_s=feature_selections[feature_selections.is_selected==True].feature.values
dtrain=xgb.DMatrix(data=train.loc[:,feature_s].astype('float'), label=train['y'].astype('int'))
dval=xgb.DMatrix(data=val.loc[:,feature_s].astype('float'), label=val['y'].astype('int'))
dtest=xgb.DMatrix(data=test.loc[:,feature_s].astype('float'), label=test['y'].astype('int'))


##相似度筛选##
corr = train.loc[:,feature].fillna(0).corr()
selection=[]
#计算变量间的相关系数#
for i in range(len(feature)):
    selection.append([feature[i],corr.loc[:,[feature[i]]][(np.abs(corr.loc[:,[feature[i]]].values)>0.8)&(corr.loc[:,[feature[i]]].values!=1)].index])


##调参用sklearn的api##


###预测 ####
test=pd.read_csv("./data/crawler_test.txt",sep='\t')
test.head()
dtest=xgb.DMatrix(data=test.loc[:,feature].astype('float'))
preds=model.predict(dtest)
dfff=pd.concat([test.loc[:,["clientid","updatetime","label","score"]],pd.DataFrame(preds,columns=["pred"])],axis=1)
dfff
dfff.to_csv("crawler_0708prediction.csv",index=False,header=False )

###测试模型文件线上线下是否一致##
df_test=pd.DataFrame({"allianceid":["na"]
,"avginterval2minutes":[799.4]
,"avginterval5minutes":[813.13336]
,"clientid":["09031172210287250176"]
,"clientip2minutes":[1]
,"clientiphash":[124080]
,"clientippart0":[10]
,"clientippart1":[32]
,"clientippart2":[151]
,"clientippart3":[3]
,"frequency10minutes":[51]
,"frequency2minutes":[10]
,"hasbookingcheckdata":[1]
,"hasfilter":[0]
,"hasgethotelrooms":[1]
,"haslogin":[0]
,"interval":[3213]
,"interval25percentile2minutes":[73]
,"interval25percentile5minutes":[73]
,"interval50percentile2minutes":[178]
,"interval50percentile5minutes":[178]
,"interval75percentile2minutes":[474]
,"interval75percentile5minutes":[561]
,"intervaltimestampstd10min":[155784.3]
,"intervaltimestampstd2min":[2434.537]
,"intervaltimestampstd5min":[10902.005]
,"intervaltimestampstd5minpertime":[726.80035]
,"isfirstrequest":[0]
,"islandingpage":[0]
,"maxinterval2minutes":[3232]
,"maxinterval5minutes":[3244]
,"maxpercent":[0.200000]
,"maxreqinterval2min":[2674]
,"maxreqinterval5min":[16459]
,"maxservicecodehash":[15834]
,"messageid":["190725.093646.10.2.7.142.21304.1"]
,"mininterval2minutes":[69]
,"mininterval5minutes":[69]
,"minreqinterval2min":[26]
,"minreqinterval5min":[26]
,"notnullpageid5min":[15]
,"notnullpageid5minrate":[1]
,"pageid":[1]
,"pageid10minutes":[4]
,"pageid2minutes":[1]
,"paramhash":[152229]
,"paramlength":[7]
,"processtime":[1564018605918]
,"referer":[1]
,"refererhash":[114361]
,"reqinterval25percentile2min":[441]
,"reqinterval25percentile5min":[441]
,"reqinterval50percentile2min":[679]
,"reqinterval50percentile5min":[732]
,"reqinterval75percentile2min":[889]
,"reqinterval75percentile5min":[2268]
,"requesturlhash":[205484]
,"score":[0.23050553]
,"servicecode10minutes":[9]
,"servicecode2minutes":[5]
,"servicecode2minutespatial10minutes":[0]
,"servicecodehash":[15834]
,"servicecoderepeat":[1.0]
,"sessionid":[1]
,"sessionid10minutes":[1]
,"sessionid2minutes":[1]
,"sid":["na"]
,"stddevpopinterval10minutes":[1006.37286]
,"stddevpopinterval2minutes":[1219.7711]
,"stddevpopinterval5minutes":[1218.2521]
,"stddevpopreqinterval10min":[0.0]
,"stddevpopreqinterval2min":[882.45087]
,"stddevpopreqinterval5min":[4084.967]
,"stormprocesstime":[1564018564054]
,"timelag10minutes":[369509]
,"timelag5minutes":[28930]
,"timestamp":[1564018560789]
,"updatetimehour":[9]
,"updatetimeminute":[36]
,"updatetimesecond":[0]
,"useragenthash":[169253]
,"useragentminutes":[1]})

dtest=xgb.DMatrix(data=df_test.loc[:,feature].astype('float'))
model.predict(dtest)


