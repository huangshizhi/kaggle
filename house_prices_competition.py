# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 09:54:45 2017

@author: huangshizhi

https://www.dataquest.io/blog/kaggle-getting-started/
"""

import pandas as pd
import numpy as np

#1.加载数据
train = pd.read_csv(r'D:\kaggle\house_prices\data\train.csv')
test = pd.read_csv(r'D:\kaggle\house_prices\data\test.csv')

import matplotlib.pyplot as plt
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

#2.探究数据，查看数据的统计特征，skew为分布的不对称度
train.SalePrice.describe()
print ("Skew is:", train.SalePrice.skew())
plt.hist(train.SalePrice, color='blue')
plt.show()


target = np.log(train.SalePrice)
print ("Skew is:", target.skew())
plt.hist(target, color='blue')
plt.show()

#抽取数值型变量特征，Working with Numeric Features
numeric_features = train.select_dtypes(include=[np.number])
numeric_features.dtypes

#计算协方差矩阵
corr = numeric_features.corr()


print (corr['SalePrice'].sort_values(ascending=False)[:5], '\n')
print (corr['SalePrice'].sort_values(ascending=False)[-5:])

salePrice_top5 = corr['SalePrice'].sort_values(ascending=False)[:5]

salePrice_bottom5 = corr['SalePrice'].sort_values(ascending=False)[-5:]


#对房子的整体材料和成品率进行评估
train.OverallQual.unique()

quality_pivot = train.pivot_table(index='OverallQual',
                                  values='SalePrice', aggfunc=np.median)
                                  
quality_pivot.plot(kind='bar', color='blue')
plt.xlabel('Overall Quality')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()       

#居住面积
plt.scatter(x=train['GrLivArea'], y=target)
plt.ylabel('Sale Price')
plt.xlabel('Above grade (ground) living area square feet')
plt.show()
     
#车库大小
plt.scatter(x=train['GarageArea'], y=target)
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
plt.show()        

#去掉异常值之后
train = train[train['GarageArea'] < 1200]      

plt.scatter(x=train['GarageArea'], y=np.log(train.SalePrice))
plt.xlim(-200,1600) # This forces the same scale as before
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
plt.show()


#Handling Null Values
nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False))
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'

#不包括其他类别的杂项特性
print ("Unique values are:", train.MiscFeature.unique())

#抽取非数值型变量特征
categoricals = train.select_dtypes(exclude=[np.number])
cate_desc = categoricals.describe()

print ("Original: \n") 
print (train.Street.value_counts(), "\n")

#One-Hot Encoding 
train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
test['enc_street'] = pd.get_dummies(train.Street, drop_first=True)

print ('Encoded: \n') 
print (train.enc_street.value_counts())

condition_pivot = train.pivot_table(index='SaleCondition',
                                    values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()

def encode(x): return 1 if x == 'Partial' else 0
train['enc_condition'] = train.SaleCondition.apply(encode)
test['enc_condition'] = test.SaleCondition.apply(encode)

condition_pivot = train.pivot_table(index='enc_condition', values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Encoded Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()

#填充缺失值
data = train.select_dtypes(include=[np.number]).interpolate().dropna() 
#测试
sum(data.isnull().sum() != 0)

#3.建立回归模型，训练数据
y = np.log(train.SalePrice)
X = data.drop(['SalePrice', 'Id'], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, random_state=42, test_size=.33)
 
#建模                                   
from sklearn import linear_model
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)

print ("R^2 is: \n", model.score(X_test, y_test))

predictions = model.predict(X_test)


from sklearn.metrics import mean_squared_error
print ('RMSE is: \n', mean_squared_error(y_test, predictions))


actual_values = y_test
plt.scatter(predictions, actual_values, alpha=.75,color='b') 

#plt.plot(X, y_rbf, color='black', lw=lw, label='RBF model')

#alpha helps to show overlapping data
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Linear Regression Model')
plt.show()



for i in range (-2, 3):
    alpha = 10**i
    rm = linear_model.Ridge(alpha=alpha)
    ridge_model = rm.fit(X_train, y_train)
    preds_ridge = ridge_model.predict(X_test)
    
    plt.scatter(preds_ridge, actual_values, alpha=.75, color='b')
    plt.xlabel('Predicted Price')
    plt.ylabel('Actual Price')
    plt.title('Ridge Regularization with alpha = {}'.format(alpha))
    overlay = 'R^2 is: {}\nRMSE is: {}'.format(
                    ridge_model.score(X_test, y_test),
                    mean_squared_error(y_test, preds_ridge))
    plt.annotate(s=overlay,xy=(12.1,10.6),size='x-large')
    plt.show()


#4.提交结果
submission = pd.DataFrame()
submission['Id'] = test.Id

feats = test.select_dtypes(
        include=[np.number]).drop(['Id'], axis=1).interpolate()

predictions = model.predict(feats)
final_predictions = np.exp(predictions)
submission['SalePrice'] = final_predictions
submission.head()
        
submission.to_csv('D:\kaggle\house_prices\submission1.csv', index=False)
        
