import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
from xgboost import plot_importance
from sklearn.preprocessing import Imputer

def loadDataset(filePath):
    df = pd.read_csv(filepath_or_buffer=filePath)
    return df


def featureSet(data):
    data_num = len(data)
    XList = []
    for row in range(0, data_num):
        tmp_list = []
        tmp_list.append(data.iloc[row]['feature_1'])
        tmp_list.append(data.iloc[row]['feature_2'])
        tmp_list.append(data.iloc[row]['feature_3'])
        XList.append(tmp_list)
    yList = data.target.values
    return XList, yList


def loadTestData(filePath):
    data = pd.read_csv(filepath_or_buffer=filePath)
    data_num = len(data)
    XList = []
    for row in range(0, data_num):
        tmp_list = []
        tmp_list.append(data.iloc[row]['feature_1'])
        tmp_list.append(data.iloc[row]['feature_2'])
        tmp_list.append(data.iloc[row]['feature_3'])
        XList.append(tmp_list)
    return XList
trainFilePath = '/Users/shiyuan/Desktop/essaydata/kaggle-recommendation/all/train.csv'
testFilePath = '/Users/shiyuan/Desktop/essaydata/kaggle-recommendation/all/test.csv'
data = loadDataset(trainFilePath)
X_train, y_train = featureSet(data)
X_test = loadTestData(testFilePath)

model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, silent=False, objective='reg:linear')
model.fit(X_train, y_train)

plot_importance(model)
plt.show()

ans = model.predict(X_test)
ans = ans.reshape((123623, 1))

df_submit = pd.read_csv('/Users/shiyuan/Desktop/essaydata/kaggle-recommendation/all/sample_submission.csv')
df_submit.target = ans
df_submit.to_csv('/Users/shiyuan/Desktop/submit.csv', index=None)