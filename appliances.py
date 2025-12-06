import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score,confusion_matrix,r2_score,f1_score


appliance_df = pd.read_csv(r"/Appliance/Appliances/energydata_complete.csv")
appliance_df.drop(['rv1','rv2'],inplace=True,axis=1)
appliance_df['date'] = pd.to_datetime(appliance_df['date'])
appliance_df['hour'] = appliance_df['date'].dt.hour
x=1
room_tem_col = []
room_Rh = []
for i in appliance_df.columns:
    if 'T' + str(x) in i:
        room_tem_col.append(i)
    elif 'RH_' + str(x) in i:
        room_Rh.append(i)
        x=x+1
appliance_df['T_mean'] = np.mean((appliance_df[room_tem_col]),axis=1)
appliance_df['R_mean'] = np.mean((appliance_df[room_Rh]),axis=1)
#-------------------- checked how appliance usage with hour in a day-------------------------------------------
# appliance_group_hour = appliance_df['Appliances'].groupby(by=appliance_df['hour']).mean().reset_index()
# appliance_group_hour['appliance_log'] = np.log(appliance_group_hour['Appliances'])
# print(appliance_group_hour)
# plt.plot(appliance_group_hour['hour'], appliance_group_hour['appliance_log'])
# plt.show()

#----------------- check through graph between mean of each room temperature in a house with target
# appliance_df['T_mean'] = np.round(appliance_df['T_mean'],decimals=0)
# grouping_dataset = appliance_df['Appliances'].groupby(by=appliance_df['T_mean']).mean().reset_index()
# print(grouping_dataset)
# plt.plot(grouping_dataset['T_mean'], (grouping_dataset['Appliances']))
# plt.show()

#----------------- check through graph between mean of each room presure in a house with target
# appliance_df['R_mean'] = np.round(appliance_df['R_mean'],decimals=0)
# grouping_dataset = appliance_df['Appliances'].groupby(by=appliance_df['R_mean']).mean().reset_index()
# print(grouping_dataset)
# plt.plot(grouping_dataset['R_mean'], (grouping_dataset['Appliances']))
# plt.show()

#----------------- check through graph between mean of each room presure in a house with mean room temperature
# appliance_df[['T_out','RH_out']] = np.round(appliance_df[['T_out','RH_out']],decimals=0)
# grouping_dataset = appliance_df['RH_out'].groupby(by=appliance_df['T_out']).mean().reset_index()
# plt.plot(np.log(grouping_dataset['T_out']), np.log((grouping_dataset['RH_out'])))
# plt.show()

#----------------- check mean humidity with time hour and
# appliance_df[['T_out','RH_out']] = np.round(appliance_df[['T_out','RH_out']],decimals=0)
# grouping_dataset1 = appliance_df['R_mean'].groupby(by=appliance_df['hour']).mean().reset_index()
# grouping_dataset2 = appliance_df['T_mean'].groupby(by=appliance_df['hour']).mean().reset_index()
# grouping_dataset3 = appliance_df['Appliances'].groupby(by=appliance_df['hour']).mean().reset_index()
#
# plt.plot((grouping_dataset1['hour']), np.log((grouping_dataset1['R_mean'])))
# plt.plot((grouping_dataset2['hour']), np.log ((grouping_dataset2['T_mean'])))
# plt.plot((grouping_dataset3['hour']), np.log((grouping_dataset3['Appliances'])))
# plt.legend(('RH','T','App'))
# plt.show()


#------------Use singular value decomposition to find the collinearity between independent features -------------------
Model_appliance_df = appliance_df.copy()
Model_appliance_df.drop(['T_mean','R_mean'],axis=1,inplace=True)
Model_appliance_df['Minutes'] = Model_appliance_df['date'].dt.minute
Model_appliance_df['Day'] = Model_appliance_df['date'].dt.day
Model_appliance_df['Month'] = Model_appliance_df['date'].dt.month
Model_appliance_df['year'] = Model_appliance_df['date'].dt.year
Model_appliance_df.sort_values('date').reset_index()
Model_appliance_df.drop(columns=['date'],inplace=True,axis=1)
X_Feature = Model_appliance_df.drop(columns=['Appliances','lights','year'],axis=1)
Y_features = Model_appliance_df[['Appliances','lights']]

# X_Train,X_Test,Y_Train,Y_Test = train_test_split(X_Feature,Y_features,test_size=0.2,random_state=False)

train_Feature_idx = int(len(X_Feature)*0.8)
X_Train = X_Feature[:train_Feature_idx]
Y_Train = Y_features[:train_Feature_idx]
X_Test = X_Feature[train_Feature_idx:len(X_Feature)]
Y_Test = Y_features[train_Feature_idx:len(Y_features)]


Pipeline = Pipeline(
    [
        ('scale',StandardScaler()),
        ('model',LinearRegression())
    ]
)

X_pipe_CV = TimeSeriesSplit(n_splits=5)

for x_train_idx,x_test_idx in X_pipe_CV.split(X_Train):
    X_train_CV,X_test_CV = X_Train.iloc[x_train_idx], X_Train.iloc[x_test_idx]
    Y_train_CV,Y_test_CV = Y_Train.iloc[x_train_idx], Y_Train.iloc[x_test_idx]
    Pipeline.fit(X_train_CV,Y_train_CV) # by doing this way we can't
    Y_predict = Pipeline.predict(X_test_CV)
    print(r2_score(Y_test_CV,Y_predict))

    # print(r2_score(Y_test_CV,round(Y_Predict)))
    # print(f1_score(Y_test_CV['Appliances'],round(Y_Predict['Appliances'])))







# Y_Feature = Model_appliance_df['Appliances','light']
# print(X_Feature.head(10).to_string())


# print(X_Feature)


# print(Model_appliance_df.head(10).to_string())



#---------------find correlation

# print(appliance_df.corr().to_string())
# sns.heatmap(appliance_df.corr())
# plt.show()
#
#
# print((appliance_df[room_tem_col].values)[1])
# appliance_df['T_mean'] = np.mean((appliance_df[room_tem_col]),axis=1)

# print(appliance_df['Appliances'].cov(appliance_df[room_tem_col]))






# x_log = np.log(appliance_df['T_mean'])








# appliance_df['mean_T'] = appliance_df[]

# print(appliance_df.head(10).to_string())
# Y = appliance_df['Appliances']


# for feature in appliance_df.columns:
#     if feature not in ['date','Appliances']:
#         X=appliance_df[feature]
#         X = np.mean(X)
#         plt.plot(X,Y)
#         plt.title(feature + ' vs ' + 'Appliances')
#         plt.show()



# for feature in appliance_df.columns:
#     values = appliance_df.loc[appliance_df[feature].isnull()].sum()
#     print(values)