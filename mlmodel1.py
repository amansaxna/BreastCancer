
'''
This is a simple linear regression model to predit the CO2 emmission from cars
Dataset:
FuelConsumption.csv, which contains model-specific fuel consumption ratings and estimated carbon dioxide emissions
for new light-duty vehicles for retail sale in Canada
'''

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pickle

from sklearn.metrics import f1_score,confusion_matrix

df = pd.read_csv("data/wisconsin.csv", )

df = df.drop(columns=["id"])

label = "diagnosis"

def min_max_scale(column):
    if column.name == label:
        return column
    else:
        old_values = column.values.reshape(-1, 1)
        new_values = MinMaxScaler().fit_transform(old_values)
        new_values = new_values.reshape(-1)
        return new_values

df = df.apply(min_max_scale)

drop_list1 = ['perimeter_mean','radius_mean','compactness_mean','concave points_mean','radius_se','perimeter_se','radius_worst','perimeter_worst','compactness_worst','concave points_worst','compactness_se','concave points_se','texture_worst','area_worst']
df = df.drop(drop_list1,axis = 1 )

df_train, df_test = train_test_split(df, test_size=0.3, stratify=df[label], random_state=42)

X_train = df_train.loc[:, df_train.columns != label]
y_train = df_train.loc[:, df_train.columns == label]

X_test = df_test.loc[:, df_test.columns != label]
y_test = df_test.loc[:, df_test.columns == label]

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
select_feature = SelectKBest(chi2, k=5).fit(X_train, y_train)
X_train = select_feature.transform(X_train)
X_test = select_feature.transform(X_test)

svc_obj = SVC()
svc_params = {
    'kernel':('linear', 'rbf', 'sigmoid'),
    'C':[i for i in range(1, 11)]
}

svc_gs_clf = GridSearchCV(svc_obj, svc_params)

svc_gs_clf.fit(X_train, y_train.values.reshape(-1))

# Saving model to disk
# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
pickle.dump(svc_gs_clf, open('model.pkl','wb'))
'''
#Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2.6, 8, 10.1,5.5,6.3]]))
'''
