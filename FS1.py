import pandas as pd
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]

from sklearn.feature_selection import mutual_info_classif
importance = mutual_info_classif(X,Y)
feat_importance = pd.Series(importance, dataframe.columns[0: len(dataframe.columns)-1])
feat_importance.plot(kind='barh', color='teal')
