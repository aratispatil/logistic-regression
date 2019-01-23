# logistic-regression
the problem is based on predicting insurance brought based on age according to the dataset
logistic regression

data1=pd.read_csv('C:/Users/ARTI/Downloads/insurance_data.csv')
data1.head()
plt.scatter(data1.age,data1.bought_insurance)

from sklearn.linear_model import LogisticRegression

model=LogisticRegression()
model.fit(data1[['age']],data1.bought_insurance)

model.predict(32)

model.predict_proba(20)

model.score(data1[['age']],data1.bought_insurance)
