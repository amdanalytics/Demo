# Run this locally to generate model.pkl
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle

iris = load_iris()
X, y = iris.data, iris.target
model = RandomForestClassifier()
model.fit(X, y)

with open('app/model.pkl', 'wb') as file:
    pickle.dump(model, file)