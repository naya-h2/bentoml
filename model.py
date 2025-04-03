from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

import bentoml

# 데이터셋 가져오기
data = load_wine()
# data.target[[10, 80, 140]]
# print(list(data.target_names))

# training set, test set 만들기
x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=11)

# 모델1. KNN
knn_model = KNeighborsClassifier(n_neighbors=8)
knn_model.fit(x_train,y_train)

# 모델2. RF
rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)

# 모델 bentoml에 저장
bentoml.sklearn.save_model("knn_model", knn_model)
bentoml.sklearn.save_model("rf_model", rf_model)