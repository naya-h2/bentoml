import requests
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# 데이터 준비
data = load_wine()
x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.05, random_state=11)
x_test_df = pd.DataFrame(x_test, columns=data.feature_names)

# print(x_test_df)

# JSON 데이터 생성
x_test_json = x_test_df.to_json(orient="records")

# print(x_test_json)

# API 요청 보내기
url_knn = "http://127.0.0.1:3002/predict_knn_model"  # BentoML 서비스 URL_knn
headers = {"Content-Type": "application/json"}
response1 = requests.post(url_knn, data=x_test_json, headers=headers)

# 결과 출력
print("knn --- ")
print(response1.json())

url_rf = "http://127.0.0.1:3002/predict_rf_model"
response2 = requests.post(url_rf, data=x_test_json, headers=headers)

print("rf --- ")
print(response2.json())

url_all = "http://127.0.0.1:3002/predict_all"
response3 = requests.post(url_all, data=x_test_json, headers=headers)

print("all --- ")
print(response3.json())