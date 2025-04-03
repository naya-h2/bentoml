import requests
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# 데이터 준비
data = load_wine()
x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=11)
x_test_df = pd.DataFrame(x_test, columns=data.feature_names)

# print(x_test_df)

# JSON 데이터 생성
x_test_json = x_test_df.to_json(orient="records")

# print(x_test_json)

# API 요청 보내기
url = "http://127.0.0.1:3000/predict_knn_model"  # BentoML 서비스 URL
headers = {"Content-Type": "application/json"}
response = requests.post(url, data=x_test_json, headers=headers)

# 결과 출력
print(response.status_code)
print(response.json())