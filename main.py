import pandas as pd
import bentoml
from bentoml.io import PandasDataFrame
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine

# 아까 저장한 모델 불러오기
knn_model = bentoml.sklearn.get("knn_model")
rf_model = bentoml.sklearn.get("rf_model")

# sklearn 모델은 따로 runner로 감싸줘야 한다 !!!
knn_runner = knn_model.to_runner()
rf_runner = rf_model.to_runner()

# BentoML 서비스 정의
svc = bentoml.Service("wine_classifier", runners=[knn_runner, rf_runner])

@svc.api(input=PandasDataFrame(), output=PandasDataFrame())
async def predict_knn_model(df: pd.DataFrame):
    # knn 모델 예측
    # print(df)

    prediction = await knn_runner.async_run(df)
    return pd.DataFrame(prediction, columns=["prediction"])


@svc.api(input=PandasDataFrame(), output=PandasDataFrame())
async def predict_rf_model(df: pd.DataFrame):
    # rf 모델 예측
    prediction = await rf_runner.async_run(df)
    return pd.DataFrame(prediction, columns=["prediction"])
