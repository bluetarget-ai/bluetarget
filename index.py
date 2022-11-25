import pandas as pd
import numpy as np
import uuid
import random

from bluetarget.monitor import Monitor
from bluetarget.entities import ColumnMapping


from sklearn.datasets import fetch_california_housing


monitor = Monitor(api_key="nuKS4WiD7aysMyCMu87CGV",
                  monitor_id="i4PoGQ7QZUX24PStfS9JgB",
                  version_id="v3")

data = fetch_california_housing(as_frame=True)
housing_data = data.frame


housing_data.rename(columns={'MedHouseVal': 'target'}, inplace=True)
housing_data['prediction'] = housing_data['target'].values + \
    np.random.normal(0, 5, housing_data.shape[0])


monitor.get_inference_dataset()

# monitor.add_reference_dataset(
#     housing_data, column_mapping=ColumnMapping(features=["MedInc", "AveRooms", "AveOccup", "AveBedrms", "Population", "Latitude", "HouseAge", "Longitude"], prediction="prediction", target="target"))

# reference = housing_data.sample(n=10000, replace=False)

# predictions = []

# count = 0

# for row in reference.iterrows():

#     value = row[1]['target']
#     features = dict(row[1].drop(['target', 'prediction']))

#     predictions.append({
#         "prediction_id": uuid.uuid4().hex,
#         "features": features,
#         "value": value
#     })
#     count += 1

#     if count >= 500:
#         print("################ LOG PREDICTIONS ########################")
#         monitor.log_predictions(monitor_id="i4PoGQ7QZUX24PStfS9JgB",
#                                 version_id="v3", predictions=predictions)

#         actuals = []

#         for prediction in predictions:
#             actual = prediction["value"] * random.random()

#             actuals.append(
#                 {
#                     "prediction_id": prediction["prediction_id"],
#                     "value": actual
#                 }
#             )
#         print("################ LOG ACTUALS ########################")

#         monitor.log_actuals(monitor_id="i4PoGQ7QZUX24PStfS9JgB",
#                             version_id="v3", actuals=actuals)

#         predictions = []
#         count = 0


# actuals = []


# for prediction in predictions:
#     actual = prediction["value"] * random.random()

#     actuals.append(
#         {
#             "prediction_id": prediction["prediction_id"],
#             "value": actual
#         }
#     )


# monitor.log_actuals(monitor_id="i4PoGQ7QZUX24PStfS9JgB",
#                     version_id="v3", actuals=actuals)
