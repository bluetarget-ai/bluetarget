import uuid
from bluetarget import Monitor, ModelSchema, ModelSchemaVersion, MonitorPredictionType, MonitorSchemaType, Prediction, PredictionActual
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
iris_frame = pd.DataFrame(iris.data, columns=iris.feature_names)
X = iris.data
y = np.array([iris.target_names[i] for i in iris.target])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model Training
clf = svm.SVC(gamma='scale', kernel='rbf', probability=True)
clf.fit(X, y)

prediction_probabilities = list(clf.predict_proba(X))
prediction = list(clf.predict(X))


monitor = Monitor(api_key="nuKS4WiD7aysMyCMu87CGV")

id = uuid.uuid4().hex

# Create monitor
monitor.create(
    ModelSchema(
        monitorId=id,
        name="Iris sklearn",
        description="sklearn model, rbf kernel",
        predictionType=MonitorPredictionType.CATEGORICAL,
    )
)

# Create a version of the monitor with the model schema
monitor.create_version(id, ModelSchemaVersion(
    versionId="v1",
    model_schema={
        "sepal_length": MonitorSchemaType.FLOAT,
        "sepal_width": MonitorSchemaType.FLOAT,
        "petal_length": MonitorSchemaType.FLOAT,
        "petal_width": MonitorSchemaType.FLOAT
    }
))

feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

for i in range(150):

    features = {feature_names[j]: float(X[i][j]) for j in range(4)}

    probabilities = {iris.target_names[j]: float(
        prediction_probabilities[i][j]) for j in range(3)}

    prediction_id = uuid.uuid4().hex

    monitor.log_predictions([
        Prediction(
            prediction_id=prediction_id,
            features=features,
            value=prediction[i],
            probabilities=probabilities
        )
    ])

    monitor.log_actuals(actuals=[PredictionActual(
        prediction_id=prediction_id,
        actual=y[i]
    )])


# prediction_id = uuid.uuid4().hex,
# features = features,
# value = prediction[i],
# probabilities = iris.target_names[j]: float(prediction_probabilities[i][j])
# for j in range(3)
