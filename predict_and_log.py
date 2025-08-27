import time, uuid, joblib, csv
from datetime import datetime

model = joblib.load("rf_model.joblib")
X_test = joblib.load("X_test.joblib")

for i in range(5):
    query = X_test[i].reshape(1, -1)
    start = time.time()
    y_pred = model.predict(query)
    y_proba = model.predict_proba(query)
    runtime = round(time.time() - start, 5)

    log_entry = [str(uuid.uuid4()), datetime.now(), int(y_pred[0]), y_proba.tolist(), query.shape, runtime]
    with open("predictions.log", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(log_entry)

print("✅ 5 Predictions logged to predictions.log")
