import joblib
import shap

# Load model and test data
X_test = joblib.load("X_test.joblib")
model = joblib.load("rf_model.joblib")

# Create SHAP explainer and get values for the first sample
explainer = shap.Explainer(model, X_test)
shap_values = explainer(X_test[:1])  # Get explanation object

# Manually create SHAP Explanation for class 1
shap_values_class1 = shap.Explanation(
    values=shap_values.values[0, :, 1],
    base_values=shap_values.base_values[0, 1],
    data=shap_values.data[0],
    feature_names=shap_values.feature_names
)

# Plot
shap.plots.waterfall(shap_values_class1)
