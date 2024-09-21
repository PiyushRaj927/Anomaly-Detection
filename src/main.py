import numpy as np
import polars as pl
from sklearn.ensemble import IsolationForest
import hvplot.pandas  # for hvPlot to work on Polars DataFrame

np.random.seed(42)
data = np.random.randn(1000, 2)

# Add some anomalies
anomalies = np.random.uniform(low=-10, high=10, size=(20, 2))
data = np.vstack([data, anomalies])

df = pl.DataFrame(data, schema=["feature_1", "feature_2"])

# Isolation Forest for anomaly detection
iso_forest = IsolationForest(contamination=0.02, random_state=42)
df_numpy = df.to_numpy()

predictions = iso_forest.fit_predict(df_numpy)

df = df.with_columns([
    pl.Series(name="anomaly", values=predictions)
])

df = df.with_column(
    (df["anomaly"] == -1).alias("is_anomaly")
)

df = df.drop("anomaly")

df_pd = df.to_pandas()

# Plot using hvplot 
df_pd.hvplot.scatter(
    x='feature_1', 
    y='feature_2', 
    c='is_anomaly',  # Color based on anomaly detection
    cmap={True: 'red', False: 'blue'},  # Anomalies in red, normal points in blue
    title="Anomaly Detection with Isolation Forest",
    xlabel="Feature 1",
    ylabel="Feature 2",
    size=10,
    height=400,
    width=600
)
