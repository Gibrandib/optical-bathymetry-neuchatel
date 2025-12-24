import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import mean_squared_error


# 1) Load Neuchâtel dataset

df = pd.read_csv("L8_2020_dataset.csv")


df = df.rename(columns={"Lac_Neuchatel_MNT_Littoral_Bathy_25cm": "Depth"})
df["Depth"] = df["Depth"].abs()
df = df.dropna(subset=["Depth"])



# 2a) Physically consistent cleaning function 

def clean_bathy_dataset(df):
    df = df.copy()
    eps = 1e-6

    if {"NIR", "Red"}.issubset(df.columns):
        df["NDVI"] = (df["NIR"] - df["Red"]) / (df["NIR"] + df["Red"] + eps)
        df = df[df["NDVI"] < 0.2]

    df = df[(df["Depth"] >= 0.2) & (df["Depth"] <= 8)]
    df = df[(df["SWIR1"] < 0.015) & (df["SWIR2"] < 0.015)]
    df = df[(df["Blue"] / (df["Green"] + eps) < 2)]

    for b in ["Blue", "Green", "Red", "SWIR1", "SWIR2"]:
        low, high = df[b].quantile([0.01, 0.99])
        df = df[(df[b] >= low) & (df[b] <= high)]

    return df.replace([np.inf, -np.inf], np.nan).dropna()



# 2b) Band reflectance vs depth — BEFORE / AFTER cleaning

band_colors = {
    "Blue": "blue",
    "Green": "green",
    "Red": "red",
    "SWIR1": "purple",
    "SWIR2": "brown"
}

bands = ["Blue", "Green", "Red", "SWIR1", "SWIR2"]

df_clean = clean_bathy_dataset(df)

for b in bands:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].scatter(df["Depth"], df[b], s=5, alpha=0.4, color=band_colors[b])
    ax[0].set_title(f"{b} reflectance vs depth — BEFORE cleaning")
    ax[0].set_xlabel("Depth (m)")
    ax[0].set_ylabel("Reflectance")
    ax[0].grid(True)

    ax[1].scatter(df_clean["Depth"], df_clean[b], s=5, alpha=0.4, color=band_colors[b])
    ax[1].set_title(f"{b} reflectance vs depth — AFTER cleaning")
    ax[1].set_xlabel("Depth (m)")
    ax[1].set_ylabel("Reflectance")
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()


print("Number of pixels after cleaning:", len(df_clean))



# 3) Features  (log RGB)

eps = 1e-6
df_clean["log_Blue"]  = np.log(df_clean["Blue"]  + eps)
df_clean["log_Green"] = np.log(df_clean["Green"] + eps)
df_clean["log_Red"]   = np.log(df_clean["Red"]   + eps)

df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()

X_log = df_clean[["log_Blue", "log_Green", "log_Red"]]
y = df_clean["Depth"]



# 4a) Spatial train / test split 75 / 25 

x_thresh = df_clean["x"].quantile(0.75)

train_idx = df_clean["x"] <= x_thresh
test_idx  = df_clean["x"] >  x_thresh

X_train, X_test = X_log.loc[train_idx], X_log.loc[test_idx]
y_train, y_test = y.loc[train_idx], y.loc[test_idx]

print(f"Train set: {len(X_train)} pixels ({len(X_train)/len(X_log)*100:.1f} %)")
print(f"Test set : {len(X_test)} pixels ({len(X_test)/len(X_log)*100:.1f} %)")

# Visual check of spatial split
plt.figure(figsize=(7,6))
plt.scatter(df_clean.loc[train_idx, "x"], df_clean.loc[train_idx, "y"],
            s=3, label="Train", alpha=0.5)
plt.scatter(df_clean.loc[test_idx, "x"], df_clean.loc[test_idx, "y"],
            s=3, label="Test", alpha=0.8)
plt.legend()
plt.gca().invert_yaxis()
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.title("Spatial train / test split (75 / 25)")
plt.grid(True)
plt.show()



# 4b) Depth distribution — TRAIN vs TEST

bins = np.arange(0, 9, 1)
df_clean["depth_bin"] = pd.cut(df_clean["Depth"], bins=bins)

df_train = df_clean.loc[train_idx]
df_test  = df_clean.loc[test_idx]

depth_counts = pd.DataFrame({
    "Train_count": df_train["depth_bin"].value_counts().sort_index(),
    "Test_count":  df_test["depth_bin"].value_counts().sort_index()
})

print("\nNumber of pixels per depth bin (TRAIN vs TEST):")
print(depth_counts)

#Model performance could not be evaluated for depths between 7 and 8 m due to the absence of 
#independent test samples in this depth range, reflecting their limited spatial extent 
#in the study area.


# 5) Baseline — RegLin in log space

ols = LinearRegression()
ols.fit(X_train, y_train)

y_pred_test_ols = ols.predict(X_test)
rmse_ols = np.sqrt(mean_squared_error(y_test, y_pred_test_ols))

print("\n=== LYZENGA OLS — TEST SET ===")
print("RMSE =", rmse_ols)



# 6) Linear Regression residuals — map & RMSE by depth

df_test = df_clean.loc[test_idx].copy()
df_test["residual_ols"] = y_test - y_pred_test_ols

plt.figure(figsize=(7,6))
plt.scatter(
    df_test["x"], df_test["y"],
    c=df_test["residual_ols"], cmap="coolwarm", s=5
)
plt.colorbar(label="Residual (m)")
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.title("Lyzenga OLS — Spatial residuals (test set)")
plt.gca().invert_yaxis()
plt.grid(True)
plt.show()

rmse_depth_ols = df_test.groupby("depth_bin")["residual_ols"].apply(
    lambda r: np.sqrt(np.mean(r**2))
)

print("\nRMSE by depth bin — OLS (test set)")
print(rmse_depth_ols)



# 7) Random Forest — hyperparameter tuning on TRAIN (spatial CV)

from sklearn.cluster import KMeans

coords_train = df_clean.loc[train_idx, ["x", "y"]].values
clusters_train = KMeans(n_clusters=5, random_state=42).fit_predict(coords_train)

outer_cv = GroupKFold(n_splits=5)

rf = RandomForestRegressor(random_state=42, n_jobs=-1)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [8, 12, 16],
    "min_samples_leaf": [2, 4],
    "max_features": [1, 2]
}

gs = GridSearchCV(
    rf,
    param_grid,
    scoring="neg_root_mean_squared_error",
    cv=outer_cv.split(X_train, y_train, clusters_train),
    n_jobs=-1,
    verbose=1
)

gs.fit(X_train, y_train)

print("\nBest RF hyperparameters:", gs.best_params_)



# Visualisation des clusters spatiaux utilisés pour le Spatial CV (TRAIN SET)


plt.figure(figsize=(7,6))
sc = plt.scatter(
    df_clean.loc[train_idx, "x"],
    df_clean.loc[train_idx, "y"],
    c=clusters_train,
    cmap="tab10",
    s=5
)

plt.gca().invert_yaxis()
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.title("Spatial clusters used for Random Forest cross-validation (training set)")

# Légende (clusters 1 à 5)
plt.legend(
    handles=sc.legend_elements()[0],
    labels=["1", "2", "3", "4", "5"],
    title="Cluster",
    loc="upper right",
    frameon=True
)

plt.grid(True)
plt.show()


# 8) Random Forest — retrain on 75 % and final test

best_rf = gs.best_estimator_
best_rf.fit(X_train, y_train)

y_pred_test_rf = best_rf.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_test_rf))

print("\n=== RANDOM FOREST — TEST SET ===")
print("RMSE =", rmse_rf)



# 9) RF residuals — map & RMSE by depth

df_test["residual_rf"] = y_test - y_pred_test_rf

plt.figure(figsize=(7,6))
plt.scatter(
    df_test["x"], df_test["y"],
    c=df_test["residual_rf"], cmap="coolwarm", s=5
)
plt.colorbar(label="Residual (m)")
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.title("Random Forest — Spatial residuals (test set)")
plt.gca().invert_yaxis()
plt.grid(True)
plt.show()

rmse_depth_rf = df_test.groupby("depth_bin")["residual_rf"].apply(
    lambda r: np.sqrt(np.mean(r**2))
)

print("\nRMSE by depth bin — Random Forest (test set)")
print(rmse_depth_rf)



