import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib-cache-ml-task"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "EDA"
RESULTS_DIR = DATA_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def fix_text(value):
    if not isinstance(value, str):
        return value
    try:
        return value.encode("cp1251").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value


def outlier_percent(series):
    series = series.dropna()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    mask = (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)
    return mask.mean() * 100


def save_figure(file_name):
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / file_name, dpi=140, bbox_inches="tight")
    plt.close()


data = pd.read_csv(DATA_DIR / "data.csv")
info = pd.read_csv(DATA_DIR / "info.csv")

data = data.loc[:, ~data.columns.str.startswith("Unnamed")]
info = info.loc[:, ~info.columns.str.startswith("Unnamed")]

for column in [
    "msc_name",
    "piont_name",
    "Direction",
    "spectr_name",
    "Description",
]:
    info[column] = info[column].map(fix_text)

data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
data["Value1"] = pd.to_numeric(data["Value1"], errors="coerce")
data["Value2"] = pd.to_numeric(data["Value2"], errors="coerce")

merged = data.merge(info, on="idMeasure", how="left", validate="many_to_one")
merged["sensor"] = merged["piont_name"] + " / " + merged["Direction"]

sns.set_theme(style="whitegrid")

date_min = data["Date"].min()
date_max = data["Date"].max()
active_days = data["Date"].dt.date.nunique()
median_interval = (
    data.sort_values(["idMeasure", "Date"])
    .groupby("idMeasure")["Date"]
    .diff()
    .dt.total_seconds()
    .div(60)
    .median()
)

monthly_counts = data.set_index("Date").resample("MS").size()
plt.figure(figsize=(12, 5))
sns.barplot(
    x=monthly_counts.index.strftime("%Y-%m"),
    y=monthly_counts.values,
    color="#4C72B0",
)
plt.title("Количество измерений по месяцам")
plt.xlabel("Месяц")
plt.ylabel("Количество строк")
plt.xticks(rotation=60, ha="right")
save_figure("01_period.png")

measure_counts = data.groupby("idMeasure").size()
sensor_volume = merged.groupby("sensor").size().sort_values()
plt.figure(figsize=(10, 6))
sns.barplot(x=sensor_volume.values, y=sensor_volume.index, color="#4C72B0")
plt.title("Объем данных по датчикам")
plt.xlabel("Количество строк")
plt.ylabel("Точка / направление")
save_figure("02_sensor_volume.png")

duplicate_mask = data.duplicated(subset=["idMeasure", "Date"])
allowed_alarm_statuses = {0, 30, 40, 45, 60, 90}
invalid_alarm_count = int(
    (~data["AlarmStatus"].isin(allowed_alarm_statuses)).sum()
)

quality = data.groupby("idMeasure").agg(
    row_count=("Value1", "size"),
    missing_value1_pct=("Value1", lambda x: x.isna().mean() * 100),
    unique_value1=("Value1", "nunique"),
)
quality["duplicate_count"] = (
    data.loc[duplicate_mask]
    .groupby("idMeasure")
    .size()
    .reindex(quality.index, fill_value=0)
)
quality["finite_value1_pct"] = data.groupby("idMeasure")["Value1"].apply(
    lambda x: np.isfinite(x.dropna()).mean() * 100 if x.notna().any() else 0
)
quality["outlier_value1_pct"] = data.groupby("idMeasure")["Value1"].apply(
    outlier_percent
)
quality = quality.join(
    info[["idMeasure", "spectr_name"]].set_index("idMeasure")
)

minimum_rows = quality["row_count"].median() * 0.75
quality["is_good"] = (
    (quality["missing_value1_pct"] <= 5)
    & (quality["duplicate_count"] == 0)
    & (quality["finite_value1_pct"] == 100)
    & (quality["unique_value1"] > 1)
    & (quality["row_count"] >= minimum_rows)
)

missing_pct = data.isna().mean().mul(100).sort_values(ascending=False)
plt.figure(figsize=(9, 4))
sns.barplot(x=missing_pct.index, y=missing_pct.values, color="#55A868")
plt.title("Доля пропусков в data.csv")
plt.ylabel("Пропуски, %")
plt.xlabel("")
plt.xticks(rotation=30, ha="right")
save_figure("03_missing_values.png")

outliers_by_parameter = (
    quality.groupby("spectr_name")["outlier_value1_pct"].mean().sort_values()
)
plt.figure(figsize=(10, 6))
sns.barplot(
    x=outliers_by_parameter.values,
    y=outliers_by_parameter.index,
    color="#C44E52",
)
plt.title("Средняя доля IQR-выбросов по параметрам")
plt.xlabel("Выбросы, %")
plt.ylabel("Параметр")
save_figure("04_outliers.png")

good_ids = quality.index[quality["is_good"]]
good_data = merged[merged["idMeasure"].isin(good_ids)].copy()
good_data["hour"] = good_data["Date"].dt.floor("h")

hourly_by_measure = good_data.pivot_table(
    index="hour", columns="idMeasure", values="Value1", aggfunc="mean"
).sort_index()
full_hour_index = pd.date_range(
    hourly_by_measure.index.min(), hourly_by_measure.index.max(), freq="h"
)
hourly_by_measure = hourly_by_measure.reindex(full_hour_index)

spectrum_v = good_data[good_data["spectr_name"] == "Спектр-V"]
spectrum_v_hourly = (
    spectrum_v.groupby("hour")["Value1"].mean().reindex(full_hour_index)
)
spectrum_v_rolling = spectrum_v_hourly.rolling(24, min_periods=6).mean()

observed_hours = spectrum_v_hourly.dropna().index.to_series()
hour_gaps = observed_hours.diff()
largest_gap = hour_gaps.max()
recent_start = hour_gaps.idxmax()

plt.figure(figsize=(13, 5))
plt.plot(
    spectrum_v_hourly.index,
    spectrum_v_hourly,
    alpha=0.28,
    linewidth=0.7,
    label="Среднее за час",
)
plt.plot(
    spectrum_v_rolling.index,
    spectrum_v_rolling,
    linewidth=1.5,
    color="#C44E52",
    label="Среднее за 24 часа",
)
plt.axvline(
    recent_start,
    color="black",
    linestyle="--",
    alpha=0.6,
    label="Начало второго периода",
)
plt.title("Спектр-V после приведения к часовой шкале")
plt.xlabel("Дата")
plt.ylabel("Среднее Value1")
plt.legend()
save_figure("05_hourly_series.png")

recent_hourly = spectrum_v_hourly.loc[recent_start:].dropna()
recent_daily = recent_hourly.resample("1D").mean().dropna()
daily_rolling = recent_daily.rolling(7, min_periods=3).mean()
trend_x = (recent_daily.index - recent_daily.index.min()).days.to_numpy()
trend_slope, trend_intercept = np.polyfit(
    trend_x, recent_daily.to_numpy(), 1
)
trend_line = trend_slope * trend_x + trend_intercept
hour_profile = recent_hourly.groupby(recent_hourly.index.hour).mean()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(
    recent_daily.index, recent_daily, alpha=0.45, label="Среднее за день"
)
axes[0].plot(
    daily_rolling.index, daily_rolling, linewidth=2, label="Среднее за 7 дней"
)
axes[0].plot(
    recent_daily.index,
    trend_line,
    linestyle="--",
    color="black",
    label="Линейный тренд",
)
axes[0].set_title("Тренд во втором периоде")
axes[0].set_xlabel("Дата")
axes[0].set_ylabel("Среднее Value1")
axes[0].legend()
axes[0].tick_params(axis="x", rotation=30)

axes[1].plot(hour_profile.index, hour_profile.values, marker="o")
axes[1].set_title("Среднее значение по часу суток")
axes[1].set_xlabel("Час")
axes[1].set_ylabel("Среднее Value1")
axes[1].set_xticks(range(0, 24, 2))
save_figure("06_trend_and_seasonality.png")

parameter_hourly = good_data.pivot_table(
    index="hour", columns="spectr_name", values="Value1", aggfunc="mean"
).sort_index()
correlation = parameter_hourly.corr(min_periods=30)

plt.figure(figsize=(12, 10))
sns.heatmap(
    correlation,
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.3,
)
plt.title("Корреляция параметров после часового усреднения")
plt.xlabel("")
plt.ylabel("")
save_figure("07_correlation.png")

correlation_pairs = correlation.where(
    np.triu(np.ones(correlation.shape), k=1).astype(bool)
).stack()
strongest_pairs = correlation_pairs.reindex(
    correlation_pairs.abs().sort_values(ascending=False).index
).head(5)

print(f"Строк: {len(data)}")
print(f"Каналов: {data['idMeasure'].nunique()}")
print(f"Период: {date_min} — {date_max}")
print(f"Дней с данными: {active_days}")
print(f"Медианный интервал: {median_interval:.1f} минут")
print(f"Самый большой разрыв: {largest_gap}")
print(
    f"Записей на канал: от {measure_counts.min()} до {measure_counts.max()}"
)
print(f"Дубликаты: {duplicate_mask.sum()}")
print(f"Пропуски Value1: {data['Value1'].isna().sum()}")
print(f"Пропуски Value2: {data['Value2'].isna().sum()}")
print(f"Недопустимые AlarmStatus: {invalid_alarm_count}")
print(f"Качественные каналы: {quality['is_good'].sum()} из {len(quality)}")
print(f"Начало второго периода: {recent_start}")
print(f"Наклон тренда за день: {trend_slope:.5f}")
print(f"Час максимума: {hour_profile.idxmax()}:00")
print(f"Час минимума: {hour_profile.idxmin()}:00")
print("Самые сильные корреляции:")
for (first, second), value in strongest_pairs.items():
    print(f"{first} / {second}: {value:.3f}")
print(f"Графики сохранены в {RESULTS_DIR}")
