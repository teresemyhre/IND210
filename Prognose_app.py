import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import streamlit as st

# === STREAMLIT UI ===
st.title("📈 Prognose med Prophet")
st.write("Velg om du vil bruke en standardfil eller laste opp din egen fil. Deretter kan du velge startdato, antall uker framover og ønsket salgskanal.")

# === Brukervalg ===
use_default = st.checkbox("Bruk standardfil (salg_risbrod.csv)", value=True)

# Hvis brukeren velger å bruke standardfil
if use_default:
    df = pd.read_csv("salg_risbrod.csv", na_values="-")
    st.success("Standardfil lastet inn.")
else:
    # Hvis brukeren laster opp egen fil
    uploaded_file = st.file_uploader("Last opp CSV-fil", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, na_values="-")
        st.success("Fil lastet inn.")
    else:
        st.warning("Vennligst last opp en CSV-fil for å fortsette.")
        st.stop()

# === Forbered data ===
df = df.dropna()
df.columns = df.columns.str.strip()

# Legg til uke-kolonne
df['Year-Week'] = df['Year'].astype(str) + '-' + df['Week'].astype(str).str.zfill(2)

# Vis lagerbeholdning siste uke for grossist og detaljist
lager_wholesaler = df["Inventory in dpk / Wholesaler"].iloc[-1]*14
lager_retailer = df["Inventory in dpk / Retailer"].iloc[-1]*14
siste_uke = df['Year-Week'].iloc[-1]

st.subheader(f"📦 Lagerbeholdning siste uke i datasettet ({siste_uke}):")
col1, col2 = st.columns(2)
col1.metric(label="Wholesaler (fpk)", value=int(lager_wholesaler))
col2.metric(label="Retailer (fpk)", value=int(lager_retailer))

# Finn relevante salgskolonner
sales_columns = [col for col in df.columns if "Sales per week" in col]

if not sales_columns:
    st.error("Ingen salgskolonner funnet i filen.")
else:
    valgt_kolonne = st.selectbox("Velg salgskanal", options=sales_columns)

    # === Brukerinput for Prophet-modell ===
    startdato = st.text_input("Startdato", "2015-04-12")
    fremtidig_uker = st.slider("Antall uker fremover å predikere", min_value=1, max_value=52, value=12)

    # === Prophet-modell ===
    df_prophet = df[['Year-Week', valgt_kolonne]].copy()
    df_prophet.columns = ['year_week', 'y']
    df_prophet['ds'] = pd.date_range(start=startdato, periods=len(df_prophet), freq='W')

    model = Prophet(weekly_seasonality=True)
    model.fit(df_prophet[['ds', 'y']])

    future = model.make_future_dataframe(periods=fremtidig_uker, freq='W')
    forecast = model.predict(future)

    # Lag fremtidige etiketter
    future_weeks = df['Year-Week'].tolist() + [f'Fremtid-{i+1}' for i in range(fremtidig_uker)]
    forecast['Year-Week'] = future_weeks

    # === Plotting ===
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(forecast['Year-Week'], forecast['yhat'], label='Prognose', linestyle='--', color='green')
    ax.plot(df_prophet['year_week'], df_prophet['y'], label='Faktisk salg', linestyle='-')
    ax.fill_between(forecast['Year-Week'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.2, label='95% intervall', color='green')
    ax.axvline(x=forecast['Year-Week'][len(df_prophet)-1], color='grey', linestyle='-.', label='Prognosestart')
    ax.set_xticks(np.arange(0, len(forecast), 6))
    ax.set_xticklabels(forecast['Year-Week'][::6], rotation=45)
    ax.set_xlabel("Uke (År-Uke)")
    ax.set_ylabel("Salg (fpk)")
    ax.set_title(f"Prophet-prognose: {valgt_kolonne}", fontsize=16)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# === Lager vs prognose: Hvor lenge varer lageret? ===
# Hent siste lagerbeholdning og konverter fra dpk til fpk
dpk_to_fpk = 14
siste_lager_wholesaler = df['Inventory in dpk / Wholesaler'].iloc[-1] * dpk_to_fpk
siste_lager_retailer = df['Inventory in dpk / Retailer'].iloc[-1] * dpk_to_fpk

# Simuler uke-for-uke hvor lenge lageret varer
wholesale_lager = siste_lager_wholesaler
retail_lager = siste_lager_retailer
wholesale_uker = 0
retail_uker = 0

for uke_salg in forecast['yhat'][-fremtidig_uker:]:
    if wholesale_lager >= uke_salg:
        wholesale_lager -= uke_salg
        wholesale_uker += 1
    if retail_lager >= uke_salg:
        retail_lager -= uke_salg
        retail_uker += 1

# === Lager vs prognose: Hvor lenge varer lageret hos grossist og detaljist? ===
dpk_to_fpk = 14
siste_lager_wholesaler = df['Inventory in dpk / Wholesaler'].iloc[-1] * dpk_to_fpk
siste_lager_retailer = df['Inventory in dpk / Retailer'].iloc[-1] * dpk_to_fpk

# --- Prognose for grossist ---
df_grossist = df[['Year-Week', 'Sales per week in fpk / Wholesaler']].copy()
df_grossist.columns = ['year_week', 'y']
df_grossist['ds'] = pd.date_range(start=startdato, periods=len(df_grossist), freq='W')

model_grossist = Prophet(weekly_seasonality=True)
model_grossist.fit(df_grossist[['ds', 'y']])
future_grossist = model_grossist.make_future_dataframe(periods=fremtidig_uker, freq='W')
forecast_grossist = model_grossist.predict(future_grossist)

# --- Prognose for detaljist ---
df_retailer = df[['Year-Week', 'Sales per week in fpk / Retailer']].copy()
df_retailer.columns = ['year_week', 'y']
df_retailer['ds'] = pd.date_range(start=startdato, periods=len(df_retailer), freq='W')

model_retailer = Prophet(weekly_seasonality=True)
model_retailer.fit(df_retailer[['ds', 'y']])
future_retailer = model_retailer.make_future_dataframe(periods=fremtidig_uker, freq='W')
forecast_retailer = model_retailer.predict(future_retailer)

# === Lageranalyse: Simuler uke for uke ===
wholesale_lager = siste_lager_wholesaler
retail_lager = siste_lager_retailer
wholesale_uker = 0
retail_uker = 0

for uke_salg in forecast_grossist['yhat'][-fremtidig_uker:]:
    if wholesale_lager >= uke_salg:
        wholesale_lager -= uke_salg
        wholesale_uker += 1

for uke_salg in forecast_retailer['yhat'][-fremtidig_uker:]:
    if retail_lager >= uke_salg:
        retail_lager -= uke_salg
        retail_uker += 1

# === Vis resultat til bruker ===
st.markdown("### 🧮 Lageranalyse basert på prognose")
st.write(f"📦 Lager hos grossist (wholesaler) varer i ca. **{wholesale_uker} uker** gitt prognosen.")
st.write(f"🛒 Lager hos detaljist (retailer) varer i ca. **{retail_uker} uker** gitt prognosen.")

if wholesale_uker < fremtidig_uker:
    manko_uke = forecast_grossist['ds'].iloc[len(df_grossist) + wholesale_uker].strftime("%Y-%U")
    st.warning(f"⚠️ **Grossistlager kan gå tomt i uke {manko_uke}**.")
if retail_uker < fremtidig_uker:
    manko_uke = forecast_retailer['ds'].iloc[len(df_retailer) + retail_uker].strftime("%Y-%U")
    st.warning(f"⚠️ **Detaljistlager kan gå tomt i uke {manko_uke}**.")
