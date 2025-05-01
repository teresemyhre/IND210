import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import streamlit as st
import io

# === STREAMLIT UI ===
st.title("📈 Prognose med Prophet")
st.write("Velg om du vil bruke standardfilen eller laste opp din egen fil med historiske salgsdata. Deretter kan du velge startdato, antall uker framover og ønsket salgskanal.")

# === Brukervalg ===
use_default = st.checkbox("Bruk standardfil (salg_risbrod.csv)", value=True)

with st.expander("📊 Last ned og oppdater standardfilen (salg_risbrod.csv)"):
    st.markdown(
        "Standardfilen brukes dersom du ikke laster opp din egen fil. "
        "Om ønskelig, kan du laste den ned her, oppdatere med dine egne data og så laste den opp igjen i appen."
    )
    with open("salg_risbrod.csv", "rb") as f:
        st.download_button(
            label="📥 Last ned standardfil (salg_risbrod.csv)",
            data=f,
            file_name="salg_risbrod.csv",
            mime="text/csv"
        )

# Eksempeldata
data = {
    "Year": [2015, 2015, 2015],
    "Week": [1, 2, 3],
    "Accumulated sales volume in fpk / Supplier": [100, 150, 200],
    "Accumulated sales volume in fpk / Wholesaler": [80, 120, 160],
    "Accumulated sales volume in fpk / Retailer": [60, 90, 120],
    "Accumulated sales volume in fpk / total": [240, 360, 480],
    "Sales per week in fpk / Supplier": [10, 15, 20],
    "Sales per week in fpk / Wholesaler": [8, 12, 16],
    "Sales per week in fpk / Retailer": [6, 9, 12],
    "Inventory in dpk / Wholesaler": [1000, 950, 900],
    "Inventory in dpk / Retailer": [800, 750, 700],
    "rekkevidde / Wholesaler": [12, 11, 10],
    "rekkevidde / Retailer": [10, 9, 8]
}

df_example = pd.DataFrame(data)

# Lag en binær buffer
buffer = io.BytesIO()
buffer.write(df_example.to_csv(index=False).encode('utf-8'))
buffer.seek(0)

# Hvis brukeren velger å bruke standardfil
if use_default:
    df = pd.read_csv("salg_risbrod.csv", na_values="-")
    st.success("Standardfil lastet inn.")
else:
    # Hvis brukeren laster opp egen fil
    st.markdown("""
    ### 📄 Eksempelfil for datastruktur
    
    For å bruke denne appen med egne data, må datasettet ditt ha samme struktur som eksempelfilen under. Filen viser hvilke kolonner som kreves, og hvordan dataene bør være organisert uke for uke.
    
    Du kan laste ned og bruke den som en mal:
    """)
    
    # Last ned-knapp
    st.download_button(
        label="Last ned eksempel-fil",
        data=buffer,
        file_name="eksempel_datasett.csv",
        mime="text/csv"
    )
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


# Finn relevante salgskolonner
sales_columns = [col for col in df.columns if "Sales per week" in col]

if not sales_columns:
    st.error("Ingen salgskolonner funnet i filen.")
else:
    # Finn index til Retailer-kolonnen hvis den finnes
    default_index = next((i for i, col in enumerate(sales_columns) if "Retailer" in col), 0)
    valgt_kolonne = st.selectbox("Velg salgskanal", options=sales_columns, index=default_index)

    # === Brukerinput for Prophet-modell ===
    startdato = st.text_input("Startdato", "2015-04-12")
    startdato = pd.to_datetime(startdato)  # Konverter startdato til datetime
    fremtidig_uker = st.slider("Antall uker fremover å predikere", min_value=1, max_value=52, value=12)

    # === Prophet-modell ===
    df_prophet = df[['Year-Week', valgt_kolonne]].copy()
    df_prophet.columns = ['year_week', 'y']
    df_prophet['ds'] = pd.to_datetime(df['Year'].astype(str) + df['Week'].astype(str) + '7', format='%G%V%u')

    # Filtrer datasettet for å kun bruke data fra startdato og fremover
    df_filtered = df_prophet[df_prophet['ds'] >= startdato]

    # Vis det filtrerte datasettet
    st.write(df_filtered)

    model = Prophet(weekly_seasonality=True)
    model.fit(df_filtered[['ds', 'y']])

    future = model.make_future_dataframe(periods=fremtidig_uker, freq='W')
    forecast = model.predict(future)

    # Lag fremtidige etiketter
    future_weeks = df['Year-Week'].tolist() + [f'Fremtid-{i+1}' for i in range(fremtidig_uker)]
    forecast['Year-Week'] = future_weeks

    # === Plotting ===
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(forecast['Year-Week'], forecast['yhat'], label='Prognose', linestyle='--', color='green')
    ax.plot(df_filtered['year_week'], df_filtered['y'], label='Faktisk salg', linestyle='-')
    ax.fill_between(forecast['Year-Week'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.2, label='95% intervall', color='green')
    ax.axvline(x=forecast['Year-Week'][len(df_filtered)-1], color='grey', linestyle='-.', label='Prognosestart')
    ax.set_xticks(np.arange(0, len(forecast), 6))
    ax.set_xticklabels(forecast['Year-Week'][::6], rotation=45)
    ax.set_xlabel("Uke (År-Uke)")
    ax.set_ylabel("Salg (fpk)")
    ax.set_title(f"Prophet-prognose: {valgt_kolonne}", fontsize=16)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

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
st.markdown(
    "<p style='font-size: 0.95rem; color: gray; font-style: italic;'>Obs: Lageranalysen viser hvor lenge lageret rekker dersom det ikke gjøres nye bestillinger i perioden.</p>",
    unsafe_allow_html=True
)





# Vis lagerbeholdning siste uke for grossist og detaljist
lager_wholesaler = df["Inventory in dpk / Wholesaler"].iloc[-1]*14
lager_retailer = df["Inventory in dpk / Retailer"].iloc[-1]*14
siste_uke = df['Year-Week'].iloc[-1]

st.markdown(f"##### 📦 Lagerbeholdning siste uke i datasettet ({siste_uke}):")
col1, col2 = st.columns(2)
col1.metric(label="Wholesaler (fpk)", value=int(lager_wholesaler))
col2.metric(label="Retailer (fpk)", value=int(lager_retailer))






st.write(f"📦 Lager hos wholesaler varer i ca. **{wholesale_uker} uker** gitt prognosen.")
st.write(f"🛒 Lager hos retailer varer i ca. **{retail_uker} uker** gitt prognosen.")

if wholesale_uker < fremtidig_uker:
    manko_uke = forecast_grossist['ds'].iloc[len(df_grossist) + wholesale_uker].strftime("%Y-%U")
    st.warning(f"⚠️ **Wholesale-lager kan gå tomt i uke {manko_uke}**.")
if retail_uker < fremtidig_uker:
    manko_uke = forecast_retailer['ds'].iloc[len(df_retailer) + retail_uker].strftime("%Y-%U")
    st.warning(f"⚠️ **Retail-lager kan gå tomt i uke {manko_uke}**.")

st.caption("Merk: Lageranalysen er begrenset til valgt prognoseperiode. Lageret kan vare lenger enn vist dersom prognoseperioden utvides.")
