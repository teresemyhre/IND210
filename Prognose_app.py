import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import streamlit as st

# === STREAMLIT UI ===
st.title("📈 Prognose med Prophet")
st.write("Last opp en fil, velg startdato, antall uker framover, og ønsket salgskanal.")

# === Brukervalg ===
uploaded_file = st.file_uploader("Last opp CSV-fil", type=["csv"])
startdato = st.text_input("Startdato (må være første uke i datasettet)", "2015-04-12")
fremtidig_uker = st.slider("Antall uker fremover å predikere", min_value=1, max_value=52, value=12)

# === Når bruker laster opp fil ===
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, na_values="-")
    df = df.dropna()
    df.columns = df.columns.str.strip()

    # Legg til uke-kolonne
    df['Year-Week'] = df['Year'].astype(str) + '-' + df['Week'].astype(str).str.zfill(2)

    # Finn relevante salgskolonner
    sales_columns = [col for col in df.columns if "Sales per week" in col]

    if not sales_columns:
        st.error("Ingen salgskolonner funnet i filen.")
    else:
        valgt_kolonne = st.selectbox("Velg salgskanal", options=sales_columns)

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