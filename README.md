# IND210

## Prognose med Prophet - README

### Beskrivelse

Dette er en enkel app for prognose av salgsvolum ved hjelp av Prophet-modellen. Appen lar deg bruke historiske salgsdata for å lage fremtidsprognoser, og den inkluderer også en lageranalyse som estimerer hvor lenge lageret hos grossist og detaljist vil vare basert på prognosen.

### Funksjonalitet:
	1.	Filvalg:
	•	Du kan velge å bruke en standardfil (salg_risbrod.csv) med eksempeldata, eller laste opp din egen fil med historiske salgsdata.
	2.	Prognoseinnstillinger:
	•	Velg hvilken salgskanal du vil bruke (grossist, detaljist, etc.).
	•	Angi startdato for prognosen.
	•	Velg antall uker du vil predikere fremover.
	3.	Salgsprognose:
	•	Appen bruker Prophet-modellen til å lage salgsprognoser basert på historiske data og valgt salgskanal.
	4.	Lageranalyse:
	•	Beregn hvor lenge lageret hos grossist og detaljist vil vare basert på salgsprognosene.
	•	Får varsler hvis lageret er i ferd med å gå tomt i løpet av den valgte prognoseperioden.

### Krav

For å bruke denne appen, trenger du følgende Python-biblioteker:
	•	pandas
	•	numpy
	•	matplotlib
	•	prophet
	•	streamlit

### Installasjon

Du kan installere de nødvendige bibliotekene ved å bruke pip:

pip install pandas numpy matplotlib prophet streamlit

### Bruk
	1.	Kjør appen:
 	•	For å starte appen, kan du bruke Streamlit ved å kjøre følgende kommando i terminalen: streamlit run prognose_app.py
	2.	Velg fil:
	•	Hvis du vil bruke den standardfilen (salg_risbrod.csv), kan du bare krysse av for å bruke den.
	•	Hvis du har egne data, kan du laste opp en CSV-fil med samme struktur som eksempelfilen.
	3.	Angi innstillinger:
	•	Velg ønsket salgskanal (for eksempel “Sales per week in fpk / Retailer”).
	•	Skriv inn ønsket startdato for prognosen.
	•	Juster antall uker du vil predikere fremover.
	4.	Vis prognose og analyse:
	•	Appen vil vise salgsprognoser sammen med et 95% konfidensintervall.
	•	Den viser også hvor lenge lageret hos grossist og detaljist vil vare.
	5.	Lagre og oppdater fil:
	•	Du kan laste ned den oppdaterte filen med prognose og salgsdata.

### Eksempeldata

Eksempeldatasettet er en CSV-fil som inneholder ukentlig salgsvolum for grossist og detaljist, lagerbeholdning, samt rekkevidde for grossist og detaljist. Detaljene kan lastes ned og brukes som mal for egne data.

### Lageranalyse

Lageranalysen gir en estimert tid (i uker) som lageret vil vare basert på salgsprognosen. Dersom lageret er i ferd med å gå tomt i løpet av prognoseperioden, vil du få en advarsel.

Merk: Prognosemodellen og lageranalysen er basert på historiske data og kan ikke garantere fremtidige resultater. Sørg for å justere prognoseinnstillingene etter behov.
