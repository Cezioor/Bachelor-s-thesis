# Wpływ kontuzji ACL na powrót do sportu

[🇬🇧 English](README.md) · **🇵🇱 Polski**

**▶ Aplikacja online: [acl-injury-analysis.streamlit.app](https://acl-injury-analysis.streamlit.app/?lang=pl)** · dostępna po polsku i angielsku

Interaktywna analiza danych pokazująca, jak zerwanie więzadła krzyżowego przedniego (ACL) zmienia osiągi sportowców.
Porównuje statystyki zawodników przed i po kontuzji w lidze **NBA, WNBA, NFL oraz w męskiej piłce nożnej**.
Projekt powstał w ramach mojej pracy licencjackiej z bioinformatyki na Uniwersytecie Gdańskim.

## Co można zrobić w aplikacji

- **Porównać zawodników** przed i po kontuzji: tabela zmian procentowych i grupowane wykresy słupkowe dla wybranych statystyk
- **Badać zmiany wielu statystyk naraz** na wykresach punktowych, radarowych i heatmapie
- **Pogrupować zawodników według powrotu do formy**: PCA na standaryzowanych zmianach, następnie klasteryzacja k-means z automatyczną interpretacją każdego klastra
- **Sprawdzić, które statystyki najbardziej cierpią** po kontuzji ACL (ranking wg średniej zmiany)
- **Porównać mężczyzn i kobiety** (NBA vs WNBA)

## Metody

- Statystyki każdego zawodnika są podsumowane dla sezonów przed i po kontuzji; aplikacja liczy zmianę procentową każdej statystyki (dla udziału meczów w pierwszej piątce – zmianę w punktach procentowych)
- Analiza wielowymiarowa: `StandardScaler` → `PCA(n_components=2)` → `KMeans`, z wyjaśnioną wariancją na osiach
- Część teoretyczna i pełna dyskusja wyników: [`Praca_Licencjacka_teoria.pdf`](Praca_Licencjacka_teoria.pdf)

## Dane

Wszystkie dane są w pliku [`Excel-licencjat.xlsx`](Excel-licencjat.xlsx), jeden arkusz na ligę (oraz po jednym na każdą kategorię statystyk NFL: passing, rushing, receiving, defense, scoring).

## Uruchomienie lokalne

```bash
git clone https://github.com/Cezioor/Bachelor-s-thesis.git
cd Bachelor-s-thesis
pip install -r requirements.txt
streamlit run app.py
```

## Technologie

Python · Streamlit · pandas · NumPy · scikit-learn (PCA, k-means) · Plotly · Matplotlib · Seaborn
