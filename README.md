# How ACL injuries affect return to sport

**🇬🇧 English** · [🇵🇱 Polski](README.pl.md)

**▶ Live app: [acl-injury-analysis.streamlit.app](https://acl-injury-analysis.streamlit.app)** · available in English and Polish ([wersja polska](https://acl-injury-analysis.streamlit.app/?lang=pl))

An interactive data analysis of how an anterior cruciate ligament (ACL) tear changes athletes' performance.
It compares player statistics before and after the injury across the **NBA, WNBA, NFL and men's football (soccer)**.
Built as my bachelor's thesis project in Bioinformatics at the University of Gdańsk.

## What you can do in the app

- **Compare players** before and after the injury: a table of percentage changes and grouped bar charts for any chosen statistics
- **Explore changes across many statistics at once** with scatter plots, radar charts and a heatmap
- **Group players by how they returned to form**: PCA on the standardised changes, then k-means clustering, with an automatic interpretation of each cluster
- **See which statistics suffer most** after an ACL injury, ranked by mean change
- **Compare men and women** (NBA vs WNBA)

## Methods

- Each player's statistics are summarised for the seasons before and after the injury; the app computes the percentage change for every statistic (percentage-point change for the share of games started)
- Multivariate analysis: `StandardScaler` → `PCA(n_components=2)` → `KMeans`, with the explained variance shown on the axes
- The theory and full discussion are in [`Praca_Licencjacka_teoria.pdf`](Praca_Licencjacka_teoria.pdf) (in Polish)

## Data

All data is in [`Excel-licencjat.xlsx`](Excel-licencjat.xlsx), one sheet per league (and one per NFL statistics category: passing, rushing, receiving, defense, scoring).

## Run locally

```bash
git clone https://github.com/Cezioor/Bachelor-s-thesis.git
cd Bachelor-s-thesis
pip install -r requirements.txt
streamlit run app.py
```

## Stack

Python · Streamlit · pandas · NumPy · scikit-learn (PCA, k-means) · Plotly · Matplotlib · Seaborn
