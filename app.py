import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --------- POLA STATYSTYK ---------
fields_nba = [
    ("games played", "Mecze rozegrane"),
    ("GS_percent", "Udział meczów w pierwszej piątce (%)"),
    ("Minutes played per match", "Minuty/mecz"),
    ("PTS", "Punkty"),
    ("AST", "Asysty"),
    ("REB", "Zbiórki"),
    ("OR", "Zbiórki atak"),
    ("DR", "Zbiórki obrona"),
    ("BLK", "Bloki"),
    ("STL", "Przechwyty"),
    ("FG_made", "Rzuty z gry (trafione)"),
    ("FG_att", "Rzuty z gry (próby)"),
    ("FG%", "Skuteczność z gry (%)"),
    ("3PT_made", "Rzuty za 3 (trafione)"),
    ("3PT_att", "Rzuty za 3 (próby)"),
    ("3PT%", "Skuteczność za 3 (%)"),
    ("FT_made", "Rzuty wolne (trafione)"),
    ("FT_att", "Rzuty wolne (próby)"),
    ("FT%", "Skuteczność rzutów wolnych (%)"),
    ("TO", "Straty"),
    ("PF", "Faule"),
]
fields_nfl_passing = [
    ("GP", "Mecze rozegrane"),
    ("CMP", "Celne podania"),
    ("ATT", "Próby podań"),
    ("CMP%", "Celność podań (%)"),
    ("YDS", "Yardy podań"),
    ("AVG", "Yardy na próbę"),
    ("TD", "Przyłożenia podaniem"),
    ("INT", "Przechwyty podań"),
    ("LNG", "Najdłuższe podanie"),
    ("SACK", "Sacks"),
    ("RTG", "Ocena podającego"),
    ("QBR", "QBR"),
]

fields_nfl_rushing = [
    ("GP", "Mecze rozegrane"),
    ("CAR", "Próby biegu"),
    ("YDS", "Yardy biegu"),
    ("AVG", "Średnia jardów/bieg"),
    ("TD", "Przyłożenia biegiem"),
    ("LNG", "Najdłuższy bieg"),
    ("FD", "Pierwsze próby biegiem"),
    ("FUM", "Fumble"),
    ("LST", "Fumble stracone"),
]

fields_nfl_receiving = [
    ("GP", "Mecze rozegrane"),
    ("REC", "Złapane podania"),
    ("TGTS", "Celem podania"),
    ("YDS", "Yardy po złapaniu"),
    ("AVG", "Średnia jardów/odbiory"),
    ("TD", "Przyłożenia"),
    ("LNG", "Najdłuższe złapane"),
    ("FD", "Pierwsze próby"),
    ("FUM", "Fumble"),
    ("LST", "Fumble stracone"),
]

fields_nfl_defense = [
    ("GP", "Mecze rozegrane"),
    ("TOT", "Łącznie tackli"),
    ("SOLO", "Solo tackli"),
    ("AST", "Asysty w tacklach"),
    ("SACK", "Sacki"),
    ("FF", "Forced fumble"),
    ("FR", "Odzyskane fumble"),
    ("YDS", "Yardy przechwytów/fumble"),
    ("INT", "Interceptions"),
    ("AVG", "Średnia jardów na interception"),
    ("TD", "Przyłożenia po obronie"),
    ("LNG", "Najdłuższa akcja"),
    ("PD", "Passes defended"),
    ("STF", "Stuffs"),
    ("STFYDS", "Stuff yards"),
    ("KB", "Zablokowane kopnięcia"),
]

fields_nfl_scoring = [
    ("GP", "Mecze rozegrane"),
    ("PASS", "TD po podaniu"),
    ("RUSH", "TD po biegu"),
    ("REC", "TD po odbiorze"),
    ("RET", "Return TD"),
    ("TD", "Wszystkie touchdowny"),
    ("2PT", "Dwupunktowe"),
    ("PAT", "Extra point kick"),
    ("FG", "Field goals"),
    ("PTS", "Punkty"),
]

fields_soccer = [
    ("Matches played", "Mecze rozegrane"),
    ("Matches started", "Mecze w pierwszym składzie"),
    ("PPG", "Punkty na mecz"),
    ("Goals per match", "Gole na mecz"),
    ("Assists per match", "Asysty na mecz"),
    ("Own goals", "Samobóje"),
    ("Substitutions in", "Wejścia z ławki"),
    ("Substitution out", "Zejścia z boiska"),
    ("Yellow card per match", "Żółte kartki na mecz"),
    ("Second yellow card per match", "Drugie żółte na mecz"),
    ("Red cards per match", "Czerwone kartki na mecz"),
    ("Penalty goals", "Gole z karnych"),
    ("Minutes per goal", "Co ile minut gol"),
    ("Minutes played per match summary", "Minuty/mecz"),
]


# --------- MAPOWANIE NA KOLUMNY EXCEL ---------
def get_excel_col(field_code):
    # Mapowanie dla skuteczności
    if field_code == "FG%":
        if "Field goal percentage" in df.columns:
            return "Field goal percentage"
        elif "Skuteczność z gry (%)" in df.columns:
            return "Skuteczność z gry (%)"
        elif "FG%" in df.columns:
            return "FG%"
    if field_code == "3PT%":
        if "Three-point field goal percentage" in df.columns:
            return "Three-point field goal percentage"
        elif "Skuteczność za 3 (%)" in df.columns:
            return "Skuteczność za 3 (%)"
        elif "3PT%" in df.columns:
            return "3PT%"
    if field_code == "FT%":
        if "Free throw percentage" in df.columns:
            return "Free throw percentage"
        elif "Skuteczność rzutów wolnych (%)" in df.columns:
            return "Skuteczność rzutów wolnych (%)"
        elif "FT%" in df.columns:
            return "FT%"
    if field_code.startswith("FG_"):
        return "Field goals made-attempted per game"
    if field_code.startswith("3PT_"):
        return "Three-point field goals made-attempted per game"
    if field_code.startswith("FT_"):
        return "Free throws made-attempted per game"
    return field_code

# --------- ROZBIJANIE WARTOŚCI (trafione-próby) ---------
def split_value(val):
    if pd.isnull(val):
        return (np.nan, np.nan)
    s = str(val).replace(",", ".")
    if '-' in s:
        try:
            a, b = [x.strip() for x in s.split('-')]
            return (round(float(a),2), round(float(b),2))
        except:
            return (np.nan, np.nan)
    try:
        return (round(float(s),2), np.nan)
    except:
        return (np.nan, np.nan)
    
# --------- Dolicza 0 do analizy -----------
    
def procentowa_zmiana(before, after, max_display=200):
    try:
        before = float(str(before).replace(",", "."))
        after = float(str(after).replace(",", "."))
        if before == 0 and after == 0:
            return 0.0
        elif before == 0 and after > 0:
            return f"> {max_display}%"  # albo np.nan, albo "nowy udział"
        elif before == 0 and after < 0:
            return np.nan
        else:
            raw = ((after - before) / before) * 100
            if abs(raw) > max_display:
                return f"> {max_display}%"
            return round(raw, 1)
    except Exception:
        return np.nan

# --------- POBIERANIE STATYSTYKI DLA RÓŻNYCH POL ---------
def get_stat_value(row, field_code):
    # Obsługa specjalna dla GS% – udział startów w pierwszej piątce
    if field_code in ["GS_percent", "GS%", "GS_pct"]:
        games_played = row["games played"] if "games played" in row else np.nan
        gs = row["GS"] if "GS" in row else np.nan
        if not pd.isnull(gs) and not pd.isnull(games_played) and games_played >= 1:
            return round(100 * gs / games_played, 2)
        else:
            return np.nan
        
    col = get_excel_col(field_code)
    val = row[col] if col in row else np.nan
    if "_made" in field_code:
        return split_value(val)[0]
    elif "_att" in field_code:
        return split_value(val)[1]
    else:
        return val if not pd.isnull(val) else np.nan

# --------- WCZYTANIE DANYCH ---------
@st.cache_data
def load_data():
    df_nba = pd.read_excel("Excel-licencjat.xlsx", sheet_name="Basketball-man")
    df_wnba = pd.read_excel("Excel-licencjat.xlsx", sheet_name="Basketball-woman")
    nfl_sheets = {
        "Passing": pd.read_excel("Excel-licencjat.xlsx", sheet_name="NFL - Passing"),
        "Rushing": pd.read_excel("Excel-licencjat.xlsx", sheet_name="NFL - Rushing"),
        "Receiving": pd.read_excel("Excel-licencjat.xlsx", sheet_name="NFL - Receiving"),
        "Defense": pd.read_excel("Excel-licencjat.xlsx", sheet_name="NFL - Defense"),
        "Scoring": pd.read_excel("Excel-licencjat.xlsx", sheet_name="NFL - Scoring"),
    }
    df_soccer_man = pd.read_excel("Excel-licencjat.xlsx", sheet_name="Soccer-Man")
    df_soccer_woman = pd.read_excel("Excel-licencjat.xlsx", sheet_name="Soccer-Woman")
    return df_nba, df_wnba, nfl_sheets, df_soccer_man, df_soccer_woman

df_nba, df_wnba, nfl_sheets, df_soccer_man, df_soccer_woman = load_data()

def get_gs_percent(row):
    try:
        gs = float(row['GS'])
        gp = float(row['games played'])
        if gp == 0:
            return np.nan
        return 100 * gs / gp
    except:
        return np.nan

# --------- WYBÓR LIGI ---------
liga = st.radio("Wybierz ligę do szczegółowej analizy:", ["NBA", "WNBA", "NFL", "Soccer-Man", "Soccer-Woman"])
if liga == "NBA":
    df = df_nba
    fields = fields_nba
elif liga == "WNBA":
    df = df_wnba
    fields = fields_nba  # możesz zrobić osobno fields_wnba, jeśli inne
elif liga == "NFL":
    nfl_sheet = st.selectbox(
        "Wybierz rodzaj statystyk NFL:", 
        list(nfl_sheets.keys()), 
        index=0
    )
    df = nfl_sheets[nfl_sheet]
    if nfl_sheet == "Passing":
        fields = fields_nfl_passing
    elif nfl_sheet == "Rushing":
        fields = fields_nfl_rushing
    elif nfl_sheet == "Receiving":
        fields = fields_nfl_receiving
    elif nfl_sheet == "Defense":
        fields = fields_nfl_defense
    elif nfl_sheet == "Scoring":
        fields = fields_nfl_scoring
elif liga == "Soccer-Man":
    df = df_soccer_man
    fields = fields_soccer
elif liga == "Soccer-Woman":
    df = df_soccer_woman
    fields = fields_soccer    
# --------- FUNKCJA: ZMIANY STATYSTYK ---------
def build_stat_change(df):
    players_all = df[df["Name - Season"].str.contains("summary before")]["Name - Season"].str.replace(" summary before", "").tolist()
    rows = []
    for player in players_all:
        before = df[df["Name - Season"] == f"{player} summary before"].iloc[0]
        after = df[df["Name - Season"] == f"{player} summary after"].iloc[0]
        row = {}
        for code, desc in fields:
            b = get_stat_value(before, code)
            a = get_stat_value(after, code)
            if desc == "Udział meczów w pierwszej piątce (%)":
                if not pd.isnull(a) and not pd.isnull(b):
                    zmiana = a - b
                else:
                    zmiana = np.nan
            else:
                if b and not pd.isnull(b):
                    zmiana = ((a-b)/b)*100
                else:
                    zmiana = np.nan
            row[desc] = zmiana
        rows.append(row)
    return pd.DataFrame(rows)
# ---------------------- SZCZEGÓŁOWA ANALIZA -------------------------

st.divider()
st.title(f"Analiza wybranej ligi: {liga}")

tab1, tab2 = st.tabs(["🔢 Porównanie zawodników", "📊 Analiza wielowymiarowa"])

with tab1:
    st.header("Porównanie wybranych zawodników przed i po kontuzji")
    players_all = df[df["Name - Season"].str.contains("summary before")]["Name - Season"].str.replace(" summary before", "").tolist()
    players_select = st.multiselect(
        "Wybierz zawodników do porównania (domyślnie 2):", players_all, default=players_all[:2]
    )
    if st.checkbox("Zaznacz wszystkich zawodników"):
        players_select = players_all

    stat_desc_all = [desc for code, desc in fields]

    default_stats = stat_desc_all[:3] if len(stat_desc_all) >= 3 else stat_desc_all

    stat_choices = st.multiselect(
    "Wybierz statystyki do porównania:",
    stat_desc_all,
    default=default_stats,
    key="multi_stats"
)
    select_all_stats = st.checkbox("Zaznacz wszystkie statystyki")

    if select_all_stats and set(stat_choices) != set(stat_desc_all):
        stat_choices = stat_desc_all
    elif not select_all_stats and set(stat_choices) == set(stat_desc_all):
        stat_choices = ["Punkty", "Asysty", "Zbiórki"]

    # Tabela porównawcza
    rows = []
    for player in players_select:
        before = df[df["Name - Season"] == f"{player} summary before"].iloc[0]
        after = df[df["Name - Season"] == f"{player} summary after"].iloc[0]
        row = {"Zawodnik": player}
        for code, desc in fields:
            if desc in stat_choices:
                b = get_stat_value(before, code)
                a = get_stat_value(after, code)
                row[f"{desc} przed"] = b
                row[f"{desc} po"] = a
                if desc == "Udział meczów w pierwszej piątce (%)":
                    if not pd.isnull(a) and not pd.isnull(b):
                        zmiana = round(a - b, 2)
                    else:
                        zmiana = "-"
                else:
                    zmiana = round(((a - b) / b) * 100, 2) if b and not pd.isnull(b) else "-"
                row[f"{desc} % zmiana"] = zmiana
        rows.append(row)
    table = pd.DataFrame(rows)
    st.dataframe(table, use_container_width=True)

    # Interaktywny wykres
    plot_labels = [desc for desc in stat_choices]
    fig = go.Figure()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    for i, player in enumerate(players_select):
        before = df[df["Name - Season"] == f"{player} summary before"].iloc[0]
        after = df[df["Name - Season"] == f"{player} summary after"].iloc[0]
        before_vals = []
        after_vals = []
        for label in plot_labels:
            code = [c for c, d in fields if d == label][0]
            before_val = get_stat_value(before, code)
            after_val = get_stat_value(after, code)
            before_vals.append(before_val)
            after_vals.append(after_val)
        fig.add_trace(go.Bar(
            name=f"{player} Przed", x=plot_labels, y=before_vals,
            marker_color=colors[i % len(colors)], opacity=0.7, text=before_vals, textposition="auto"
        ))
        fig.add_trace(go.Bar(
            name=f"{player} Po", x=plot_labels, y=after_vals,
            marker_color=colors[(i+3) % len(colors)], opacity=0.7, text=after_vals, textposition="auto"
        ))
    fig.update_layout(barmode='group', yaxis_title="Wartość", xaxis_title="Statystyka")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Analiza danych wielowymiarowych – wybrana liga")

    stats_avail = [desc for code, desc in fields]
    # Tworzymy ramkę danych do analiz
    player_rows = []
    players_all = df[df["Name - Season"].str.contains("summary before")]["Name - Season"].str.replace(" summary before", "").tolist()
    for player in players_all:
        before = df[df["Name - Season"] == f"{player} summary before"].iloc[0]
        after = df[df["Name - Season"] == f"{player} summary after"].iloc[0]
        row = {"Zawodnik": player}
        for code, desc in fields:
            if desc in stats_avail:
                b = get_stat_value(before, code)
                a = get_stat_value(after, code)
                if code in ["GS_percent", "GS%", "GS_pct"]:
                    # Różnica udziału w pierwszej piątce (procenty)
                    zmiana = a - b if not pd.isnull(a) and not pd.isnull(b) else np.nan
                else:
                    # Procentowa zmiana dla pozostałych statystyk
                    zmiana = ((a-b)/b)*100 if b and not pd.isnull(b) else np.nan
                row[desc] = zmiana
        player_rows.append(row)
    df_zmiany = pd.DataFrame(player_rows).set_index("Zawodnik")

    # SCATTER PLOT
    st.subheader("Scatter plot (dwuwymiarowy):")
    col1, col2 = st.columns(2)
    with col1:
        stat_x = st.selectbox("Wybierz statystykę (oś X, scatter):", stats_avail, index=0)
    with col2:
        stat_y = st.selectbox("Wybierz statystykę (oś Y, scatter):", stats_avail, index=2)
    scatter_players = st.multiselect(
        "Wybierz zawodników do scatter plot:", players_all, default=players_all[:6], key="scatter_players"
    )
    scatter_all = st.checkbox("Zaznacz wszystkich zawodników do scatter", key="scatter_all")
    if scatter_all:
        scatter_players = players_all
    fig2 = go.Figure()
    for player in scatter_players:
        if player in df_zmiany.index and not (np.isnan(df_zmiany.loc[player, stat_x]) or np.isnan(df_zmiany.loc[player, stat_y])):
            fig2.add_trace(go.Scatter(
                x=[df_zmiany.loc[player, stat_x]], y=[df_zmiany.loc[player, stat_y]],
                mode='markers+text', text=[player], name=player,
                marker=dict(size=12),
                textposition="top center"
            ))
    fig2.update_layout(
        xaxis_title=f"% zmiana {stat_x}",
        yaxis_title=f"% zmiana {stat_y}",
        showlegend=False
    )
    st.plotly_chart(fig2, use_container_width=True)

    # RADAR CHART
    st.subheader("Radar chart (wielowymiarowy pająk):")
    with st.container():
        default_radar_stats = stats_avail[:3] if len(stats_avail) >= 3 else stats_avail
        radar_stats = st.multiselect(
            "Statystyki do radar/heatmapy:", 
            stats_avail, 
            default=default_radar_stats, 
            key="radar_stats"
        )
        all_stats_radar = st.checkbox("Zaznacz wszystkie statystyki do radar", key="all_stats_radar")
        if all_stats_radar and set(radar_stats) != set(stats_avail):
            radar_stats = stats_avail
        elif not all_stats_radar and set(radar_stats) == set(stats_avail):
            radar_stats = default_radar_stats

    chosen_for_radar = st.multiselect(
        "Zawodnicy do radar chart:", players_all, default=players_all[:3], key="radar_players"
    )
    radar_all = st.checkbox("Zaznacz wszystkich zawodników do radar", key="radar_all")
    if radar_all:
        chosen_for_radar = players_all

    fig3 = go.Figure()
    for player in chosen_for_radar:
        if player in df_zmiany.index:
            values = [df_zmiany.loc[player, stat] for stat in radar_stats]
            fig3.add_trace(go.Scatterpolar(
                r=values,
                theta=radar_stats,
                fill='toself',
                name=player
            ))
    fig3.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[-100, 100])),
        showlegend=True
    )
    st.plotly_chart(fig3, use_container_width=True)

    # HEATMAPA
    st.subheader("Heatmapa zmian procentowych:")
    with st.container():
        default_heatmap_stats = stats_avail[:3] if len(stats_avail) >= 3 else stats_avail
        heatmap_stats = st.multiselect(
            "Statystyki do heatmapy:", 
            stats_avail, 
            default=default_heatmap_stats, 
            key="heatmap_stats"  
        )
        all_stats_heatmap = st.checkbox("Zaznacz wszystkie statystyki do heatmapy", key="all_stats_heatmap")
        if all_stats_heatmap and set(heatmap_stats) != set(stats_avail):
            heatmap_stats = stats_avail
        elif not all_stats_heatmap and set(heatmap_stats) == set(stats_avail):
            heatmap_stats = default_heatmap_stats

    heatmap_players = st.multiselect(
        "Zawodnicy do heatmapy:", players_all, default=players_all[:8], key="heatmap_players"
    )
    heatmap_all = st.checkbox("Zaznacz wszystkich zawodników do heatmapy", key="heatmap_all")
    if heatmap_all:
        heatmap_players = players_all

    heat_data = df_zmiany.loc[heatmap_players, heatmap_stats]
    heat_data_filled = heat_data.fillna(0)
    annot = heat_data.round(1).astype(str)
    annot = annot.mask(heat_data.isna(), '-')
    fig4, ax = plt.subplots(figsize=(len(heatmap_stats)*1.5, len(heatmap_players)*0.4 + 2))
    sns.heatmap(
        heat_data_filled,
        annot=True, fmt=".1f", linewidths=.5, cmap="coolwarm", center=0, ax=ax,
        cbar_kws={"label": "% zmiana"}
    )
    plt.xlabel("Statystyka")
    plt.ylabel("Zawodnik")
    st.pyplot(fig4)


    # ====== PCA + KMeans KLASTRY ======
    st.subheader("Analiza PCA + Klasteryzacja – profile powrotu do formy")
    pca_players = st.multiselect(
        "Wybierz zawodników do PCA:", players_all, default=players_all[:12], key="pca_players"
    )
    pca_all = st.checkbox("Zaznacz wszystkich zawodników do PCA", key="pca_all")
    if pca_all:
        pca_players = players_all

    with st.container():
        # PODMIANA DEFAULT na unikalny domyślny zestaw dla Twojej ligi
        default_pca_stats = stats_avail[:5] if len(stats_avail) >= 5 else stats_avail
        pca_stats = st.multiselect(
            "Statystyki do PCA:", stats_avail, default=default_pca_stats, key="pca_stats"
        )
        all_stats_pca = st.checkbox("Zaznacz wszystkie statystyki do PCA", key="all_stats_pca")
        if all_stats_pca and set(pca_stats) != set(stats_avail):
            pca_stats = stats_avail
        elif not all_stats_pca and set(pca_stats) == set(stats_avail):
            pca_stats = default_pca_stats

    pca_data = df_zmiany.loc[pca_players, pca_stats].dropna()
    if not pca_data.empty:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(pca_data)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        explained = pca.explained_variance_ratio_ * 100

        if len(pca_data) > 1:
            max_clusters = min(8, len(pca_data))  # max 8 lub mniej, żeby nie przesadzić
            n_clusters = st.slider("Wybierz liczbę klastrów", 2, max_clusters, value=min(3, max_clusters))
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            labels = kmeans.fit_predict(X_pca)
        else:
            n_clusters = 1
            labels = [0]*len(pca_data)

        fig_pca = go.Figure()
        for i, player in enumerate(pca_data.index):
            fig_pca.add_trace(go.Scatter(
                x=[X_pca[i,0]], y=[X_pca[i,1]],
                mode='markers+text',
                marker=dict(size=14, color=labels[i], colorscale='Viridis', line=dict(width=2, color='black')),
                text=[player], textposition="top center",
                name=f"{player} (klaster {labels[i]+1})"
            ))
        fig_pca.update_layout(
            xaxis_title=f"Główna Składowa 1 ({explained[0]:.1f}% wariancji)",
            yaxis_title=f"Główna Składowa 2 ({explained[1]:.1f}% wariancji)",
            title="PCA + KMeans: Grupy zawodników wg profilu zmian",
            showlegend=False
        )
        st.plotly_chart(fig_pca, use_container_width=True)

        # Podsumowanie klastrów
        pca_data_clean = pca_data.copy()
        pca_data_clean['klaster'] = labels
        summary = pca_data_clean.groupby('klaster')[pca_stats].mean().round(1)
        summary.index = [f"Klaster {i+1}" for i in summary.index]
        st.markdown("**Średnie zmiany statystyk w każdym klastrze:**")
        st.dataframe(summary)

        st.markdown("**Zawodnicy w każdym klastrze:**")
        for k in range(n_clusters):
            players_in_cluster = pca_data_clean.index[pca_data_clean['klaster']==k].tolist()
            st.write(f"**Klaster {k+1}:** {', '.join(players_in_cluster)}")

        cluster_sums = summary.sum(axis=1)
        sorted_clusters = cluster_sums.sort_values()
        st.markdown("**Automatyczna interpretacja klastrów:**")
        for idx, clust in enumerate(summary.index):
            suma = cluster_sums[clust]
            position = sorted_clusters.index.get_loc(clust)
            if position == 0:
                desc = "Największy spadek (najgorszy powrót do formy)"
            elif position == len(sorted_clusters)-1:
                desc = "Najmniejszy spadek / największy wzrost (najlepszy powrót do formy)"
            else:
                desc = "Średni powrót do formy"
            st.write(f"- {clust}: {desc} (suma średnich zmian: {suma:.1f})")

    # ====== JAKOŚCIOWA ANALIZA ======
    st.subheader("Jakościowa analiza: które statystyki najbardziej cierpią po kontuzji ACL?")

    mean_changes = df_zmiany.mean(axis=0).sort_values()
    st.write("**Średnia procentowa zmiana każdej statystyki (wszyscy zawodnicy):**")
    st.dataframe(mean_changes.round(2).to_frame(name="Średnia zmiana (%)"))

    most_decreased = mean_changes.idxmin()
    most_increased = mean_changes.idxmax()

    st.markdown(f"""
    - **Statystyka, która najbardziej ucierpiała:**  
      **{most_decreased}** (średnia zmiana: {mean_changes[most_decreased]:.1f}%)
    - **Statystyka, która najmniej ucierpiała lub nawet wzrosła:**  
      **{most_increased}** (średnia zmiana: {mean_changes[most_increased]:.1f}%)
    """)

    st.write("**Ranking statystyk wg średniego spadku/wzrostu:**")
    for stat, change in mean_changes.items():
        if change < 0:
            trend = "spadek"
        elif change > 0:
            trend = "wzrost"
        else:
            trend = "brak zmiany"
        st.write(f"- {stat}: {change:.1f}% ({trend})")

    improved_count = (df_zmiany > 0).sum()
    most_improved_stat = improved_count.idxmax()
    st.write(f"**Najwięcej zawodników poprawiło się w statystyce:** {most_improved_stat} ({improved_count[most_improved_stat]} zawodników)")

    fig, ax = plt.subplots()
    mean_changes.plot(kind='barh', color=['red' if v<0 else 'green' for v in mean_changes], ax=ax)
    ax.set_xlabel("Średnia zmiana (%)")
    ax.set_title("Średnia procentowa zmiana statystyk po kontuzji ACL")
    st.pyplot(fig)

    # SEKCJA: Porównanie statystyk NBA vs WNBA (pokazuje się tylko po zaznaczeniu checkboxa)
    st.divider()
    porownaj = st.checkbox("Porównaj statystyki mężczyzn i kobiet (NBA vs WNBA)", value=False)

    if porownaj:
        st.header("Porównanie średnich zmian statystyk po ACL – NBA vs WNBA")

        nba_zmiany = build_stat_change(df_nba)
        wnba_zmiany = build_stat_change(df_wnba)

        nba_mean = nba_zmiany.mean().round(2)
        wnba_mean = wnba_zmiany.mean().round(2)
        nba_mean = nba_mean.dropna()
        wnba_mean = wnba_mean[nba_mean.index]

        stat_labels = nba_mean.index.tolist()
        nba_vals = nba_mean.values
        wnba_vals = wnba_mean.values

        fig, ax = plt.subplots(figsize=(max(10, len(stat_labels) * 0.7), 6))
        width = 0.35
        x = np.arange(len(stat_labels))
        ax.bar(x - width/2, nba_vals, width, label='NBA')
        ax.bar(x + width/2, wnba_vals, width, label='WNBA')
        ax.set_xticks(x)
        ax.set_xticklabels(stat_labels, rotation=45, ha="right")
        ax.set_ylabel("Średnia % zmiana po ACL")
        ax.legend()
        ax.set_title("Średnia zmiana statystyk po ACL – NBA vs WNBA")
        st.pyplot(fig)

        st.markdown("**Tabela: średnia procentowa zmiana statystyk po ACL:**")
        st.dataframe(pd.DataFrame({"NBA": nba_vals, "WNBA": wnba_vals}, index=stat_labels).round(2))

