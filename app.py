import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="ACL Injury Analysis", page_icon="🦵", layout="wide")

# ====================================================================
# TRANSLATIONS
# ====================================================================
TEXTS = {
    "pl": {
        "app_title": "Wpływ kontuzji ACL na powrót do sportu",
        "app_intro": (
            "Interaktywna analiza statystyk zawodników NBA, WNBA, NFL i piłki nożnej "
            "przed i po zerwaniu więzadła krzyżowego przedniego (ACL). "
            "Projekt z pracy licencjackiej (bioinformatyka, Uniwersytet Gdański)."
        ),
        "choose_league": "Wybierz ligę do szczegółowej analizy:",
        "choose_nfl": "Wybierz rodzaj statystyk NFL:",
        "league_title": "Analiza wybranej ligi: {league}",
        "tab_compare": "🔢 Porównanie zawodników",
        "tab_multi": "📊 Analiza wielowymiarowa",
        "compare_header": "Porównanie wybranych zawodników przed i po kontuzji",
        "choose_players": "Wybierz zawodników do porównania (domyślnie 2):",
        "all_players": "Zaznacz wszystkich zawodników",
        "choose_stats": "Wybierz statystyki do porównania:",
        "all_stats": "Zaznacz wszystkie statystyki",
        "player": "Zawodnik",
        "before": "przed",
        "after": "po",
        "pct_change": "% delta",
        "before_cap": "Przed",
        "after_cap": "Po",
        "value": "Wartość",
        "statistic": "Statystyka",
        "multi_header": "Analiza danych wielowymiarowych – wybrana league",
        "scatter_title": "Wykres punktowy (dwuwymiarowy):",
        "scatter_x": "Wybierz statystykę (oś X):",
        "scatter_y": "Wybierz statystykę (oś Y):",
        "scatter_players": "Wybierz zawodników do wykresu punktowego:",
        "scatter_all": "Zaznacz wszystkich zawodników do wykresu punktowego",
        "radar_title": "Wykres radarowy (wielowymiarowy):",
        "radar_stats": "Statystyki do wykresu radarowego:",
        "radar_all_stats": "Zaznacz wszystkie statystyki do wykresu radarowego",
        "radar_players": "Zawodnicy do wykresu radarowego:",
        "radar_all": "Zaznacz wszystkich zawodników do wykresu radarowego",
        "heatmap_title": "Heatmapa zmian procentowych:",
        "heatmap_stats": "Statystyki do heatmapy:",
        "heatmap_all_stats": "Zaznacz wszystkie statystyki do heatmapy",
        "heatmap_players": "Zawodnicy do heatmapy:",
        "heatmap_all": "Zaznacz wszystkich zawodników do heatmapy",
        "pca_title": "Analiza PCA + klasteryzacja – profile powrotu do formy",
        "pca_players": "Wybierz zawodników do PCA:",
        "pca_all": "Zaznacz wszystkich zawodników do PCA",
        "pca_stats": "Statystyki do PCA:",
        "pca_all_stats": "Zaznacz wszystkie statystyki do PCA",
        "n_clusters": "Wybierz liczbę klastrów",
        "cluster": "Klaster",
        "pc": "Główna składowa {n} ({var:.1f}% wariancji)",
        "pca_plot_title": "PCA + k-means: grupy zawodników wg profilu zmian",
        "cluster_means": "**Średnie zmiany statystyk w każdym klastrze:**",
        "cluster_players": "**Zawodnicy w każdym klastrze:**",
        "cluster_interp": "**Automatyczna interpretacja klastrów:**",
        "cluster_worst": "Największy spadek (najgorszy powrót do formy)",
        "cluster_best": "Najmniejszy spadek / największy wzrost (najlepszy powrót do formy)",
        "cluster_mid": "Średni powrót do formy",
        "cluster_sum": "suma średnich zmian",
        "pca_too_few": "Za mało danych do PCA – wybierz więcej zawodników lub statystyk.",
        "quality_title": "Analiza jakościowa: które statystyki najbardziej cierpią po kontuzji ACL?",
        "mean_change_all": "**Średnia procentowa delta każdej statystyki (wszyscy zawodnicy):**",
        "mean_change_col": "Średnia delta (%)",
        "most_decreased": "Statystyka, która najbardziej ucierpiała:",
        "most_increased": "Statystyka, która najmniej ucierpiała lub nawet wzrosła:",
        "mean_change_short": "średnia delta",
        "ranking": "**Ranking statystyk wg średniego spadku/wzrostu:**",
        "trend_down": "spadek",
        "trend_up": "wzrost",
        "trend_none": "brak zmiany",
        "most_improved": "**Najwięcej zawodników poprawiło się w statystyce:** {stat} ({n} zawodników)",
        "mean_change_plot": "Średnia procentowa delta statystyk po kontuzji ACL",
        "compare_sexes": "Porównaj statystyki mężczyzn i kobiet (NBA vs WNBA)",
        "sexes_header": "Porównanie średnich zmian statystyk po ACL – NBA vs WNBA",
        "sexes_ylabel": "Średnia % delta po ACL",
        "sexes_title": "Średnia delta statystyk po ACL – NBA vs WNBA",
        "sexes_table": "**Tabela: średnia procentowa delta statystyk po ACL:**",
        "leagues": {
            "NBA": "NBA",
            "WNBA": "WNBA",
            "NFL": "NFL",
            "Soccer-Man": "Piłka nożna (mężczyźni)",
            "Soccer-Woman": "Piłka nożna (kobiety)",
        },
    },
    "en": {
        "app_title": "How ACL injuries affect return to sport",
        "app_intro": (
            "Interactive analysis of NBA, WNBA, NFL and football (soccer) player statistics "
            "before and after an anterior cruciate ligament (ACL) tear. "
            "Bachelor's thesis project (Bioinformatics, University of Gdańsk)."
        ),
        "choose_league": "Choose a league to analyse:",
        "choose_nfl": "Choose the NFL statistics category:",
        "league_title": "League analysis: {league}",
        "tab_compare": "🔢 Player comparison",
        "tab_multi": "📊 Multivariate analysis",
        "compare_header": "Selected players before and after the injury",
        "choose_players": "Choose players to compare (2 by default):",
        "all_players": "Select all players",
        "choose_stats": "Choose statistics to compare:",
        "all_stats": "Select all statistics",
        "player": "Player",
        "before": "before",
        "after": "after",
        "pct_change": "% change",
        "before_cap": "Before",
        "after_cap": "After",
        "value": "Value",
        "statistic": "Statistic",
        "multi_header": "Multivariate analysis – selected league",
        "scatter_title": "Scatter plot (two dimensions):",
        "scatter_x": "Statistic on the X axis:",
        "scatter_y": "Statistic on the Y axis:",
        "scatter_players": "Players in the scatter plot:",
        "scatter_all": "Select all players for the scatter plot",
        "radar_title": "Radar chart (multidimensional):",
        "radar_stats": "Statistics for the radar chart:",
        "radar_all_stats": "Select all statistics for the radar chart",
        "radar_players": "Players in the radar chart:",
        "radar_all": "Select all players for the radar chart",
        "heatmap_title": "Heatmap of percentage changes:",
        "heatmap_stats": "Statistics for the heatmap:",
        "heatmap_all_stats": "Select all statistics for the heatmap",
        "heatmap_players": "Players in the heatmap:",
        "heatmap_all": "Select all players for the heatmap",
        "pca_title": "PCA + clustering – return-to-form profiles",
        "pca_players": "Players for PCA:",
        "pca_all": "Select all players for PCA",
        "pca_stats": "Statistics for PCA:",
        "pca_all_stats": "Select all statistics for PCA",
        "n_clusters": "Number of clusters",
        "cluster": "Cluster",
        "pc": "Principal component {n} ({var:.1f}% of variance)",
        "pca_plot_title": "PCA + k-means: player groups by change profile",
        "cluster_means": "**Mean change of each statistic per cluster:**",
        "cluster_players": "**Players in each cluster:**",
        "cluster_interp": "**Automatic cluster interpretation:**",
        "cluster_worst": "Largest decline (worst return to form)",
        "cluster_best": "Smallest decline / largest improvement (best return to form)",
        "cluster_mid": "Average return to form",
        "cluster_sum": "sum of mean changes",
        "pca_too_few": "Not enough data for PCA – choose more players or statistics.",
        "quality_title": "Which statistics suffer most after an ACL injury?",
        "mean_change_all": "**Mean percentage change of each statistic (all players):**",
        "mean_change_col": "Mean change (%)",
        "most_decreased": "Statistic that suffered most:",
        "most_increased": "Statistic that suffered least or even improved:",
        "mean_change_short": "mean change",
        "ranking": "**Statistics ranked by mean decline/improvement:**",
        "trend_down": "decline",
        "trend_up": "improvement",
        "trend_none": "no change",
        "most_improved": "**Most players improved in:** {stat} ({n} players)",
        "mean_change_plot": "Mean percentage change of statistics after an ACL injury",
        "compare_sexes": "Compare men and women (NBA vs WNBA)",
        "sexes_header": "Mean change after ACL injury – NBA vs WNBA",
        "sexes_ylabel": "Mean % change after ACL",
        "sexes_title": "Mean change of statistics after ACL – NBA vs WNBA",
        "sexes_table": "**Table: mean percentage change after ACL:**",
        "leagues": {
            "NBA": "NBA",
            "WNBA": "WNBA",
            "NFL": "NFL",
            "Soccer-Man": "Football (men)",
            "Soccer-Woman": "Football (women)",
        },
    },
}

# --------- LANGUAGE ---------
# English by default; a link with ?lang=pl opens the Polish version.
LANGUAGES = {"en": "🇬🇧 English", "pl": "🇵🇱 Polski"}


def reset_widgets():
    """Clear widget selections after a language change, because statistic names change.
    The chosen league and NFL category are carried over to the new language."""
    new_lang = st.session_state["lang"]
    keep = {
        name: value
        for name in ("league", "nfl_sheet")
        for key, value in st.session_state.items()
        if key.startswith(f"{name}_")
    }
    for key in list(st.session_state.keys()):
        if key != "lang":
            del st.session_state[key]
    for name, value in keep.items():
        st.session_state[f"{name}_{new_lang}"] = value
    st.query_params["lang"] = new_lang


if "lang" not in st.session_state:
    url_lang = st.query_params.get("lang", "en")
    st.session_state["lang"] = url_lang if url_lang in LANGUAGES else "en"

_, lang_col = st.columns([5, 1])
with lang_col:
    lang = st.radio(
        "Language / Język",
        options=list(LANGUAGES),
        format_func=LANGUAGES.get,
        key="lang",
        horizontal=True,
        label_visibility="collapsed",
        on_change=reset_widgets,
    )

T = TEXTS[lang]


def t(key, **kwargs):
    return T[key].format(**kwargs) if kwargs else T[key]


def k(name):
    """Separate widget keys per language, so that after switching languages
    Streamlit does not show stale (untranslated) selections."""
    return f"{name}_{lang}"


# --------- STATISTIC FIELDS: (Excel column code, PL label, EN label) ---------
fields_nba = [
    ("games played", "Mecze rozegrane", "Games played"),
    ("GS_percent", "Udział meczów w pierwszej piątce (%)", "Games started (%)"),
    ("Minutes played per match", "Minuty/mecz", "Minutes per game"),
    ("PTS", "Punkty", "Points"),
    ("AST", "Asysty", "Assists"),
    ("REB", "Zbiórki", "Rebounds"),
    ("OR", "Zbiórki atak", "Offensive rebounds"),
    ("DR", "Zbiórki obrona", "Defensive rebounds"),
    ("BLK", "Bloki", "Blocks"),
    ("STL", "Przechwyty", "Steals"),
    ("FG_made", "Rzuty z gry (trafione)", "Field goals made"),
    ("FG_att", "Rzuty z gry (próby)", "Field goals attempted"),
    ("FG%", "Skuteczność z gry (%)", "Field goal %"),
    ("3PT_made", "Rzuty za 3 (trafione)", "3-pointers made"),
    ("3PT_att", "Rzuty za 3 (próby)", "3-pointers attempted"),
    ("3PT%", "Skuteczność za 3 (%)", "3-point %"),
    ("FT_made", "Rzuty wolne (trafione)", "Free throws made"),
    ("FT_att", "Rzuty wolne (próby)", "Free throws attempted"),
    ("FT%", "Skuteczność rzutów wolnych (%)", "Free throw %"),
    ("TO", "Straty", "Turnovers"),
    ("PF", "Faule", "Personal fouls"),
]
fields_nfl_passing = [
    ("GP", "Mecze rozegrane", "Games played"),
    ("CMP", "Celne podania", "Completions"),
    ("ATT", "Próby podań", "Pass attempts"),
    ("CMP%", "Celność podań (%)", "Completion %"),
    ("YDS", "Yardy podań", "Passing yards"),
    ("AVG", "Yardy na próbę", "Yards per attempt"),
    ("TD", "Przyłożenia podaniem", "Passing touchdowns"),
    ("INT", "Przechwyty podań", "Interceptions thrown"),
    ("LNG", "Najdłuższe podanie", "Longest pass"),
    ("SACK", "Sacks", "Sacks"),
    ("RTG", "Ocena podającego", "Passer rating"),
    ("QBR", "QBR", "QBR"),
]
fields_nfl_rushing = [
    ("GP", "Mecze rozegrane", "Games played"),
    ("CAR", "Próby biegu", "Rushing attempts"),
    ("YDS", "Yardy biegu", "Rushing yards"),
    ("AVG", "Średnia jardów/bieg", "Yards per carry"),
    ("TD", "Przyłożenia biegiem", "Rushing touchdowns"),
    ("LNG", "Najdłuższy bieg", "Longest run"),
    ("FD", "Pierwsze próby biegiem", "Rushing first downs"),
    ("FUM", "Fumble", "Fumbles"),
    ("LST", "Fumble stracone", "Fumbles lost"),
]
fields_nfl_receiving = [
    ("GP", "Mecze rozegrane", "Games played"),
    ("REC", "Złapane podania", "Receptions"),
    ("TGTS", "Celem podania", "Targets"),
    ("YDS", "Yardy po złapaniu", "Receiving yards"),
    ("AVG", "Średnia jardów/odbiory", "Yards per reception"),
    ("TD", "Przyłożenia", "Touchdowns"),
    ("LNG", "Najdłuższe złapane", "Longest reception"),
    ("FD", "Pierwsze próby", "First downs"),
    ("FUM", "Fumble", "Fumbles"),
    ("LST", "Fumble stracone", "Fumbles lost"),
]
fields_nfl_defense = [
    ("GP", "Mecze rozegrane", "Games played"),
    ("TOT", "Łącznie tackli", "Total tackles"),
    ("SOLO", "Solo tackli", "Solo tackles"),
    ("AST", "Asysty w tacklach", "Assisted tackles"),
    ("SACK", "Sacki", "Sacks"),
    ("FF", "Forced fumble", "Forced fumbles"),
    ("FR", "Odzyskane fumble", "Fumble recoveries"),
    ("YDS", "Yardy przechwytów/fumble", "Return yards (INT/fumble)"),
    ("INT", "Interceptions", "Interceptions"),
    ("AVG", "Średnia jardów na interception", "Yards per interception"),
    ("TD", "Przyłożenia po obronie", "Defensive touchdowns"),
    ("LNG", "Najdłuższa akcja", "Longest play"),
    ("PD", "Passes defended", "Passes defended"),
    ("STF", "Stuffs", "Stuffs"),
    ("STFYDS", "Stuff yards", "Stuff yards"),
    ("KB", "Zablokowane kopnięcia", "Blocked kicks"),
]
fields_nfl_scoring = [
    ("GP", "Mecze rozegrane", "Games played"),
    ("PASS", "TD po podaniu", "Passing TD"),
    ("RUSH", "TD po biegu", "Rushing TD"),
    ("REC", "TD po odbiorze", "Receiving TD"),
    ("RET", "Return TD", "Return TD"),
    ("TD", "Wszystkie touchdowny", "Total touchdowns"),
    ("2PT", "Dwupunktowe", "Two-point conversions"),
    ("PAT", "Extra point kick", "Extra points"),
    ("FG", "Field goals", "Field goals"),
    ("PTS", "Punkty", "Points"),
]
fields_soccer = [
    ("Matches played", "Mecze rozegrane", "Matches played"),
    ("Matches started", "Mecze w pierwszym składzie", "Matches started"),
    ("PPG", "Punkty na mecz", "Points per game"),
    ("Goals per match", "Gole na mecz", "Goals per match"),
    ("Assists per match", "Asysty na mecz", "Assists per match"),
    ("Own goals", "Samobóje", "Own goals"),
    ("Substitutions in", "Wejścia z ławki", "Substituted in"),
    ("Substitution out", "Zejścia z boiska", "Substituted out"),
    ("Yellow card per match", "Żółte kartki na mecz", "Yellow cards per match"),
    ("Second yellow card per match", "Drugie żółte na mecz", "Second yellows per match"),
    ("Red cards per match", "Czerwone kartki na mecz", "Red cards per match"),
    ("Penalty goals", "Gole z karnych", "Penalty goals"),
    ("Minutes per goal", "Co ile minut gol", "Minutes per goal"),
    ("Minutes played per match summary", "Minuty/mecz", "Minutes per match"),
]

# Statistics measured as a percentage-point difference instead of a percentage change
GS_CODES = ["GS_percent", "GS%", "GS_pct"]


def localize(raw_fields):
    """(code, PL, EN) -> (code, label in the selected language)"""
    return [(code, pl if lang == "pl" else en) for code, pl, en in raw_fields]


# --------- MAPPING TO EXCEL COLUMNS ---------
def get_excel_col(field_code):
    # Made/attempted values are stored together as "made-attempted" in one column
    if field_code.startswith("FG_"):
        return "Field goals made-attempted per game"
    if field_code.startswith("3PT_"):
        return "Three-point field goals made-attempted per game"
    if field_code.startswith("FT_"):
        return "Free throws made-attempted per game"
    return field_code


# --------- SPLITTING "made-attempted" VALUES ---------
def split_value(val):
    if pd.isnull(val):
        return (np.nan, np.nan)
    s = str(val).replace(",", ".")
    if "-" in s:
        try:
            a, b = [x.strip() for x in s.split("-")]
            return (round(float(a), 2), round(float(b), 2))
        except ValueError:
            return (np.nan, np.nan)
    try:
        return (round(float(s), 2), np.nan)
    except ValueError:
        return (np.nan, np.nan)


# --------- READING A STATISTIC FROM A ROW ---------
def get_stat_value(row, field_code):
    # Special case for GS% – share of games started
    if field_code in GS_CODES:
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


def change(code, before_val, after_val):
    """Change of a statistic: percentage points for GS%, percentage change for everything else."""
    if code in GS_CODES:
        if pd.isnull(after_val) or pd.isnull(before_val):
            return np.nan
        return after_val - before_val
    if before_val and not pd.isnull(before_val):
        return ((after_val - before_val) / before_val) * 100
    return np.nan


# --------- LOADING DATA ---------
@st.cache_data
def load_data():
    path = "ACL_injury_data.xlsx"
    df_nba = pd.read_excel(path, sheet_name="Basketball-man")
    df_wnba = pd.read_excel(path, sheet_name="Basketball-woman")
    nfl_sheets = {
        "Passing": pd.read_excel(path, sheet_name="NFL - Passing"),
        "Rushing": pd.read_excel(path, sheet_name="NFL - Rushing"),
        "Receiving": pd.read_excel(path, sheet_name="NFL - Receiving"),
        "Defense": pd.read_excel(path, sheet_name="NFL - Defense"),
        "Scoring": pd.read_excel(path, sheet_name="NFL - Scoring"),
    }
    df_soccer_man = pd.read_excel(path, sheet_name="Soccer-Man")
    df_soccer_woman = pd.read_excel(path, sheet_name="Soccer-Woman")
    return df_nba, df_wnba, nfl_sheets, df_soccer_man, df_soccer_woman


df_nba, df_wnba, nfl_sheets, df_soccer_man, df_soccer_woman = load_data()


def players_in(frame):
    return (
        frame[frame["Name - Season"].str.contains("summary before")]["Name - Season"]
        .str.replace(" summary before", "")
        .tolist()
    )


def before_after(frame, player):
    before = frame[frame["Name - Season"] == f"{player} summary before"].iloc[0]
    after = frame[frame["Name - Season"] == f"{player} summary after"].iloc[0]
    return before, after


# --------- HEADER ---------
st.title(t("app_title"))
st.caption(t("app_intro"))

# --------- LEAGUE SELECTION ---------
# Only show leagues whose Excel sheet contains data
league_frames = {"NBA": df_nba, "WNBA": df_wnba, "Soccer-Man": df_soccer_man, "Soccer-Woman": df_soccer_woman}
available_leagues = [
    lg for lg in ["NBA", "WNBA", "NFL", "Soccer-Man", "Soccer-Woman"]
    if lg == "NFL" or not league_frames[lg].empty
]
league = st.radio(
    t("choose_league"),
    available_leagues,
    format_func=lambda key: T["leagues"][key],
    horizontal=True,
    key=k("league"),
)
if league == "NBA":
    df = df_nba
    fields = localize(fields_nba)
elif league == "WNBA":
    df = df_wnba
    fields = localize(fields_nba)
elif league == "NFL":
    nfl_sheet = st.selectbox(t("choose_nfl"), list(nfl_sheets.keys()), index=0, key=k("nfl_sheet"))
    df = nfl_sheets[nfl_sheet]
    fields = localize(
        {
            "Passing": fields_nfl_passing,
            "Rushing": fields_nfl_rushing,
            "Receiving": fields_nfl_receiving,
            "Defense": fields_nfl_defense,
            "Scoring": fields_nfl_scoring,
        }[nfl_sheet]
    )
elif league == "Soccer-Man":
    df = df_soccer_man
    fields = localize(fields_soccer)
else:
    df = df_soccer_woman
    fields = localize(fields_soccer)


# --------- STATISTIC CHANGES ---------
def build_stat_change(frame, frame_fields):
    rows = []
    for player in players_in(frame):
        before, after = before_after(frame, player)
        rows.append(
            {
                desc: change(code, get_stat_value(before, code), get_stat_value(after, code))
                for code, desc in frame_fields
            }
        )
    return pd.DataFrame(rows)


def multiselect_with_all(label, options, default, key, all_label):
    """Multiselect with a 'select all' checkbox."""
    chosen = st.multiselect(label, options, default=default, key=k(key))
    if st.checkbox(all_label, key=k(f"{key}_all")):
        chosen = options
    return chosen


# ---------------------- DETAILED ANALYSIS -------------------------
st.divider()
st.header(t("league_title", league=T["leagues"][league]))

tab1, tab2 = st.tabs([t("tab_compare"), t("tab_multi")])
players_all = players_in(df)
stats_avail = [desc for code, desc in fields]
code_of = {desc: code for code, desc in fields}

with tab1:
    st.subheader(t("compare_header"))
    players_select = multiselect_with_all(
        t("choose_players"), players_all, players_all[:2], "compare_players", t("all_players")
    )
    stat_choices = multiselect_with_all(
        t("choose_stats"), stats_avail, stats_avail[:3], "compare_stats", t("all_stats")
    )

    # Comparison table
    rows = []
    for player in players_select:
        before, after = before_after(df, player)
        row = {t("player"): player}
        for desc in stat_choices:
            code = code_of[desc]
            b = get_stat_value(before, code)
            a = get_stat_value(after, code)
            row[f"{desc} {t('before')}"] = b
            row[f"{desc} {t('after')}"] = a
            delta = change(code, b, a)
            row[f"{desc} {t('pct_change')}"] = np.nan if pd.isnull(delta) else round(delta, 2)
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), width="stretch")

    # Interactive chart
    fig = go.Figure()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728",
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    for i, player in enumerate(players_select):
        before, after = before_after(df, player)
        before_vals = [get_stat_value(before, code_of[label]) for label in stat_choices]
        after_vals = [get_stat_value(after, code_of[label]) for label in stat_choices]
        fig.add_trace(go.Bar(
            name=f"{player} {t('before_cap')}", x=stat_choices, y=before_vals,
            marker_color=colors[i % len(colors)], opacity=0.7, text=before_vals, textposition="auto",
        ))
        fig.add_trace(go.Bar(
            name=f"{player} {t('after_cap')}", x=stat_choices, y=after_vals,
            marker_color=colors[(i + 3) % len(colors)], opacity=0.7, text=after_vals, textposition="auto",
        ))
    fig.update_layout(barmode="group", yaxis_title=t("value"), xaxis_title=t("statistic"))
    st.plotly_chart(fig, width="stretch")

with tab2:
    st.subheader(t("multi_header"))

    # Data frame with changes for all players
    player_rows = []
    for player in players_all:
        before, after = before_after(df, player)
        row = {t("player"): player}
        for code, desc in fields:
            row[desc] = change(code, get_stat_value(before, code), get_stat_value(after, code))
        player_rows.append(row)
    df_changes = pd.DataFrame(player_rows).set_index(t("player"))

    # SCATTER PLOT
    st.markdown(f"#### {t('scatter_title')}")
    col1, col2 = st.columns(2)
    with col1:
        stat_x = st.selectbox(t("scatter_x"), stats_avail, index=0, key=k("scatter_x"))
    with col2:
        stat_y = st.selectbox(t("scatter_y"), stats_avail, index=min(2, len(stats_avail) - 1), key=k("scatter_y"))
    scatter_players = multiselect_with_all(
        t("scatter_players"), players_all, players_all[:6], "scatter_players", t("scatter_all")
    )
    fig2 = go.Figure()
    for player in scatter_players:
        x_val, y_val = df_changes.loc[player, stat_x], df_changes.loc[player, stat_y]
        if not (pd.isnull(x_val) or pd.isnull(y_val)):
            fig2.add_trace(go.Scatter(
                x=[x_val], y=[y_val], mode="markers+text", text=[player], name=player,
                marker=dict(size=12), textposition="top center",
            ))
    fig2.update_layout(
        xaxis_title=f"{t('pct_change')} {stat_x}",
        yaxis_title=f"{t('pct_change')} {stat_y}",
        showlegend=False,
    )
    st.plotly_chart(fig2, width="stretch")

    # RADAR CHART
    st.markdown(f"#### {t('radar_title')}")
    radar_stats = multiselect_with_all(
        t("radar_stats"), stats_avail, stats_avail[:3], "radar_stats", t("radar_all_stats")
    )
    chosen_for_radar = multiselect_with_all(
        t("radar_players"), players_all, players_all[:3], "radar_players", t("radar_all")
    )
    fig3 = go.Figure()
    for player in chosen_for_radar:
        values = [df_changes.loc[player, stat] for stat in radar_stats]
        fig3.add_trace(go.Scatterpolar(r=values, theta=radar_stats, fill="toself", name=player))
    fig3.update_layout(polar=dict(radialaxis=dict(visible=True, range=[-100, 100])), showlegend=True)
    st.plotly_chart(fig3, width="stretch")

    # HEATMAP
    st.markdown(f"#### {t('heatmap_title')}")
    heatmap_stats = multiselect_with_all(
        t("heatmap_stats"), stats_avail, stats_avail[:3], "heatmap_stats", t("heatmap_all_stats")
    )
    heatmap_players = multiselect_with_all(
        t("heatmap_players"), players_all, players_all[:8], "heatmap_players", t("heatmap_all")
    )
    if heatmap_stats and heatmap_players:
        heat_data = df_changes.loc[heatmap_players, heatmap_stats]
        fig4, ax = plt.subplots(figsize=(len(heatmap_stats) * 1.5 + 2, len(heatmap_players) * 0.4 + 2))
        sns.heatmap(
            heat_data.fillna(0), annot=True, fmt=".1f", linewidths=0.5, cmap="coolwarm", center=0,
            ax=ax, cbar_kws={"label": t("pct_change")},
        )
        ax.set_xlabel(t("statistic"))
        ax.set_ylabel(t("player"))
        st.pyplot(fig4)

    # ====== PCA + KMEANS ======
    st.markdown(f"#### {t('pca_title')}")
    pca_players = multiselect_with_all(
        t("pca_players"), players_all, players_all[:12], "pca_players", t("pca_all")
    )
    pca_stats = multiselect_with_all(
        t("pca_stats"), stats_avail, stats_avail[:5], "pca_stats", t("pca_all_stats")
    )

    pca_data = df_changes.loc[pca_players, pca_stats].dropna()
    if len(pca_data) >= 2 and len(pca_stats) >= 2:
        X_scaled = StandardScaler().fit_transform(pca_data)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        explained = pca.explained_variance_ratio_ * 100

        max_clusters = min(8, len(pca_data))
        if max_clusters > 2:
            n_clusters = st.slider(t("n_clusters"), 2, max_clusters, value=min(3, max_clusters), key=k("n_clusters"))
        else:
            n_clusters = 2
        labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(X_pca)

        fig_pca = go.Figure()
        for i, player in enumerate(pca_data.index):
            fig_pca.add_trace(go.Scatter(
                x=[X_pca[i, 0]], y=[X_pca[i, 1]], mode="markers+text",
                marker=dict(size=14, color=labels[i], colorscale="Viridis",
                            cmin=0, cmax=n_clusters - 1, line=dict(width=2, color="black")),
                text=[player], textposition="top center",
                name=f"{player} ({t('cluster')} {labels[i] + 1})",
            ))
        fig_pca.update_layout(
            xaxis_title=t("pc", n=1, var=explained[0]),
            yaxis_title=t("pc", n=2, var=explained[1]),
            title=t("pca_plot_title"),
            showlegend=False,
        )
        st.plotly_chart(fig_pca, width="stretch")

        # Cluster summary
        pca_data_clean = pca_data.copy()
        pca_data_clean["cluster"] = labels
        summary = pca_data_clean.groupby("cluster")[pca_stats].mean().round(1)
        summary.index = [f"{t('cluster')} {i + 1}" for i in summary.index]
        st.markdown(t("cluster_means"))
        st.dataframe(summary)

        st.markdown(t("cluster_players"))
        for cluster_id in range(n_clusters):
            members = pca_data_clean.index[pca_data_clean["cluster"] == cluster_id].tolist()
            st.write(f"**{t('cluster')} {cluster_id + 1}:** {', '.join(members)}")

        cluster_sums = summary.sum(axis=1)
        sorted_clusters = cluster_sums.sort_values()
        st.markdown(t("cluster_interp"))
        for clust in summary.index:
            position = sorted_clusters.index.get_loc(clust)
            if position == 0:
                desc = t("cluster_worst")
            elif position == len(sorted_clusters) - 1:
                desc = t("cluster_best")
            else:
                desc = t("cluster_mid")
            st.write(f"- {clust}: {desc} ({t('cluster_sum')}: {cluster_sums[clust]:.1f})")
    else:
        st.info(t("pca_too_few"))

    # ====== WHICH STATISTICS SUFFER MOST ======
    st.markdown(f"#### {t('quality_title')}")
    mean_changes = df_changes.mean(axis=0).dropna().sort_values()
    st.write(t("mean_change_all"))
    st.dataframe(mean_changes.round(2).to_frame(name=t("mean_change_col")))

    most_decreased = mean_changes.idxmin()
    most_increased = mean_changes.idxmax()
    st.markdown(
        f"- **{t('most_decreased')}** **{most_decreased}** "
        f"({t('mean_change_short')}: {mean_changes[most_decreased]:.1f}%)\n"
        f"- **{t('most_increased')}** **{most_increased}** "
        f"({t('mean_change_short')}: {mean_changes[most_increased]:.1f}%)"
    )

    st.write(t("ranking"))
    for stat, value in mean_changes.items():
        trend = t("trend_down") if value < 0 else t("trend_up") if value > 0 else t("trend_none")
        st.write(f"- {stat}: {value:.1f}% ({trend})")

    improved_count = (df_changes > 0).sum()
    most_improved_stat = improved_count.idxmax()
    st.write(t("most_improved", stat=most_improved_stat, n=improved_count[most_improved_stat]))

    fig, ax = plt.subplots()
    mean_changes.plot(kind="barh", color=["red" if v < 0 else "green" for v in mean_changes], ax=ax)
    ax.set_xlabel(t("mean_change_col"))
    ax.set_title(t("mean_change_plot"))
    st.pyplot(fig)

    # NBA vs WNBA comparison
    st.divider()
    if st.checkbox(t("compare_sexes"), value=False, key=k("compare_sexes")):
        st.markdown(f"#### {t('sexes_header')}")
        basketball_fields = localize(fields_nba)
        nba_mean = build_stat_change(df_nba, basketball_fields).mean().round(2).dropna()
        wnba_mean = build_stat_change(df_wnba, basketball_fields).mean().round(2)[nba_mean.index]

        stat_labels = nba_mean.index.tolist()
        fig, ax = plt.subplots(figsize=(max(10, len(stat_labels) * 0.7), 6))
        width = 0.35
        x = np.arange(len(stat_labels))
        ax.bar(x - width / 2, nba_mean.values, width, label="NBA")
        ax.bar(x + width / 2, wnba_mean.values, width, label="WNBA")
        ax.set_xticks(x)
        ax.set_xticklabels(stat_labels, rotation=45, ha="right")
        ax.set_ylabel(t("sexes_ylabel"))
        ax.legend()
        ax.set_title(t("sexes_title"))
        st.pyplot(fig)

        st.markdown(t("sexes_table"))
        st.dataframe(pd.DataFrame({"NBA": nba_mean.values, "WNBA": wnba_mean.values}, index=stat_labels).round(2))
