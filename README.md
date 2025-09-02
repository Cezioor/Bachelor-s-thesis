# Analiza wpływu urazów ACL na powrót do sportu

## 📌 Opis projektu

Projekt realizowany w ramach pracy dyplomowej dotyczy analizy
statystycznej wpływu kontuzji więzadła krzyżowego przedniego (ACL) na
powrót sportowców do pełnej sprawności.\
W analizie wykorzystano dane z kilku dyscyplin sportowych: - 🏀 NBA
(koszykówka mężczyzn)\
- 🏀 WNBA (koszykówka kobiet)\
- 🏈 NFL (futbol amerykański)\
- ⚽ Piłka nożna

Aplikacja umożliwia: - porównywanie statystyk zawodników **przed i po
kontuzji**, - analizę głównych komponentów (PCA) i klasteryzację
zawodników, - wizualizację zmian w różnych kategoriach statystyk, -
ocenę jakości powrotu do gry po urazie.

------------------------------------------------------------------------

## ⚙️ Technologie

-   Python 3.x\
-   Streamlit (interfejs użytkownika)\
-   Pandas, NumPy (analiza danych)\
-   Scikit-learn (PCA, klasteryzacja)\
-   Matplotlib / Seaborn (wizualizacje)\
-   OpenPyXL (obsługa plików Excel)

------------------------------------------------------------------------

## 🚀 Uruchamianie projektu

1.  Sklonuj repozytorium:

    ``` bash
    git clone https://github.com/twoj-login/twoje-repozytorium.git
    cd twoje-repozytorium
    ```

2.  Zainstaluj wymagane biblioteki:

    ``` bash
    pip install -r requirements.txt
    ```

3.  Uruchom aplikację:

    ``` bash
    streamlit run app.py
    ```

------------------------------------------------------------------------

## 📊 Dane wejściowe

Projekt wykorzystuje pliki Excela zawierające statystyki zawodników.\
Przykładowy plik (`Excel-licencjat.xlsx`) zawiera kilka arkuszy
odpowiadających różnym dyscyplinom.\
Każdy arkusz można wczytać osobno i analizować w aplikacji.

------------------------------------------------------------------------

## 📈 Przykładowe wyniki

Aplikacja generuje: - **heatmapy** zmian statystyk zawodników,\
- **wykresy PCA** pokazujące profile powrotu do gry,\
- **tabele klasteryzacji** z automatyczną interpretacją wyników.

------------------------------------------------------------------------

## 📌 Cel projektu

Celem projektu jest stworzenie narzędzia, które pozwoli na: - lepsze
zrozumienie wpływu kontuzji ACL na sportowców,\
- porównanie dynamiki powrotu do gry w różnych dyscyplinach,\
- stworzenie podstaw do dalszych badań naukowych nad prewencją i
rehabilitacją urazów.

------------------------------------------------------------------------

✍️ Autor: *Cezary Muszalski*
