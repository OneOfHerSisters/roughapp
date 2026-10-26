# Struktura pracy dyplomowej — roughapp

> **System webowy do analizy i eliminacji niespójności w tabelach decyzyjnych**  
> w oparciu o teorię zbiorów przybliżonych (Rough Set Theory)  
>
> Repozytorium: [https://github.com/OneOfHerSisters/roughapp](https://github.com/OneOfHerSisters/roughapp)  
> Stack: Python 3.13 · FastAPI · React 19 · PostgreSQL · Docker

---

## Jak czytać ten dokument

Ten plik łączy **Twój plan** z **strukturą przykładowej pracy inżynierskiej** (system rezerwacji wycieczek).  
Każdy rozdział zawiera:

- **Cel rozdziału** — po co jest w pracy,
- **Propozycję spisu podrozdziałów**,
- **Wskazówki / szkic treści** — co napisać,
- **Odniesienie do diagramów** — gdzie wstawić rysunki.

Oznaczenia:


| Symbol | Znaczenie                                                        |
| ------ | ---------------------------------------------------------------- |
| ✅      | Treść już przygotowana (w tym dokumencie lub w `docs/diagrams/`) |
| 📝     | Do uzupełnienia przez autora                                     |
| 🖼     | Wymagany rysunek / diagram                                       |


---



## Mapowanie: Twój plan → rozdziały pracy


| Element z Twojego planu           | Gdzie w pracy                                                             |
| --------------------------------- | ------------------------------------------------------------------------- |
| Wykorzystane technologie          | **4.3** Architektura systemu + **6.** Dokumentacja techniczna             |
| Opis projektu                     | **1. Wstęp** + **4.** (wstęp do rozdz.) + **5.1** (krótki opis aplikacji) |
| Podstawowe funkcje aplikacji      | **4.5** Interfejs użytkownika + **5.2** Opis funkcji                      |
| Wymagania funkcjonalne + use-case | **4.1** + 🖼 diagram przypadków użycia                                    |
| Diagram aktywności                | **4.1** lub **5.3** (scenariusze) + 🖼 diagramy aktywności                |
| Link do GitHub, opis techniczny   | **6.** Dokumentacja techniczna (wstęp rozdz.)                             |
| Opis aplikacji                    | **5.1** + **5.2**                                                         |
| Dokumentacja użytkownika          | **5.** cały rozdział                                                      |


---



## Spis treści (propozycja)

```
Streszczenie (PL) .............................................. 📝
Abstract (EN) .................................................. 📝

1. Wstęp ....................................................... 📝
   1.1 Cel i zakres pracy ...................................... 📝
   1.2 Metodyka i narzędzia .................................... 📝

2. Teoretyczne podstawy analizy niespójności w tabelach decyzyjnych ... 📝
   2.1 Tabela decyzyjna i atrybuty warunkowe ................... 📝
   2.2 Klasy równoważności i niespójność ....................... 📝
   2.3 Miary jakości: γ, α, ρ, L, U ............................ 📝
   2.4 Strategie eliminacji niespójności ....................... 📝
   2.5 Podsumowanie ............................................ 📝

3. Przegląd istniejących narzędzi ............................... 📝
   3.1 RSES — Rough Set Exploration System ................... 📝
   3.2 Inne implementacje i biblioteki ......................... 📝
   3.3 Porównanie i uzasadnienie własnego rozwiązania .......... 📝
   3.4 Podsumowanie ............................................ 📝

4. Projekt systemu roughapp ..................................... ✅/📝
   4.1 Wymagania funkcjonalne .................................. ✅
   4.2 Wymagania niefunkcjonalne ............................... 📝
   4.3 Architektura systemu .................................... ✅
   4.4 Wykorzystane technologie ................................ ✅
   4.5 Model danych ............................................ 📝
   4.6 Interfejs użytkownika i podstawowe funkcje .............. ✅
   4.7 Diagram przypadków użycia .............................. ✅ 🖼
   4.8 Diagramy aktywności ..................................... ✅ 🖼

5. Dokumentacja użytkowa ........................................ 📝
   5.1 Opis aplikacji .......................................... 📝
   5.2 Instalacja i wymagania sprzętowe ........................ 📝
   5.3 Opis interfejsu i funkcji systemu ....................... 📝
   5.4 Przykładowe scenariusze użytkowania ..................... 📝

6. Dokumentacja techniczna ...................................... 📝
   6.1 Repozytorium i struktura projektu ....................... ✅
   6.2 Warstwa serwerowa ....................................... 📝
   6.3 Warstwa kliencka ........................................ 📝
   6.4 Baza danych i migracje .................................. 📝
   6.5 Obiekty transferu danych (DTO) .......................... 📝
   6.6 Wdrożenie (Docker Compose) .............................. 📝

7. Testy systemu ................................................ 📝
   7.1 Testy jednostkowe ....................................... 📝
   7.2 Analiza wyników testów .................................. 📝

8. Podsumowanie i kierunki rozwoju .............................. 📝

Bibliografia .................................................... 📝
Spis rysunków ................................................... 📝
Spis tabel ...................................................... 📝
Załączniki (opcjonalnie: pełne diagramy PlantUML) .............. ✅
```

---



# 1. Wstęp

**Cel rozdziału:** wprowadzić czytelnika w temat, uzasadnić potrzebę systemu, sformułować cel pracy.

### 1.1 Cel i zakres pracy

**Szkic (do rozwinięcia):**

Niniejsza praca inżynierska dotyczy projektowania i implementacji systemu webowego **roughapp**, umożliwiającego analizę niespójności w tabelach decyzyjnych oraz stosowanie strategii ich eliminacji w oparciu o teorię zbiorów przybliżonych.

**Celem pracy** jest stworzenie aplikacji internetowej, która:

- oblicza miary jakości klasyfikacji (γ, α) i identyfikuje klasy niespójne,
- umożliwia porównanie pięciu strategii rozwiązywania niespójności,
- wspiera badania eksperymentalne na zbiorach danych w formacie CSV,
- udostępnia interfejs graficzny do wizualizacji wyników i drzew decyzyjnych.

**Zakres pracy obejmuje:**

- analizę teoretyczną problemu niespójności w tabelach decyzyjnych,
- przegląd istniejących narzędzi (w tym RSES),
- projekt architektury i wymagań systemu,
- implementację backendu (Python/FastAPI) i frontendu (React),
- dokumentację użytkową i techniczną,
- testy jednostkowe algorytmów rozwiązywania.

**Poza zakresem** pozostają: uczenie maszynowe poza drzewem decyzyjnym, obsługa formatów innych niż CSV, aplikacja mobilna.

### 1.2 Metodyka i narzędzia

Krótki akapit: implementacja iteracyjna, repozytorium Git, Docker do wdrożenia, pytest do testów, analiza porównawcza strategii na zbiorach testowych z literatury/dypлomu.

---



# 2. Teoretyczne podstawy

**Cel rozdziału:** odpowiednik rozdz. 2 z przykładowej pracy (problematyka branżowa), ale dla teorii zbiorów przybliżonych.

**W przykładzie:** charakterystyka biur podróży, metody rezerwacji, wyzwania.  
**U Ciebie:** definicje formalne, metryki, strategie — to fundament dla rozdz. 4 i 7.

### Proponowane podrozdziały


| §   | Temat               | Co opisać                                                             |
| --- | ------------------- | --------------------------------------------------------------------- |
| 2.1 | Tabela decyzyjna    | Obiekty, atrybuty warunkowe C, decyzja D, format CSV                  |
| 2.2 | Klasy równoważności | IND(C), niespójność gdy |D| > 1 w klasie                              |
| 2.3 | Miary               | γ (quality), α (accuracy), ρ, dolne/górne approximations L/U          |
| 2.4 | Strategie           | MCD, Qualitative Removal, Splitting, MVD, GD — definicje z literatury |
| 2.5 | Podsumowanie        | Most do rozdz. 3                                                      |


🖼 **Rysunek 2.1** — schemat tabeli decyzyjnej z zaznaczoną klasą niespójną (opcjonalnie).

---



# 3. Przegląd istniejących narzędzi

**Cel rozdziału:** odpowiednik rozdz. 3 z przykładu (Skaleo, TravelPay…).  
Uzasadnia, **dlaczego własny system**, a nie tylko RSES.

### Proponowane podrozdziały


| §   | Temat                                                        |
| --- | ------------------------------------------------------------ |
| 3.1 | RSES — funkcje, ograniczenia, brak nowoczesnego API webowego |
| 3.2 | Inne narzędzia (ROSE2, biblioteki Python/R)                  |
| 3.3 | Tabela porównawcza + wnioski projektowe                      |
| 3.4 | Podsumowanie — lista funkcji, które roughapp ma dodać        |


🖼 **Tabela 3.1** — porównanie RSES vs roughapp (web UI, REST API, eksperymenty batch, JWT, Docker…).

---



# 4. Projekt systemu roughapp

**Cel rozdziału:** odpowiednik rozdz. 4 z przykładu — serce pracy inżynierskiej.  
Tu ląduje większość elementów z Twojego planu.

---



## 4.1 Wymagania funkcjonalne ✅

**W przykładzie:** tabela wymagań (Tabela 4.1) + opis każdego.  
**U Ciebie:** tabela WF-01 … WF-16 + diagram use-case.


| ID    | Wymaganie                                          | Priorytet |
| ----- | -------------------------------------------------- | --------- |
| WF-01 | Rejestracja użytkownika (e-mail, hasło ≥ 8 znaków) | Wysoki    |
| WF-02 | Logowanie i uwierzytelnianie JWT                   | Wysoki    |
| WF-03 | Przegląd wbudowanych zbiorów danych bez logowania  | Wysoki    |
| WF-04 | Upload pliku CSV przez zalogowanego użytkownika    | Wysoki    |
| WF-05 | Podgląd tabeli z paginacją                         | Wysoki    |
| WF-06 | Analiza niespójności (γ, α, ρ, L, U, boundary)     | Wysoki    |
| WF-07 | Obliczanie ważności atrybutów warunkowych          | Średni    |
| WF-08 | Budowa i wizualizacja drzewa decyzyjnego           | Wysoki    |
| WF-09 | Wprowadzanie niespójności (3 metody)               | Wysoki    |
| WF-10 | Uzupełnianie brakujących wartości (2 metody)       | Średni    |
| WF-11 | Uruchamianie eksperymentu z wybranymi strategiami  | Wysoki    |
| WF-12 | Zapis i prezentacja wyników eksperymentu           | Wysoki    |
| WF-13 | Porównanie strategii na wykresach                  | Wysoki    |
| WF-14 | Pobieranie wyników CSV/ZIP                         | Średni    |
| WF-15 | Zapisanie wyniku eksperymentu jako nowy zbiór      | Średni    |
| WF-16 | Usuwanie własnych zbiorów i eksperymentów          | Średni    |


🖼 **Rysunek 4.1** — diagram przypadków użycia → plik `docs/diagrams/use-case-roughapp.puml`

---



## 4.2 Wymagania niefunkcjonalne 📝

**W przykładzie:** wydajność, bezpieczeństwo, skalowalność, HTTPS.  
**Propozycja tabeli 4.2:**


| ID     | Wymaganie      | Opis                                                           |
| ------ | -------------- | -------------------------------------------------------------- |
| WNF-01 | Bezpieczeństwo | Hasła hashowane bcrypt; tokeny JWT; SECRET_KEY z env           |
| WNF-02 | Dostępność     | Aplikacja webowa w przeglądarce; responsywny layout            |
| WNF-03 | Przenośność    | Docker Compose; działanie na Linux/macOS/Windows               |
| WNF-04 | Utrzymywalność | Warstwy core / services / api / db; migracje Alembic           |
| WNF-05 | Testowalność   | pytest; ≥ 46 testów jednostkowych algorytmów                   |
| WNF-06 | Wydajność      | Eksperyment na tabelach do ~10⁴ wierszy w czasie akceptowalnym |
| WNF-07 | Obsługa danych | Format CSV; opcjonalny wiersz typów kolumn (c/n/i/r/l)         |


---



## 4.3 Architektura systemu ✅

System oparty na modelu **klient–serwer** z modułowym monolitem backendu i oddzielnym SPA.

**Trzy warstwy:**

1. Prezentacja — React SPA + Nginx
2. Aplikacja — FastAPI (Uvicorn)
3. Dane — PostgreSQL + system plików (CSV)

**Moduły backendu:** `core/` → `services/` → `api/` → `db/`

🖼 **Rysunek 4.2** — diagram komponentów → `docs/diagrams/component-architecture.puml`  
🖼 **Rysunek 4.3** — diagram wdrożenia Docker Compose (opcjonalnie)

**Tekst do wklejenia:** patrz wcześniejsza sekcja „4.3 System Architecture" (wersja EN) — przetłumacz na PL.

---



## 4.4 Wykorzystane technologie ✅



### Warstwa serwerowa


| Technologia                 | Zastosowanie                          |
| --------------------------- | ------------------------------------- |
| Python 3.13                 | Implementacja algorytmów i API        |
| FastAPI + Uvicorn           | REST API, walidacja Pydantic          |
| SQLAlchemy + Alembic        | ORM, migracje schematu                |
| PostgreSQL 16               | Baza produkcyjna                      |
| pandas, NumPy, scikit-learn | Przetwarzanie tabel, drzewa decyzyjne |
| python-jose, bcrypt         | JWT, hashowanie haseł                 |
| pytest                      | Testy jednostkowe                     |




### чв


| Technologia           | Zastosowanie                |
| --------------------- | --------------------------- |
| React 19 + TypeScript | Interfejs użytkownika (SPA) |
| Vite 8                | Bundler, dev server         |
| React Router 7        | Routing po stronie klienta  |
| TanStack React Query  | Cache zapytań HTTP          |
| Axios                 | Klient REST                 |
| Recharts              | Wykresy porównawcze         |




### Infrastruktura


| Technologia             | Zastosowanie                           |
| ----------------------- | -------------------------------------- |
| Docker + Docker Compose | Konteneryzacja (db, api, frontend)     |
| Nginx                   | Serwowanie SPA, reverse proxy `/api/*` |


---



## 4.5 Model danych 📝

**W przykładzie:** diagram ER, opis encji.  
**U Ciebie:** 4 tabele SQLAlchemy.


| Encja                | Kluczowe pola                                                                                     | Relacje                                 |
| -------------------- | ------------------------------------------------------------------------------------------------- | --------------------------------------- |
| **User**             | id, email, password_hash, created_at                                                              | 1→N Dataset, Experiment                 |
| **Dataset**          | id, name, path, rows, cols, source_id, gamma_actual, user_id                                      | N→1 User; self-ref source_id            |
| **Experiment**       | id, dataset_id, user_id, random_state, ran_at                                                     | N→1 User, Dataset; 1→N ExperimentResult |
| **ExperimentResult** | strategy, gamma_before/after, alpha_before/after, rows_*, time_ms, result_path(s), details (JSON) | N→1 Experiment                          |


🖼 **Rysunek 4.4** — diagram ER (User — Dataset — Experiment — ExperimentResult)  
📁 Pliki CSV poza bazą: `data/uploads/`, `data/experiment_results/`

---



## 4.6 Interfejs użytkownika i podstawowe funkcje aplikacji ✅



### Podział ról


| Rola                      | Dostęp                                                  |
| ------------------------- | ------------------------------------------------------- |
| **Gość**                  | Wbudowane zbiory, analiza, drzewo, ważność atrybutów    |
| **Użytkownik zalogowany** | Upload, transformacje, eksperymenty, zarządzanie danymi |




### Mapowanie ekranów (strony React)


| Strona           | Ścieżka               | Funkcje                                                      |
| ---------------- | --------------------- | ------------------------------------------------------------ |
| Datasets         | `/`                   | Lista zbiorów, upload, analiza, introduce, complete, preview |
| Experiments      | `/experiments`        | Konfiguracja i uruchomienie eksperymentu                     |
| Results          | `/experiments/:id`    | Metryki, drzewa przed/po, pobieranie CSV                     |
| Comparison       | `/results`            | Wykresy porównawcze strategii                                |
| My Data          | `/my-data`            | Usuwanie własnych zbiorów i eksperymentów                    |
| Login / Register | `/login`, `/register` | Uwierzytelnianie                                             |


🖼 **Rysunek 4.5** — zrzuty ekranu głównych widoków (wstaw po wykonaniu screenshotów)

### Pięć strategii rozwiązywania


| Skrót     | Nazwa                       |
| --------- | --------------------------- |
| MCD       | Most Common Decision        |
| QR        | Qualitative Removal         |
| Splitting | Podział na podtabele spójne |
| MVD       | Many-Valued Decision        |
| GD        | Generalized Decision        |


---



## 4.7 Diagram przypadków użycia ✅ 🖼

Plik źródłowy: `docs/diagrams/use-case-roughapp.puml`  
Eksport: **Rysunek 4.6** w pracy.

---



## 4.8 Diagramy aktywności ✅ 🖼


| Diagram                   | Plik                       | Rysunek w pracy         |
| ------------------------- | -------------------------- | ----------------------- |
| Uruchomienie eksperymentu | `activity-experiment.puml` | Rys. 4.7                |
| Wprowadzenie niespójności | `activity-introduce.puml`  | Rys. 4.8                |
| Analiza niespójności      | `activity-analyze.puml`    | Rys. 4.9                |
| Rejestracja / logowanie   | `activity-auth.puml`       | Rys. 4.10 (opcjonalnie) |


---



# 5. Dokumentacja użytkowa

**Cel rozdziału:** odpowiednik rozdz. 5 z przykładu — instrukcja dla użytkownika końcowego.

---



## 5.1 Opis aplikacji 📝

Krótki opis „co to jest roughapp" dla użytkownika nietechnicznego:

> roughapp to aplikacja webowa służąca do analizy tabel decyzyjnych. Użytkownik ładuje plik CSV (lub wybiera wbudowany zbiór), sprawdza stopień niespójności, a następnie może zastosować algorytmy eliminacji niespójności i porównać ich skuteczność.

Adres aplikacji po wdrożeniu: `http://localhost:8080` (domyślnie Docker Compose).

---



## 5.2 Instalacja i wymagania sprzętowe 📝

**W przykładzie:** Java, PostgreSQL, MongoDB, wymagania RAM.  
**U Ciebie:**


| Wymaganie               | Minimalne                          |
| ----------------------- | ---------------------------------- |
| Docker + Docker Compose | v24+                               |
| RAM                     | 4 GB                               |
| Dysk                    | 2 GB wolnego miejsca               |
| Przeglądarka            | Chrome / Firefox / Edge (aktualna) |


**Procedura instalacji (skrót):**

```bash
git clone https://github.com/OneOfHerSisters/roughapp.git
cd roughapp
cp .env.example .env          # opcjonalnie: SECRET_KEY, hasła DB
docker compose up --build
```

Aplikacja dostępna pod: `http://localhost:8080`

---



## 5.3 Opis interfejsu i funkcji systemu 📝

**Wzorzec z przykładu:** podrozdział na każdą funkcję (5.2.1, 5.2.2…).  
**Propozycja dla roughapp:**


| §      | Temat                            | Co opisać + screenshot         |
| ------ | -------------------------------- | ------------------------------ |
| 5.3.1  | Rejestracja i logowanie          | Formularze, JWT w sesji        |
| 5.3.2  | Przegląd zbiorów danych          | Lista builtin + własne         |
| 5.3.3  | Podgląd i analiza tabeli         | Preview, γ, α, per-decision    |
| 5.3.4  | Ważność atrybutów                | Tabela wpływu na γ             |
| 5.3.5  | Drzewo decyzyjne                 | Komponent TreeView             |
| 5.3.6  | Wprowadzanie niespójności        | 3 metody + parametry           |
| 5.3.7  | Uzupełnianie braków              | remove / fill                  |
| 5.3.8  | Uruchamianie eksperymentu        | Wybór strategii, seed          |
| 5.3.9  | Wyniki eksperymentu              | Tabela metryk, drzewa przed/po |
| 5.3.10 | Porównanie strategii             | Wykresy Recharts               |
| 5.3.11 | Pobieranie i zapisywanie wyników | CSV, ZIP (Splitting)           |
| 5.3.12 | Zarządzanie danymi (My Data)     | Usuwanie zbiorów/eksperymentów |


---



## 5.4 Przykładowe scenariusze użytkowania 📝

**W przykładzie:** scenariusz klienta biura podróży + administratora.  
**Propozycja 2 scenariuszy:**

### Scenariusz A — Gość analizuje wbudowany zbiór

1. Wejście na stronę główną bez logowania
2. Wybór zbioru „cars"
3. Kliknięcie „Analyze" → odczyt γ i α
4. Budowa drzewa decyzyjnego

🖼 Diagram aktywności: `activity-analyze.puml`

### Scenariusz B — Badacz przeprowadza eksperyment

1. Rejestracja / logowanie
2. Upload CSV lub introduce niespójności na zbiorze builtin
3. Experiments → wybór datasetu, strategii MCD+MVD+GD
4. Analiza wyników, porównanie na /results
5. Pobranie CSV, zapisanie wyniku jako nowy dataset

🖼 Diagram aktywności: `activity-experiment.puml`

---



# 6. Dokumentacja techniczna

**Cel rozdziału:** odpowiednik rozdz. 6 z przykładu + **link GitHub** i opis techniczny z Twojego planu.

---



## 6.1 Repozytorium i struktura projektu ✅

**Repozytorium:** [https://github.com/OneOfHerSisters/roughapp](https://github.com/OneOfHerSisters/roughapp)

```
roughapp/
├── roughapp/           # pakiet Python
│   ├── core/           # prymitywy rough sets
│   ├── services/       # logika biznesowa
│   ├── api/routers/    # endpointy REST
│   ├── db/             # modele SQLAlchemy
│   ├── auth.py
│   └── config.py
├── frontend/src/       # React SPA
├── tests/              # pytest
├── alembic/            # migracje
├── data/               # CSV (seed, uploads, results)
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---



## 6.2 Warstwa serwerowa 📝

Opisać:

- punkt wejścia: `roughapp/api/__init__.py` → `FastAPI(app)`
- routery: `auth`, `datasets`, `experiments`, `analysis`, `resolution`
- przepływ: router → service → core
- `experiment_runner.py` — orchestracja 5 strategii
- startup: `init_db()` + `seed_builtin_datasets()`

---



## 6.3 Warstwa kliencka 📝

Opisać:

- `frontend/src/main.tsx`, `App.tsx`, routing
- `api/client.ts` — Axios + interceptory JWT
- `auth.tsx` — kontekst użytkownika
- `TreeView.tsx`, strony w `pages/`

---



## 6.4 Baza danych i migracje 📝

- SQLAlchemy models w `db/models.py`
- Alembic: `alembic upgrade head` przy starcie
- PostgreSQL w Docker; SQLite lokalnie

---



## 6.5 Obiekty transferu danych (DTO) 📝

**W przykładzie:** osobny podrozdział 6.3 o MapStruct/DTO.  
**U Ciebie:** Pydantic w `services/schemas.py` + modele odpowiedzi w routerach (`DatasetOut`, `ExperimentOut`, `InconsistencyReport`…).

---



## 6.6 Wdrożenie (Docker Compose) 📝

Opisać 3 serwisy z `docker-compose.yml`:

- **db** — PostgreSQL, volume `pgdata`
- **api** — uvicorn :8000, volumes uploads/results
- **frontend** — nginx :80 → proxy `/api/` → api:8000

Zmienne środowiskowe: `.env.example` (`DATABASE_URL`, `SECRET_KEY`, `POSTGRES_*`, `WEB_PORT`).

---



# 7. Testy systemu 📝

**W przykładzie:** testy jednostkowe, integracyjne, systemowe, bezpieczeństwa.


| Plik                       | Zakres                    | Liczba testów |
| -------------------------- | ------------------------- | ------------- |
| `tests/test_resolution.py` | 5 strategii rozwiązywania | 30            |
| `tests/test_completion.py` | uzupełnianie braków       | 16            |


Uruchomienie: `pytest tests/ -v`

📝 Do rozszerzenia: opis przypadków testowych, tabela wyników, wnioski.

---



# 8. Podsumowanie i kierunki rozwoju 📝

**Szkic:**

- zrealizowano system webowy z 5 strategiami i modułem eksperymentów,
- potwierdzono poprawność algorytmów testami jednostkowymi,
- **kierunki rozwoju:** więcej formatów danych, role użytkowników, eksport raportów PDF, API publiczne, optymalizacja Splitting dla dużych tabel.

---



# Załączniki


| Zał. | Zawartość                                                   |
| ---- | ----------------------------------------------------------- |
| A    | Pełne listingi PlantUML (`docs/diagrams/*.puml`)            |
| B    | Przykładowe pliki CSV                                       |
| C    | Instrukcja uruchomienia lokalnego bez Dockera (opcjonalnie) |


---



# Spis rysunków (propozycja)


| Nr            | Tytuł                                         |
| ------------- | --------------------------------------------- |
| Rys. 2.1      | Przykład klasy niespójnej w tabeli decyzyjnej |
| Rys. 3.1      | Diagram porównania narzędzi (opcjonalnie)     |
| Rys. 4.1      | Diagram przypadków użycia systemu roughapp    |
| Rys. 4.2      | Diagram komponentów architektury              |
| Rys. 4.3      | Diagram wdrożenia Docker Compose              |
| Rys. 4.4      | Diagram encji-relacji bazy danych             |
| Rys. 4.5      | Widoki interfejsu użytkownika                 |
| Rys. 4.7–4.10 | Diagramy aktywności                           |
| Rys. 5.x      | Zrzuty ekranu dokumentacji użytkowej          |


---



# Spis tabel (propozycja)


| Nr       | Tytuł                       |
| -------- | --------------------------- |
| Tab. 3.1 | Porównanie RSES i roughapp  |
| Tab. 4.1 | Wymagania funkcjonalne      |
| Tab. 4.2 | Wymagania niefunkcjonalne   |
| Tab. 4.3 | Wykorzystane technologie    |
| Tab. 4.4 | Encje modelu danych         |
| Tab. 7.1 | Wyniki testów jednostkowych |


---

*Ostatnia aktualizacja struktury: 2026-07-04*