import math
from collections import Counter

# --- Funkcje Pomocnicze ---

def is_number(s):
    try:
        float(s)
        return True
    except:
        return False
    
# Funkcja normalize_probs jest zbędna w K-NN i została usunięta.
# Normalizacja odległości może być wykonana zewnętrznie, jeśli jest potrzebna.

def euclidean_distance(row1, row2):
    """
    Oblicza odległość euklidesową (miarę podobieństwa) między dwoma wierszami (próbkami).
    Jest to najczęściej używana metryka w K-NN. 

[Image of Euclidean Distance formula]

    """
    distance = 0.0
    # Zabezpieczenie przed różną długością wektorów (choć nie powinno się zdarzyć w poprawnych danych)
    min_len = min(len(row1), len(row2))
    for i in range(min_len):
        distance += (row1[i] - row2[i])**2
    return math.sqrt(distance)

# --- Funkcja Wczytywania Danych (Minimalna modyfikacja) ---

def read_data(x_path="input_x.txt", y_path="input_y.txt"):
    """
    Wczytuje X i Y. X może mieć opcjonalny nagłówek w pierwszym wierszu.
    Format: CSV (przecinki). Wartości cech są konwertowane na floaty.
    Y: jedna etykieta na linię (0 lub 1).
    Zwraca: X (lista list float), Y (lista int 0/1), feature_names (lista lub None)
    """
    feature_names = None
    X = []
    with open(x_path, "r", encoding="utf-8") as fx:
        lines = [line.strip() for line in fx if line.strip() != ""]
        if not lines:
            raise ValueError("Plik X jest pusty.")
        # Sprawdź czy pierwszy wiersz to nagłówek
        first_tokens = [tok.strip() for tok in lines[0].split(",")]
        data_lines = lines
        if not all(is_number(tok) for tok in first_tokens):
            feature_names = first_tokens
            data_lines = lines[1:]

        # parsuj wiersze danych
        for ln in data_lines:
            toks = [tok.strip() for tok in ln.split(",")]
            if not all(is_number(tok) for tok in toks):
                raise ValueError(f"Nie wszystkie wartości cech są liczbami w wierszu: {ln}")
            X.append([float(tok) for tok in toks]) # Używamy float

    Y = []
    with open(y_path, "r", encoding="utf-8") as fy:
        for line in fy:
            line = line.strip()
            if line == "":
                continue
            tok = line.split(",")[0].strip()
            if not is_number(tok):
                raise ValueError(f"Etykieta musi być liczbą (0/1), znaleziono: {tok}")
            val = float(tok)
            if val not in (0.0, 1.0):
                raise ValueError(f"Etykiety muszą być 0 lub 1. Znaleziono: {val}")
            Y.append(int(val))
            
    if len(X) != len(Y):
        raise ValueError(f"Liczba wierszy X ({len(X)}) nie zgadza się z liczbą etykiet Y ({len(Y)}).")

    return X, Y, feature_names

# --- NOWE FUNKCJE K-NN ---

def get_neighbors(X_train, Y_train, x_test_row, k):
    """
    Znajduje k najbliższych sąsiadów dla danej próbki testowej.
    """
    distances = []
    # Oblicz odległość do wszystkich punktów treningowych
    for i, train_row in enumerate(X_train):
        dist = euclidean_distance(x_test_row, train_row)
        distances.append((dist, Y_train[i])) # (odległość, etykieta)

    # Posortuj po odległości i wybierz k najbliższych
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    
    # Zwróć tylko etykiety sąsiadów
    return [label for dist, label in neighbors]

def predict_knn(X_train, Y_train, x_test_row, k):
    """
    Przewiduje klasę dla pojedynczej próbki testowej x_test_row
    na podstawie głosowania większościowego k najbliższych sąsiadów.
    """
    # 1. Znajdź sąsiadów
    neighbor_labels = get_neighbors(X_train, Y_train, x_test_row, k)

    # 2. Głosowanie większościowe
    # Counter liczy wystąpienia każdej etykiety.
    most_common = Counter(neighbor_labels).most_common(1)
    
    if not most_common:
        # Zdarza się tylko, gdy neighbor_labels jest puste (k=0 lub błąd)
        raise ValueError("Brak etykiet sąsiadów do głosowania. Sprawdź K.")
        
    return most_common[0][0]

# --- PRZYKŁAD UŻYCIA K-NN ---

if __name__ == "__main__":
    # K-NN NIE ma funkcji train_algorithm, ponieważ 'model' to całe dane treningowe (X, Y).
    # W K-NN całe 'uczenie' odbywa się w funkcji predykcyjnej.
    
    K_NEIGHBORS = 3 # Hiperparametr K
    
    try:
        X_train, Y_train, feature_names = read_data()
    except Exception as e:
        # Przykładowy zbiór danych w przypadku braku plików
        print("Błąd przy wczytywaniu danych:", e)
        X_train = [
            [2.781, 2.550], [1.465, 2.362], [3.396, 4.400],
            [1.388, 1.850], [3.064, 3.015], [7.627, 2.759],
            [5.332, 2.088], [6.922, 1.771], [8.675, -0.242],
            [7.673, 3.508]
        ]
        Y_train = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        feature_names = ["cecha_1", "cecha_2"]
        print("Używam przykładowego zbioru danych.")

    # Dane do testowania
    x_test_row = [5.0, 3.0]
    
    # Użycie funkcji K-NN
    print(f"\nKlasyfikuję próbkę testową: {x_test_row} z K={K_NEIGHBORS}")
    
    # 1. Znalezienie sąsiadów
    neighbors = get_neighbors(X_train, Y_train, x_test_row, k=K_NEIGHBORS)
    print(f"Etykiety {K_NEIGHBORS} najbliższych sąsiadów: {neighbors}")
    
    # 2. Predykcja
    prediction = predict_knn(X_train, Y_train, x_test_row, k=K_NEIGHBORS)
    
    print(f"Przewidziana klasa (K-NN) dla {x_test_row}: {prediction}")

    # Przykładowy wiersz z danych treningowych (do weryfikacji)
    # pred_train = predict_knn(X_train, Y_train, X_train[0], k=K_NEIGHBORS) 
    # print(f"Predykcja dla pierwszego przykładu: {pred_train}, Prawdziwa etykieta: {Y_train[0]}")