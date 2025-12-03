import math
from collections import Counter

# --- Funkcje Pomocnicze ---

def is_number(s):
    try:
        float(s)
        return True
    except:
        return False

def euclidean_distance(row1, row2):
    """
    Oblicza odległość euklidesową (miarę podobieństwa) między dwoma wierszami (próbkami).
    [Image of Euclidean Distance formula]
    """
    distance = 0.0
    # Zabezpieczenie przed różną długością wektorów (choć nie powinno się zdarzyć w poprawnych danych)
    min_len = min(len(row1), len(row2))
    for i in range(min_len):
        distance += (row1[i] - row2[i])**2
    return math.sqrt(distance)

# --- Funkcja Wczytywania Danych (bez zmian) ---

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

# --- Funkcje NN (K-NN z K=1) ---

def get_nearest_neighbor(X_train, Y_train, x_test_row):
    """
    Znajduje etykietę pojedynczego, najbliższego sąsiada (K=1).
    """
    distances = []
    # Oblicz odległość do wszystkich punktów treningowych
    for i, train_row in enumerate(X_train):
        dist = euclidean_distance(x_test_row, train_row)
        distances.append((dist, Y_train[i])) # (odległość, etykieta)

    # Posortuj po odległości i wybierz 1 najbliższy
    distances.sort(key=lambda x: x[0])
    
    # Zwróć etykietę pierwszego elementu
    if distances:
        return distances[0][1]
    else:
        raise ValueError("Brak danych treningowych.")

def predict_nn(X_train, Y_train, x_test_row):
    """
    Przewiduje klasę dla pojedynczej próbki testowej x_test_row
    na podstawie etykiety najbliższego sąsiada (NN).
    """
    # 1. Znajdź najbliższego sąsiada
    prediction = get_nearest_neighbor(X_train, Y_train, x_test_row)
    
    # 2. Predykcja jest po prostu etykietą tego sąsiada
    return prediction

# --- PRZYKŁAD UŻYCIA NN ---

if __name__ == "__main__":
    # Wartość K jest teraz ZAWSZE 1 (implikowana) dla NN
    
    try:
        X_train, Y_train, feature_names = read_data()
    except Exception as e:
        # Przykładowy zbiór danych w przypadku braku plików
        print("Błąd przy wczytywaniu danych:", e)
        X_train = [
            [2.781, 2.550], [1.465, 2.362], [3.396, 4.400], # Klasa 0
            [7.627, 2.759], [5.332, 2.088], [6.922, 1.771]  # Klasa 1
        ]
        Y_train = [0, 0, 0, 1, 1, 1]
        feature_names = ["cecha_1", "cecha_2"]
        print("Używam przykładowego zbioru danych.")

    # Dane do testowania
    x_test_row = [5.0, 3.0] # Spodziewana klasa 1 (bliżej [7.627, 2.759])
    
    # Użycie funkcji NN
    print(f"\nKlasyfikuję próbkę testową: {x_test_row}")
    
    # Predykcja NN (używa K=1 wewnątrz)
    prediction = predict_nn(X_train, Y_train, x_test_row)
    
    print(f"Przewidziana klasa (Nearest Neighbor): {prediction}")