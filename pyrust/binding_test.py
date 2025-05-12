import mini_keras as mk
import numpy as np


def test_1_mlp_reg():
    print("Module mini_keras importé avec succès !")

    mlp_reg = mk.MLP(
        layers=[1, 2, 1], is_classification=False
    )

    X_train_py = [[1.0], [2.0], [3.0]]
    y_train_py = [[1.0], [2.0], [3.0]]
    print("Entraînement du MLP categorie régression...")
    mlp_reg.fit(X_train_py, y_train_py, epochs=1000, learning_rate=0.1)

    print("Entraînement terminé.")

    test_data = [[4.0], [5.0]]
    predictions_reg = mlp_reg.predict(test_data)
    print(f"Prédictions (régression) pour {test_data}: {predictions_reg}")


def test_2_mlp_classfication():

    mlp_cls = mk.MLP(
        layers=[1, 3, 1], is_classification=True
    )

    X_train_py = [[0.0], [1.0], [2.0], [3.0]]
    y_train_py = [[0.0], [0.0], [1.0], [1.0]]

    print("Entraînement du MLP catégorie classification...")
    mlp_cls.fit(X_train_py, y_train_py, epochs=1000, learning_rate=0.1)

    print("Entraînement terminé.")

    test_data = [[-1.0], [4.0]]
    predictions_cls = mlp_cls.predict(test_data)
    print(f"Prédictions (classification) pour {test_data}: {predictions_cls}")

def test_1_classification():
    print("Module mini_keras importé avec succès !")

    model_reg = mk.LinearModel(
        learning_rate=0.01, epochs=200, mode="classification", verbose=False
    )
    print("Modèle de classification créé.")

    X_train_py = [[1.0, 1.0], [2.0, 3.0], [3.0, 3.0]]
    y_train_py = [1.0, -1.0, -1.0]

    print("Entraînement du modèle de régression...")
    model_reg.fit(X_train_py, y_train_py)
    print("Entraînement terminé.")

    test_data_1 = [[3.0, 3.0], [1.0, 0.0]]
    predictions_reg = model_reg.predict(test_data_1)

    print(f"Prédictions (classification) pour {test_data_1}: {predictions_reg}")


def test_2_classification():
    X = np.concatenate([np.random.random((50, 2)) * 0.9 + np.array([1, 1]),
                        np.random.random((50, 2)) * 0.9 + np.array([2, 2])])
    Y = np.concatenate([np.ones((50, 1)), np.ones((50, 1)) * -1.0])

    model = mk.LinearModel(learning_rate=0.01, epochs=200, mode="classification", verbose=False)

    model.fit(X, Y.flatten())

    test_data = [[1.5, 1.5], [2.5, 2.5]]
    predictions = model.predict(test_data)

    print(f"Prédictions (classification) pour {test_data}: {predictions}")


def test_3_classification():
    X = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
    Y = np.array([1, 1, -1, -1])

    model = mk.LinearModel(learning_rate=0.01, epochs=1000, mode="classification", verbose=False)

    model.fit(X, Y)

    test_data = [[0.5, 0.5], [1.5, 1.5]]
    predictions = model.predict(test_data)
    print(f"Prédictions (classification) pour {test_data}: {predictions}")


def test_4_classification():
    X = np.random.random((500, 2)) * 2.0 - 1.0
    Y = np.array([1 if abs(p[0]) <= 0.3 or abs(p[1]) <= 0.3 else -1 for p in X])

    model = mk.LinearModel(learning_rate=0.01, epochs=200, mode="classification", verbose=False)

    model.fit(X, Y)

    test_data = [[0.2, 0.1], [0.8, 0.8], [-0.4, 0.0]]
    predictions = model.predict(test_data)
    print(f"Prédictions (classification) pour {test_data}: {predictions}")


def test_1_regression():
    X = np.array([[1],[2]])
    Y = np.array([2,3])

    model = mk.LinearModel(learning_rate=0.01, epochs=200, mode="regression", verbose=False)

    model.fit(X, Y)

    test_data = [[1.5], [2.5]]
    predictions = model.predict(test_data)
    print(f"Prédictions (régression) pour {test_data}: {predictions}")

def test_2_regression():
    X = np.array([
        [1],
        [2],
        [3]
    ])
    Y = np.array([
        2,
        3,
        2.5
    ])

    model = mk.LinearModel(learning_rate=0.01, epochs=200, mode="regression", verbose=False)

    model.fit(X, Y)

    test_data = [[1.5], [2.5], [3.5]]
    predictions = model.predict(test_data)
    print(f"Prédictions (régression) pour {test_data}: {predictions}")

def test_3_regression():
    X = np.array([
        [1, 1],
        [2, 2],
        [3, 1]
    ])
    Y = np.array([
        2,
        3,
        2.5
    ])

    model = mk.LinearModel(learning_rate=0.01, epochs=200, mode="regression", verbose=True)

    model.fit(X, Y)

    test_data = [[1.5, 1.5], [2.5, 2.0], [3.5, 1.0]]
    predictions = model.predict(test_data)
    print(f"Prédictions (régression) pour {test_data}: {predictions}")

def test_4_regression():
    X = np.array([
        [1, 1],
        [2, 2],
        [3, 3]
    ])
    Y = np.array([
        1,
        2,
        3
    ])

    model = mk.LinearModel(learning_rate=0.01, epochs=200, mode="regression", verbose=False)

    model.fit(X, Y)

    test_data = [[1.5, 1.5], [2.5, 2.5], [3.5, 3.5]]
    predictions = model.predict(test_data)
    print(f"Prédictions (régression) pour {test_data}: {predictions}")

def test_5_regression():
    X = np.array([
        [1, 0],
        [0, 1],
        [1, 1],
        [0, 0],
    ])
    Y = np.array([
        2,
        1,
        -2,
        -1
    ])

    model = mk.LinearModel(learning_rate=0.01, epochs=200, mode="regression", verbose=False)

    model.fit(X, Y)

    test_data = [[0.5, 0.5], [1.5, 0.5], [0.0, 0.0]]
    predictions = model.predict(test_data)
    print(f"Prédictions (régression) pour {test_data}: {predictions}")

if __name__ == "__main__":
    test_1_mlp_reg()
    # test_1_classification()
    # test_2_classification()
    # test_3_classification()
    # test_4_classification()

    # test_1_regression()
    # test_2_regression()
    # test_3_regression()
    # test_4_regression()
    # test_5_regression()
