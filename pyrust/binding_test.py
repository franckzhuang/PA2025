import mini_keras as mk

if __name__ == "__main__":
    print("Module mk importé avec succès !")

    try:
        model_reg = mk.LinearModel(
            learning_rate=0.01, epochs=100, mode="regression", verbose=True
        )
        print("Modèle de régression créé.")

        X_train_py = [[1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [2.0, 2.0]]
        y_train_py = [2.0, 3.0, 3.0, 4.0]

        print("Entraînement du modèle de régression...")
        model_reg.fit(X_train_py, y_train_py)
        print("Entraînement terminé.")

        test_data_1 = [[3.0, 3.0], [1.0, 0.0]]
        predictions_reg = model_reg.predict(test_data_1)
        print(f"Prédictions (régression) pour {test_data_1}: {predictions_reg}")

        # ---------------------------------------

        model_clf = mk.LinearModel(
            learning_rate=0.1, epochs=50, mode="classification", verbose=False
        )
        print("\nModèle de classification créé.")

        X_train_clf_py = [[1.0, 1.0], [2.0, 2.0], [-1.0, -1.0], [-2.0, -1.0]]
        y_train_clf_py = [1.0, 1.0, -1.0, -1.0]

        print("Entraînement du modèle de classification...")
        model_clf.fit(X_train_clf_py, y_train_clf_py)
        print("Entraînement terminé.")

        test_data_2 = [[-3.0, -3.0], [3.0, 2.0]]
        predictions_clf = model_clf.predict([[-3.0, -3.0], [3.0, 2.0]])
        print(f"Prédictions (classification) pour {test_data_2}: {predictions_clf}")

    except Exception as e:
        print(f"Une erreur est survenue pendant le test : {e}")
        import traceback

        traceback.print_exc()
