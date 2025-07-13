from mini_keras import SVM

def evaluate_svm(json_str, input_data):
    model = SVM.from_json(json_str)
    preds = model.predict([input_data])
    return {"prediction": preds}
