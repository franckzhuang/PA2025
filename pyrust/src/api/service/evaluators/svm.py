from mini_keras import SVM

def evaluate_svm(json_str, input_data):
    model = SVM.from_json(json_str)
    raw_preds = model.predict([input_data])
    threshold = 0.0
    labels = [(-1 if p < threshold else 1) for p in raw_preds]

    return {"prediction": labels}