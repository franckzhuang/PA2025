from mini_keras import LinearClassification

def evaluate_linear(json_str, input_data):
    model = LinearClassification.from_json(json_str)
    pred = model.predict(input_data)
    return {"prediction": pred}
