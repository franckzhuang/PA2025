pub fn mse(predictions: Vec<f64>, actual: Vec<f64>) -> f64 {
    if predictions.len() != actual.len() {
        panic!("Error: predictions and actual values must have the same length");
    }

    if predictions.is_empty() {
        return 0.0;
    }

    let mut sum_squared_error = 0.0;
    for (pred, act) in predictions.iter().zip(actual.iter()) {
        let error = pred - act;
        sum_squared_error += error * error;
    }

    sum_squared_error / predictions.len() as f64
}

pub fn accuracy(predictions: Vec<f64>, actual: Vec<f64>) -> f64 {
    if predictions.len() != actual.len() {
        panic!("Error: predictions and actual values must have the same length");
    }

    if predictions.is_empty() {
        return 0.0;
    }

    let mut correct_count = 0;
    for (pred, act) in predictions.iter().zip(actual.iter()) {
        if pred * act > 0.0 {
            correct_count += 1;
        }
    }

    correct_count as f64 / predictions.len() as f64
}