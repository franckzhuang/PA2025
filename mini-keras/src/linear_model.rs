use crate::utils;

pub struct LinearModel {
    weights: Vec<f64>,
    bias: f64,
    learning_rate: f64,
    epochs: usize,
    mode: String,
    verbose: bool
}

impl LinearModel {
    pub fn new(learning_rate: f64, epochs: usize, mode: String, verbose: bool) -> Self {
        Self {
            weights: Vec::new(),
            bias: 0.0,
            learning_rate,
            epochs,
            mode,
            verbose
        }
    }

    fn linear_output(&self, x: Vec<f64>) -> f64 {
        let mut linear_output = self.bias;
        for (i, &weight) in self.weights.iter().enumerate() {
            linear_output += weight * x[i];
        }
        linear_output
    }

    pub fn predict(&self, x: Vec<Vec<f64>>) -> Vec<f64> {
        let mut y_preds: Vec<f64> = Vec::with_capacity(x.len());

        for x_i in x {
            let pred = self.linear_output(x_i.to_vec());
            if self.mode == "classification" {
                y_preds.push(if pred >= 0.0 { 1.0 } else { -1.0 });
            } else {
                y_preds.push(pred);
            }
        }
        y_preds
    }

    fn gradient_descent(&self, x: Vec<Vec<f64>>, y: Vec<f64>) -> (Vec<f64>, f64) {
        let m = x.len();
        let n = x[0].len();
        let mut dw = vec![0.0; n];
        let mut db = 0.0;

        for (x_i, &y_true) in x.iter().zip(y.iter()) {
            let y_pred = self.linear_output(x_i.to_vec());
            let error = if self.mode == "classification" {
                (if y_pred >= 0.0 { 1.0 } else { -1.0 }) - y_true
            } else {
                y_pred - y_true
            };

            for j in 0..n {
                dw[j] += 2.0 * x_i[j] * error;
            }
            db += 2.0 * error;
        }

        for i in 0..n {
            dw[i] /= m as f64;
        }
        db /= m as f64;

        (dw, db)
    }


    pub fn fit(&mut self, x_train: Vec<Vec<f64>>, y_train: Vec<f64>) {
        if x_train.len() != y_train.len() {
            panic!("Error: x_data and y_data must have the same length.");
        }
        let n = x_train[0].len();
        self.weights = vec![0.0; n];

        for epoch in 0..self.epochs {
            let (dw, db) = self.gradient_descent(x_train.to_vec(), y_train.to_vec());

            for (weight, &gradient) in self.weights.iter_mut().zip(dw.iter()) {
                *weight -= self.learning_rate * gradient;
            }
            self.bias -= self.learning_rate * db;
            if self.verbose && epoch % 100 == 0 {
                let predictions = self.predict(x_train.to_vec());

                if self.mode == "classification" {
                    let acc = utils::accuracy(predictions.to_vec(), y_train.to_vec());
                    println!("Epoch {}, Accuracy: {:.2}%", epoch,  acc * 100.0);
                } else {
                    let mse = utils::mse(predictions.clone().to_vec(), y_train.to_vec());
                    println!("Epoch {}, MSE: {:.4}", epoch, mse);
                }
            }
        }
    }
}