pub struct LinearClassification {
    pub weights: Vec<f64>,
    learning_rate: f64,
    max_iterations: usize,
    verbose: bool,
}

impl LinearClassification {
    pub fn new(learning_rate: f64, max_iterations: usize, verbose: bool) -> Self {
        Self {
            weights: Vec::new(),
            learning_rate,
            max_iterations,
            verbose,
        }
    }

    pub fn predict(&self, x_data: &[f64]) -> f64 {
        let activation: f64 = x_data.iter()
            .zip(self.weights.iter())
            .map(|(x, w)| w * x)
            .sum();
        if activation > 0.0 { 1.0 } else { 0.0 }
    }

    pub fn fit(&mut self, x_data: &[Vec<f64>], y_data: &[f64]) {
        if x_data.len() != y_data.len() {
            panic!("Error: x and y must have the same length.");
        }

        let x_data_bias: Vec<Vec<f64>> = x.iter()
            .map(|x_i| {
                let mut v = vec![1.0];
                v.extend(x_i.iter().cloned());
                v
            })
            .collect();

        let n = x_data_bias[0].len();
        self.weights = vec![0.0; n];

        for iter in 0..self.max_iterations {
            let mut errors = 0;
            for (x, &y) in x_data_bias.iter().zip(y_data.iter()) {
                let y_pred = self.predict(x);
                if y_pred != y {
                    self.weights = self.weights.iter()
                        .zip(x.iter())
                        .map(|(w_i, x_i)| w_i + self.learning_rate * (y - y_pred) * x_i)
                        .collect::<Vec<f64>>();
                    errors += 1;
                }
            }
            if errors == 0 {
                println!("Convergence reached after {} iterations.", iter);
                break;  
            }

        }
        if errors != 0 {
            println!("Maximum number of iterations reached, convergence no reached.")
        }
    }
}