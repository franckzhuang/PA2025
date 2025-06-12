use std::error::Error;

/// square matrix X column vector.
fn matrix_vector_product(matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| row.iter().zip(vector.iter()).map(|(a, b)| a * b).sum())
        .collect()
}

/// euclidian distance ||X1 - X2||^2
fn distance(x1: &[f64], x2: &[f64]) -> f64 {
    x1.iter()
        .zip(x2.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum()
}

/// calculate the inverse of a square matrix
fn invert_matrix(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = matrix.len();
    // Build augmented matrix [matrix | I]
    let mut aug: Vec<Vec<f64>> = matrix
        .iter()
        .zip((0..n).map(|i| {
            (0..n)
                .map(|j| if i == j { 1.0 } else { 0.0 })
                .collect::<Vec<f64>>()
        }))
        .map(|(row, id_row)| {
            let mut r = row.clone();
            r.extend(id_row);
            r
        })
        .collect();

    // Forward elimination
    for i in 0..n {
        // Find pivot
        let pivot = aug[i][i];
        if pivot.abs() < 1e-12 {
            // return Err("Matrix is singular and cannot be inverted.".into());
        }
        // Normalize pivot row
        for j in 0..2 * n {
            aug[i][j] /= (pivot+ 1e-12);
        }
        // Eliminate other rows
        for k in 0..n {
            if k != i {
                let factor = aug[k][i];
                for j in 0..2 * n {
                    aug[k][j] -= factor * aug[i][j];
                }
            }
        }
    }

    // Extract right half as inverse
    let inv: Vec<Vec<f64>> = aug.into_iter().map(|row| row[n..].to_vec()).collect();
    inv
}

/// Sigmoid for classification.
fn sigmoid(x: f64) -> f64 {
    // use sign function
    if x>= 0.0 {
        return 1.0
    }
    0.0
}

pub struct RBFNaive {
    x: Vec<Vec<f64>>,   // training inputs
    y: Vec<f64>,        // training targets
    gamma: f64,
    is_classification: bool,
    phi: Vec<Vec<f64>>, // kernel matrix
    w: Vec<f64>,        // weights
}

impl RBFNaive {
    pub fn new(
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        gamma: f64,
        is_classification: bool,
    ) -> RBFNaive {
        let mut model = RBFNaive {
            x,
            y,
            gamma,
            is_classification,
            phi: Vec::new(),
            w: Vec::new(),
        };
        model.compute_phi();
        model.compute_w();
        model
    }

    /// build kernel matrix.
    fn compute_phi(&mut self) -> Vec<Vec<f64>> {
        let n = self.x.len();
        let mut phi = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let d2 = distance(&self.x[i], &self.x[j]);
                phi[i][j] = (-self.gamma * d2).exp();
            }
        }
        self.phi = phi;
        self.phi.clone()
    }

    /// compute W: inv(phi) * Y
    fn compute_w(&mut self) -> Vec<f64> {
        let inv_phi = invert_matrix(&self.phi);
        self.w = matrix_vector_product(&inv_phi, &self.y);
        self.w.clone()
    }


    pub fn predict(&self, x_new: &[f64]) -> f64 {
        let k: Vec<f64> = self
            .x
            .iter()
            .map(|xi| (-self.gamma * distance(x_new, xi)).exp())
            .collect();
        let out: f64 = self.w.iter().zip(k.iter()).map(|(w, ki)| w * ki).sum();
        if self.is_classification {
            sigmoid(out)
        } else {
            out
        }
    }
}

// fn main() -> Result<(), Box<dyn Error>> {
//     let X = vec![vec![3.0, 4.0], vec![0.5, 0.5], vec![3.0, 3.0]];
//     let Y = vec![1.0, 0.0, 1.0];
//     let rbf = RBFNaive::new(X.clone(), Y.clone(), 0.001, false)?;
//
//     println!("Weights: {:?}", rbf.w);
//     println!("Design matrix phi:");
//     for row in &rbf.phi {
//         println!("{:?}", row);
//     }
//     let pred = rbf.predict(&[0.0, 0.0]);
//     println!("Prediction at [0,0]: {}", pred);
//
// }
