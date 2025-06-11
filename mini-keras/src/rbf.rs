use rand::prelude::SliceRandom;
use std::error::Error;

/// Multiply matrix (m x n) by vector (n) => vector (m)
fn matrix_vector_product(matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| row.iter().zip(vector.iter()).map(|(a, b)| a * b).sum())
        .collect()
}

/// Multiply two matrices: A (p x q) * B (q x r) => C (p x r)
fn matrix_multiply(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let p = a.len();
    let q = if p > 0 { a[0].len() } else { 0 };
    let r = if !b.is_empty() { b[0].len() } else { 0 };
    let mut c = vec![vec![0.0; r]; p];
    for i in 0..p {
        for k in 0..q {
            for j in 0..r {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

/// Transpose a matrix
fn transpose(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if matrix.is_empty() { return vec![]; }
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut t = vec![vec![0.0; rows]; cols];
    for i in 0..rows {
        for j in 0..cols {
            t[j][i] = matrix[i][j];
        }
    }
    t
}

/// Squared Euclidean distance between two vectors
fn distance(x1: &[f64], x2: &[f64]) -> f64 {
    x1.iter().zip(x2.iter()).map(|(a, b)| (a - b).powi(2)).sum()
}

/// Invert square matrix using Gaussian elimination
fn invert_matrix(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = matrix.len();
    let mut aug: Vec<Vec<f64>> = matrix
        .iter()
        .zip((0..n).map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect::<Vec<_>>()))
        .map(|(row, id)| {
            let mut r = row.clone();
            r.extend(id);
            r
        })
        .collect();

    for i in 0..n {
        // pivot
        let pivot = aug[i][i];
        // if pivot.abs() < 1e-12 {
        //     // return Err("Singular matrix".into());
        // }
        for j in 0..2 * n {
            aug[i][j] /= (pivot + 1e-12);
        }
        for k in 0..n {
            if k != i {
                let factor = aug[k][i];
                for j in 0..2 * n {
                    aug[k][j] -= factor * aug[i][j];
                }
            }
        }
    }
    aug.into_iter().map(|row| row[n..].to_vec()).collect()
}

/// Sigmoid activation
fn sigmoid(x: f64) -> f64 {
    // use sign function
    if x>= 0.0 {
        return 1.0
    }
    0.0
}

/// K-means clustering
pub struct KMeans {
    pub k: usize,
    pub max_iters: usize,
}

impl KMeans {
    pub fn new(k: usize, max_iters: usize) -> Self {
        Self { k, max_iters }
    }

    pub fn fit(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let mut centroids = self.initialize_centroids(x);
        let mut labels = vec![0; x.len()];

        for _ in 0..self.max_iters {
            // assign
            for (i, xi) in x.iter().enumerate() {
                labels[i] = centroids
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        let da = distance(xi, a);
                        let db = distance(xi, b);
                        da.partial_cmp(&db).unwrap()
                    })
                    .map(|(idx, _)| idx)
                    .unwrap();
            }
            // update
            let mut counts = vec![0; self.k];
            let dim = x[0].len();
            let mut sums = vec![vec![0.0; dim]; self.k];
            for (xi, &lbl) in x.iter().zip(labels.iter()) {
                counts[lbl] += 1;
                for d in 0..dim {
                    sums[lbl][d] += xi[d];
                }
            }
            let mut converged = true;
            for i in 0..self.k {
                if counts[i] > 0 {
                    for d in 0..dim {
                        let new_c = sums[i][d] / counts[i] as f64;
                        if (new_c - centroids[i][d]).abs() > 1e-6 {
                            converged = false;
                        }
                        centroids[i][d] = new_c;
                    }
                }
            }
            if converged {
                break;
            }
        }
        centroids
    }

    fn initialize_centroids(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let mut rng = rand::thread_rng();
        let mut centroids = x.to_vec();
        centroids.shuffle(&mut rng);
        centroids.truncate(self.k);
        centroids
    }
}

/// RBF using k-means centroids as centers
pub struct RBFKMeans {

    centers: Vec<Vec<f64>>,
    weights: Vec<f64>,
    gamma: f64,
    is_class: bool,
}

impl RBFKMeans {
    pub fn new(
        x: &[Vec<f64>],
        y: &[f64],
        k: usize,
        gamma: f64,
        max_iters: usize,
        is_class: bool,
    ) -> Self {
        // 1. fit k-means
        let centroids = KMeans::new(k, max_iters).fit(x);
        // 2. build design matrix Phi (m x k)
        let m = x.len();
        let mut phi = vec![vec![0.0; k]; m];
        for i in 0..m {
            for j in 0..k {
                phi[i][j] = (-gamma * distance(&x[i], &centroids[j])).exp();
            }
        }
        // 3. compute W = (Phi^T Phi)^-1 Phi^T Y
        let phi_t = transpose(&phi);            // k x m
        let phi_t_phi = matrix_multiply(&phi_t, &phi); // k x k
        let inv = invert_matrix(&phi_t_phi);   // k x k
        let phi_t_y = matrix_vector_product(&phi_t, y); // k
        let w = matrix_vector_product(&inv, &phi_t_y);

        Self {
            centers: centroids,
            weights: w,
            gamma: gamma,
            is_class: is_class,
        }
    }

    pub fn predict(&self, x_new: &[f64]) -> f64 {
        let k = self.centers.len();
        let mut out = 0.0;
        for j in 0..k {
            let basis = (-self.gamma * distance(x_new, &self.centers[j])).exp();
            out += self.weights[j] * basis;
        }
        println!("task: {}", self.is_class);
        if self.is_class {
            sigmoid(out)
        } else {
            out
        }
    }
}

// Example usage:
// fn main() -> Result<(), Box<dyn Error>> {
//     let X = vec![vec![1.0,4.0], vec![0.5,0.5], vec![2.0, 3.0]];
//     let Y = vec![1.0,0.0,1.0];
//     let model = RBFKMeans::new(
//         &X,
//         &Y,
//         3, // number of clusters
//         0.5, // gamma
//         100, // max iterations
//         true, // is classification
//     );
//     let pred = model.predict(&[0.0,0.0]);
//     println!("Prediction rbf: {}", pred);
//     Ok(())
// }
