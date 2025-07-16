
/// Multiply matrix (m x n) by vector (n) => vector (m)
pub(crate) fn matrix_vector_product(matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| row.iter().zip(vector.iter()).map(|(a, b)| a * b).sum())
        .collect()
}

/// Multiply two matrices: A (p x q) * B (q x r) => C (p x r)
pub(crate) fn matrix_multiply(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
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
pub(crate) fn transpose(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
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
pub(crate) fn distance(x1: &[f64], x2: &[f64]) -> f64 {
    x1.iter().zip(x2.iter()).map(|(a, b)| (a - b).powi(2)).sum()
}

/// Invert square matrix using Gaussian elimination
pub(crate) fn invert_matrix(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
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
            aug[i][j] /= pivot + 1e-12;
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

/// Squared Euclidean distance between two vectors
pub(crate) fn distance(x1: &[f64], x2: &[f64]) -> f64 {
    x1.iter().zip(x2.iter()).map(|(a, b)| (a - b).powi(2)).sum()
}

/// Invert square matrix using Gaussian elimination
pub(crate) fn invert_matrix(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
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
            aug[i][j] /= pivot + 1e-12;
        }
        for k in 0..n {
            if k != i {
                let factor = aug[k][i];
                for j in 0..2 * n {
                    aug[k][j] -= factor * aug[i][j];
                }
            }
        }
/// assuming classification task, compute accuracy for one input x and true label y_true
pub(crate) fn accuracy_mlp(
    x: f64,
    y_true: f64
) -> f64 {
    (x - y_true).abs()
}

pub(crate) fn batch_accuracy_mlp(
    x: &[f64],
    y_true: &[f64]
) -> Vec<f64> {
    if x.len() != y_true.len() {
        panic!("Error: x and y_true must have the same length.");
    }

    let accuracies: Vec<f64> = x.iter()
        .zip(y_true.iter())
        .map(|(&x_i, &y_i)| accuracy_mlp(x_i, y_i))
        .collect();
    accuracies
    aug.into_iter().map(|row| row[n..].to_vec()).collect()
}