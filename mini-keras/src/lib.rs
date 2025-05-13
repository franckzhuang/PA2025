#![allow(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::exceptions::PyValueError;

mod linear_model;
mod mlp;
mod utils;

use crate::linear_model::LinearModel as RustLinearModel;
// use crate::mlp::Prediction as RustPrediction;
// use crate::mlp::Prediction;
use crate::mlp::Perceptron as RustPerceptron;
use crate::mlp::DenseLayer as RustDenseLayer;
use crate::mlp::MLP as RustMLP;

#[pyclass(name = "LinearModel")]
struct PyLinearModel {
    model: RustLinearModel,
}

#[pyclass(name = "MLP")]
struct PyMLP {
    model: RustMLP,
}

#[pymethods]
impl PyMLP {

    #[new]
    fn new(layers: Vec<usize>, is_classification:bool) ->PyResult<Self> {
        let mut dense_layers: Vec<RustDenseLayer> = Vec::new();
        for i in 0..layers.len() - 1 {
            let layer = RustDenseLayer::new(layers[i], layers[i + 1]);
            dense_layers.push(layer);
        }

        let rust_model = RustMLP::new(dense_layers, is_classification);

        Ok(PyMLP { model: rust_model })

    }

    fn predict(&self, x: Vec<f64>) -> PyResult<Vec<f64>> {
        Ok(self.model.predict(&x))
    }

    fn fit(&mut self, x_train: Vec<Vec<f64>>, y_train: Vec<f64>, epochs: usize, lr : f64) -> PyResult<()> {
        self.model.train(&x_train, &y_train, epochs, lr);
        Ok(())
    }
}

#[pymethods]
impl PyLinearModel {
    #[new]
    fn new(learning_rate: f64, epochs: usize, mode: String, verbose: bool) -> PyResult<Self> {
        let rust_model =
            RustLinearModel::new(learning_rate, epochs, mode, verbose)
                .map_err(|e_str| PyValueError::new_err(e_str))?;

        Ok(PyLinearModel { model: rust_model })
    }

    fn fit(&mut self, x_train: Vec<Vec<f64>>, y_train: Vec<f64>) -> PyResult<()> {
        self.model.fit(&x_train, &y_train);
        Ok(())
    }

    fn predict(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        Ok(self.model.predict(&x))
    }
}

#[pymodule]
fn mini_keras(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyLinearModel>()?;
    m.add_class::<PyMLP>()?;
    Ok(())
}