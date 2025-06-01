#![allow(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyArray2};

mod linear_model_gradient;
mod mlp;
mod utils;
mod linear_model;

use crate::linear_model_gradient::LinearModelGradientDescent as RustLinearModelGradient;
use crate::linear_model::LinearRegression as RustLinearRegression;
use crate::linear_model::LinearClassification as RustLinearClassification;


// use crate::mlp::Prediction as RustPrediction;
// use crate::mlp::Prediction;
use crate::mlp::Perceptron as RustPerceptron;
use crate::mlp::DenseLayer as RustDenseLayer;
use crate::mlp::MLP as RustMLP;

#[pyclass(name = "LinearRegression")]
struct PyLinearRegression {
    model: RustLinearRegression,
}

#[pyclass(name = "LinearClassification")]
struct PyLinearClassification {
    model: RustLinearClassification,
}

#[pyclass(name = "LinearModelGradient")]
struct PyLinearModelGradient {
    model: RustLinearModelGradient,
}

#[pyclass(name = "MLP")]
struct PyMLP {
    model: RustMLP,
}

#[pymethods]
impl PyLinearRegression {
    #[new]
    fn new() -> PyResult<Self> {
        let rust_model = RustLinearRegression::new();
        Ok(PyLinearRegression { model: rust_model })
    }

    fn fit<'py>(&mut self, py: Python<'py>, x_train: &'py PyArray2<f64>, y_train: &'py PyArray1<f64>) -> PyResult<()> {
        let x = x_train.readonly().as_array().to_owned();
        let y = y_train.readonly().as_array().to_owned();
        self.model.fit(&x, &y);
        Ok(())
    }

    fn predict<'py>(&self, py: Python<'py>, x: &'py PyArray2<f64>) -> PyResult<&'py PyArray1<f64>> {
        let x = x.readonly().as_array().to_owned();
        let result = self.model.predict(&x);
        Ok(PyArray1::from_owned_array(py, result))
    }

}

#[pymethods]
impl PyLinearClassification {
    #[new]
    fn new(learning_rate: Option<f64>, max_iterations: Option<usize>, verbose: Option<bool>) -> PyResult<Self> {
        let rust_model = RustLinearClassification::new(learning_rate, max_iterations, verbose);
        Ok(PyLinearClassification { model: rust_model })
    }

    fn fit(&mut self, x_train: Vec<Vec<f64>>, y_train: Vec<f64>) -> PyResult<()> {
        self.model.fit(&x_train, &y_train);
        Ok(())
    }

    fn predict(&self, x: Vec<f64>) -> PyResult<f64> {
        Ok(self.model.predict(&x))
    }
    
    fn get_weights(&self) -> PyResult<Vec<f64>> {
        Ok(self.model.get_weights())
    }
    
    fn get_bias(&self) -> PyResult<f64> {
        Ok(self.model.get_bias())
    }   
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
impl PyLinearModelGradient {
    #[new]
    fn new(learning_rate: f64, epochs: usize, mode: String, verbose: bool) -> PyResult<Self> {
        let rust_model =
            RustLinearModelGradient::new(learning_rate, epochs, mode, verbose)
                .map_err(|e_str| PyValueError::new_err(e_str))?;

        Ok(PyLinearModelGradient { model: rust_model })
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
    m.add_class::<PyLinearModelGradient>()?;
    m.add_class::<PyMLP>()?;
    m.add_class::<PyLinearRegression>()?;
    m.add_class::<PyLinearClassification>()?;
    Ok(())
}