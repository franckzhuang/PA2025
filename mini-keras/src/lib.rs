#![allow(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::exceptions::PyValueError;

mod linear_model;
mod utils;

use crate::linear_model::LinearModel as RustLinearModel;

#[pyclass(name = "LinearModel")]
struct PyLinearModel {
    model: RustLinearModel,
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
    Ok(())
}