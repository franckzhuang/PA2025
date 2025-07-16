#![allow(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::exceptions::PyValueError;
// use pyo3::ffi::PyModuleDef;
use pyo3::impl_::wrap::SomeWrap;
use serde::{Deserialize, Serialize};

mod mlp;
mod utils;
mod linear_model;
mod svm;

mod rbf_naive;
mod rbf;

mod kmeans;

// use crate::linear_model::LinearRegression as RustLinearRegression;
use crate::linear_model::LinearClassification as RustLinearClassification;



use crate::mlp::Dense as RustDenseLayer;
use crate::mlp::Activation as RustActivation;
use crate::mlp::MLP as RustMLP;
use crate::svm::{SVM as RustSVM, KernelType};
use crate::rbf_naive::RBFNaive as RustRBFNaive;
use crate::rbf::RBFKMeans as RustRBFKmeans;


#[pyclass(name="RBFNaive")]
pub struct PyRBFNaive {
    model: RustRBFNaive,
}

#[pymethods]
impl PyRBFNaive {
    #[new]
    fn new(x: Vec<Vec<f64>>, y: Vec<f64>, gamma: f64, is_classification: bool) -> PyResult<Self> {
        let model = RustRBFNaive::new(x, y, gamma, is_classification);
        Ok(PyRBFNaive { model })
    }

    fn predict(&self, x_new: Vec<f64>) -> PyResult<f64> {
        Ok(self.model.predict(&x_new))
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string_pretty(&self.model)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[staticmethod]
    fn from_json(json_str: String) -> PyResult<Self> {
        let model: RustRBFNaive = serde_json::from_str(&json_str)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyRBFNaive { model })
    }
}


#[pyclass(name="RBFKMeans")]
pub struct PyRBFKMeans {
    model: RustRBFKmeans,
}

#[pymethods]
impl PyRBFKMeans {
    #[new]
    fn new(x: Vec<Vec<f64>>, y: Vec<f64>, k: usize, gamma: f64, max_iters: usize, is_classification: bool) -> PyResult<Self> {
        let model = RustRBFKmeans::new(x, y, k, gamma, max_iters, is_classification);
        Ok(PyRBFKMeans { model })
    }

    fn predict(&self, x_new: Vec<f64>) -> PyResult<f64> {
        Ok(self.model.predict(&x_new))
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string_pretty(&self.model)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[staticmethod]
    fn from_json(json_str: String) -> PyResult<Self> {
        let model: RustRBFKmeans = serde_json::from_str(&json_str)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyRBFKMeans { model })
    }
}


// #[derive(Serialize, Deserialize, Debug, Clone)]
// #[pyclass(name = "LinearRegression")]
// pub struct PyLinearRegression {
//     model: RustLinearRegression,
// }

// #[pymethods]
// impl PyLinearRegression {
//     #[new]
//     fn new() -> PyResult<Self> {
//         Ok(PyLinearRegression { model: RustLinearRegression::new() })
//     }


//     fn fit(&mut self, x_train: Vec<Vec<f64>>, y_train: Vec<f64>) -> PyResult<()> {
//         self.model.fit(&x_train, &y_train);
//         Ok(())
//     }
//     fn predict(&self, x: Vec<f64>) -> PyResult<f64> {
//         Ok(self.model.predict(&x))
//     }

//     // fn predict_many(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
//     //     Ok(x.iter().map(|sample| self.model.predict(sample)).collect())
//     // }

//     fn get_weights(&self) -> PyResult<Vec<f64>> {
//         Ok(self.model.weights.clone().unwrap_or_default())
//     }

//     fn get_bias(&self) -> PyResult<f64> {
//         Ok(self.model.bias.unwrap_or(0.0))
//     }

//     fn save(&self, path: String) -> PyResult<()> {
//         self.model.save_json(&path)
//             .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
//     }

//     #[staticmethod]
//     fn load(path: String) -> PyResult<Self> {
//         let model = RustLinearRegression::load_json(&path)
//             .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
//         Ok(PyLinearRegression { model })
//     }
// }

#[derive(Serialize, Deserialize, Debug, Clone)]
#[pyclass(name = "LinearClassification")]
struct PyLinearClassification {
    model: RustLinearClassification,
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

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string_pretty(&self.model)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[staticmethod]
    fn from_json(json_str: String) -> PyResult<Self> {
        let model: RustLinearClassification = serde_json::from_str(&json_str)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyLinearClassification { model })
    }
}


#[pyclass(name = "MLP")]
struct PyMLP {
    model: RustMLP,
}

#[pymethods]
impl PyMLP {

    #[new]
    fn new(layers: Vec<usize>, is_classification:bool, activations: Vec<String> ) ->PyResult<Self> {
        let mut dense_layers: Vec<RustDenseLayer> = Vec::new();
        for i in 0..layers.len() - 1 {
            let layer = RustDenseLayer::new(
                layers[i],
                layers[i + 1],
                match activations[i].as_str() {
                    "sigmoid" => RustActivation::Sigmoid,
                    "linear" => RustActivation::Linear,
                    _ => return Err(PyValueError::new_err("Invalid activation function")),
                }
            );
            dense_layers.push(layer);
        }

        let rust_model = RustMLP::new(dense_layers, is_classification);

        Ok(PyMLP { model: rust_model })

    }

    fn predict(&mut self, x: Vec<f64>) -> PyResult<Vec<f64>> {
        Ok(self.model.predict(&x))
    }

    fn fit(&mut self, x_train: Vec<Vec<f64>>, y_train: Vec<f64>, x_test:Vec<Vec<f64>>, y_test:Vec<f64>, epochs: usize, lr : f64) -> PyResult<Vec<Vec<f64>>> {
        let train_losses = self.model.train(&x_train, &y_train, &x_test, &y_test, epochs, lr);
        Ok(train_losses)

    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string_pretty(&self.model)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[staticmethod]
    fn from_json(json_str: String) -> PyResult<Self> {
        let model: RustMLP = serde_json::from_str(&json_str)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyMLP { model })
    }
}

#[pyclass(name = "SVM")]
struct PySVM {
    model: RustSVM,
}

#[pymethods]
impl PySVM {
    #[new]
    fn new(kernel: String, c: Option<f64>, gamma: Option<f64>) -> PyResult<Self> {
        let kernel_type = match kernel.as_str() {
            "linear" => KernelType::Linear,
            "rbf" => {
                let g = gamma.ok_or_else(|| PyValueError::new_err("Gamma required for RBF"))?;
                KernelType::RBF(g)
            }
            _ => return Err(PyValueError::new_err("Kernel must be 'linear' or 'rbf'")),
        };
        Ok(PySVM {
            model: RustSVM::new(c, kernel_type),
        })
    }

    fn fit(&mut self, x_train: Vec<Vec<f64>>, y_train: Vec<f64>) -> PyResult<()> {
        self.model.fit(&x_train, &y_train);
        Ok(())
    }

    fn predict(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<i8>> {
        Ok(self.model.predict(&x))
    }

    #[getter]
    fn b(&self) -> f64 {
        self.model.b
    }

    #[getter]
    fn alphas(&self) -> Vec<f64> {
        self.model.alphas.clone()
    }

    #[getter]
    fn support_vectors(&self) -> Vec<Vec<f64>> {
        self.model.support_vectors.clone()
    }

    #[getter]
    fn support_labels(&self) -> Vec<f64> {
        self.model.support_labels.clone()
    }

    #[getter]
    fn weights(&self) -> Option<Vec<f64>> {
        self.model.weights()
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string_pretty(&self.model)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[staticmethod]
    fn from_json(json_str: String) -> PyResult<Self> {
        let model: RustSVM = serde_json::from_str(&json_str)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PySVM { model })
    }

}

#[pymodule]
fn mini_keras(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyMLP>()?;
    // m.add_class::<PyLinearRegression>()?;
    m.add_class::<PyLinearClassification>()?;
    m.add_class::<PySVM>()?;
    m.add_class::<PyRBFNaive>()?;
    m.add_class::<PyRBFKMeans>()?;
    Ok(())
}