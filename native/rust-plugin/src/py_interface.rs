use anyhow::{Context, Result};
use faiss::gpu::{GpuIndex, StandardGpuResources};
use faiss::index::Index;
use faiss::IndexFlatL2;
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use std::sync::{Arc, Mutex};

/// GPU index wrapper that manages the Faiss GPU index and resources.
pub struct GpuFaissIndex {
    resources: StandardGpuResources,
    index: Option<Arc<Mutex<Box<dyn GpuIndex>>>>,
    dim: usize,
}

impl GpuFaissIndex {
    /// Create a new GPU index with the specified dimension.
    ///
    /// Initializes StandardGpuResources and creates an IndexFlatL2 on the GPU.
    pub fn new(dim: usize) -> Result<Self> {
        let resources = StandardGpuResources::new()
            .context("Failed to initialize StandardGpuResources")?;

        Ok(Self {
            resources,
            index: None,
            dim,
        })
    }

    /// Add vectors to the GPU index.
    ///
    /// Transfers vectors from host memory to device memory and adds them to the index.
    pub fn add_vectors(&mut self, vectors: &[f32]) -> Result<()> {
        let n_vectors = vectors.len() / self.dim;
        if n_vectors * self.dim != vectors.len() {
            anyhow::bail!(
                "Vector length {} is not divisible by dimension {}",
                vectors.len(),
                self.dim
            );
        }

        // Create a CPU index first, then transfer to GPU
        let cpu_index = IndexFlatL2::new(self.dim as i32)
            .context("Failed to create CPU IndexFlatL2")?;

        // Add vectors to CPU index
        cpu_index.add(vectors).context("Failed to add vectors to CPU index")?;

        // Transfer to GPU
        let gpu_index = self
            .resources
            .convert_index(&cpu_index)
            .context("Failed to convert index to GPU")?;

        self.index = Some(Arc::new(Mutex::new(gpu_index)));
        Ok(())
    }

    /// Search for k nearest neighbors.
    ///
    /// Transfers query vectors from host to device, performs search, and returns results.
    pub fn search_knn(&self, queries: &[f32], k: i32) -> Result<(Vec<i64>, Vec<f32>)> {
        let index = self
            .index
            .as_ref()
            .context("Index not initialized. Call add_vectors first.")?;

        let n_queries = queries.len() / self.dim;
        if n_queries * self.dim != queries.len() {
            anyhow::bail!(
                "Query length {} is not divisible by dimension {}",
                queries.len(),
                self.dim
            );
        }

        let guard = index.lock().unwrap();
        let (indices, distances) = guard.search(queries, k).context("Failed to search GPU index")?;

        Ok((indices, distances))
    }
}

/// Python interface for GPU Faiss operations.
#[pyclass]
pub struct PyGpuFaiss {
    inner: Mutex<Option<GpuFaissIndex>>,
}

#[pymethods]
impl PyGpuFaiss {
    /// Create a new GPU Faiss index with the specified dimension.
    #[new]
    fn new(dim: usize) -> PyResult<Self> {
        match GpuFaissIndex::new(dim) {
            Ok(index) => Ok(Self {
                inner: Mutex::new(Some(index)),
            }),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())),
        }
    }

    /// Add vectors to the index.
    ///
    /// Args:
    ///     vectors: 2D numpy array of shape (n_vectors, dim) with f32 values.
    fn add_vectors(&self, vectors: &PyArray2<f32>) -> PyResult<()> {
        let dims = vectors.shape();
        if dims.len() != 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "vectors must be a 2D array",
            ));
        }

        let flat_vec = vectors.readonly().as_slice().unwrap().to_vec();
        let mut index = self.inner.lock().unwrap();
        let gpu_index = index.as_mut().unwrap();

        gpu_index
            .add_vectors(&flat_vec)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Search for k nearest neighbors.
    ///
    /// Args:
    ///     queries: 2D numpy array of shape (n_queries, dim) with f32 values.
    ///     k: Number of nearest neighbors to return.
    ///
    /// Returns:
    ///     Tuple of (indices, distances) where indices is a 2D array of shape (n_queries, k)
    ///     and distances is a 2D array of shape (n_queries, k).
    fn search_knn<'py>(
        &self,
        py: Python<'py>,
        queries: &PyArray2<f32>,
        k: i32,
    ) -> PyResult<(&'py PyArray2<i64>, &'py PyArray2<f32>)> {
        let dims = queries.shape();
        if dims.len() != 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "queries must be a 2D array",
            ));
        }

        let flat_vec = queries.readonly().as_slice().unwrap().to_vec();
        let index = self.inner.lock().unwrap();
        let gpu_index = index.as_ref().unwrap();

        let (indices, distances) = gpu_index
            .search_knn(&flat_vec, k)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let n_queries = dims[0];

        // Create output arrays
        let indices_array = PyArray2::from_owned_array(
            py,
            ndarray::Array2::from_shape_vec((n_queries, k as usize), indices)
                .unwrap(),
        );
        let distances_array = PyArray2::from_owned_array(
            py,
            ndarray::Array2::from_shape_vec((n_queries, k as usize), distances)
                .unwrap(),
        );

        Ok((indices_array, distances_array))
    }
}

/// Create a GPU index (FFI-exposed function).
///
/// This function initializes faiss::gpu::StandardGpuResources and creates
/// an IndexFlatL2 for L2 distance-based nearest neighbor search.
#[pyfunction]
pub fn create_gpu_index(dim: usize) -> PyResult<PyGpuFaiss> {
    PyGpuFaiss::new(dim)
}

/// Add vectors to a GPU index (FFI-exposed function).
///
/// Handles memory transfer between host and device.
#[pyfunction]
pub fn add_vectors(index: &mut PyGpuFaiss, vectors: &PyArray2<f32>) -> PyResult<()> {
    index.add_vectors(vectors)
}

/// Search for k nearest neighbors on GPU (FFI-exposed function).
///
/// Handles memory transfer between host and device.
#[pyfunction]
pub fn search_knn<'py>(
    py: Python<'py>,
    index: &PyGpuFaiss,
    queries: &PyArray2<f32>,
    k: i32,
) -> PyResult<(&'py PyArray2<i64>, &'py PyArray2<f32>)> {
    index.search_knn(py, queries, k)
}
