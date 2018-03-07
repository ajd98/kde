# kde

## Overview
This repository provides a Python library for kernel density estimation. In comparison to other Python implementations of kernel density estimation, key features of this library include:

1. Support for weighted samples
2. A variety of kernels, including a smooth, compact kernel.
3. Interface for kernel density estimation from WESTPA data sets (https://westpa.github.io/westpa/).

## Basics
Kernel density estimation is a technique for estimation of a probability density function based on empirical data. Suppose we have some observations _xᵢ ∈ V_ where _i = 1, ..., n_ and _V_ is a vector space. Given a norm _q: V → ℝ⁺∪{0}_, a kernel function _K: ℝ → ℝ⁺∪{0}_ with _∫ᵥK(q(x))dx = 1_, and a bandwidth _h ∈ ℝ⁺_, the kernel density estimate _p: V → ℝ⁺∪{0}_ is defined as:

_p(x) := 1/(h*n) ΣᵢK(q(x-xᵢ)/h)_

Similarly, a weighted version of the kernel density estimate may be defined as:

_p(x) := 1/h ΣᵢwᵢK(q(x-xᵢ)/h)_

where _wᵢ_ is the weight of the i<sup>th</sup> sample, and _Σᵢwᵢ=1_.

This package includes the following kernel functions:

| kernel    | equation  | 
| --------- | --------- | 
| `bump`    | _p(x) ∝ exp(1/(x²-1))_ |
| `cosine`  | _p(x) ∝ cos(πx/2)_ |
| `epanechnikov` | _p(x) ∝ 1-x²_ |
| `gaussian` | _p(x) ∝ exp(-x²/2)_ |
| `logistic` | _p(x) ∝ 1/(exp(-x)+2+exp(x))_ |
| `quartic` | _p(x) ∝ (1-x²)²_ |
| `tophat` | _p(x) ∝ 1_<sub>A</sub>  |
| `triangle` | _p(x) ∝ (1-x)1_<sub>A</sub> |
| `tricube` | _p(x) ∝ (1-x³)³1_<sub>A</sub> |

In the above definitions, _1_<sub>A</sub> is the indicator function and  _A = {x: ‖x‖ < 1}_.

## Use

### Kernel density estimation with arbitrary data

Before using this library, you will need to make sure that it may be imported by Python. To do so, add the top-level directory of this git repository (the directory containing this README file) to your PYTHONPATH environment variable.  If this does not work, you may also add the following commands to the top of your Python script:

```
import sys
sys.path.append("path to this git repository")
```

Then, import the `kde` module via Python.

Kernel density estimation is performed via the `KDE` class, accessible as `kde.KDE`.

```
class kde.KDE(training_points, kernel='gaussian', weights=None, metric='euclidean_distance', bw=1)
```

Parameters:

| Parameter | Data type | Description |
| --------- | --------- | ----------- |
| `training_points` | `numpy.ndarray` | The values of the samples in ℝⁿ or (S¹)ⁿ = S¹×S¹×...×S¹ |
| `kernel`  | `string` | The kernel. Options are:<br>  `"bump"`<br>  `"cosine"`<br>  `"epanechnikov"`<br>  `"gaussian"`<br>  `"logistic"`<br>  `"quartic"`<br>  `"tophat"`<br>  `"triangle"`<br>  `"tricube"`<br>See above for kernel definitions. |
| `weights` | `numpy.ndarray` or `None` | The weights of the samples. If `None`, the samples are uniformly weighted. |
| `metric`  | `string` | The norm from which to induce the metric for distance between points.  Options are 'euclidean_distance' and 'euclidean_distance_ntorus'. 'euclidean_distance_ntorus' assumes the sample space is an n-torus (S¹×S¹×...×S¹) where each dimension runs between -180 and 180, and the distance is the minimum euclidean distance to a periodic image.|
| `bw`      | `float` | The bandwidth of the kernel |


            
Methods:

| Method | Description |
| ------ | ----------- |
| `set_kernel_type(kernel)` | Set the kernel to `kernel`. See above for options. |
| `evaluate(p)` | Evaluate the kernel density estimate at each position of `p`, an _n_-by-_k_ numpy array, where _k_ is the number of features of the samples. |

### Kernel density estimation with WESTPA data

This library provides classes for interacting with WESTPA data sets, enabling kernel density estimation from WESTPA data via Python scripts and via the command line.

From within a Python script, import the `kde` module, which provides the `kde.WKDE` class for interacting with WESTPA data sets.  The `WKDE` class should be initialized as:

```
kde.WKDE(westh5, first_iter=None, last_iter=None, load_func=None, bw=1)
```

| Parameter | Data type | Description |
| --------- | --------- | ----------- |
| `westh5` | `h5py` HDF5 File object | The WESTPA data file (typically named 'west.h5') |
| `first_iter` | `int` or `None` | The first weighted ensemble iteration from which to use data. If `None`, start at iteration 1. |
| `last_iter` | `int` or `None` | The last weighted ensemble iteration from which to use data (inclusive). |
| `load_func` | Python function or `None` | Load data using the specified Python function.  The function will be called as `load_func(iter_group, niter)` where `iter_group` is the HDF5 group corresponding to a weighted ensemble iteration, and `niter` is an integer denoting the index of the weighted ensemble iteration.  The function should return a numpy array of shape (nsegs, ntimepoints, ndim) where nsegs is the number of segments in that iteration, ntimepoints is the number of sub-iteration timepoints, and ndim is the number of dimensions of the coordinate. If `None` (default), load the progress coordinate data. |
| `bw` | `float` | The bandwidth to use for the kernel.  See the `bw` parameter of the `kde.KDE` class for more information. |

Following initialization, call the `evaluate` method as `<WKDE class instance>.evaluate(points)` to evaluate the kernel density estimate at each point in `points`.  A gaussian kernel is set automatically; to use another kernel, use the `set_kernel_type` method (see documentation for `kde.KDE`) followed by the `evaluate` method.

To interact with WESTPA data from the command line, run `python kde/w_kde.py`; include the `-h` or `--help` flag for more information.
