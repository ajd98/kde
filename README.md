# kde

## Overview
This repository provides a Python library for kernel density estimation. In comparison to other Python implementations of kernel density estimation, key features of this library include:

1. Support for weighted samples
2. A strictly positive kernel for estimation of probability density functions with support on subsets of R+
3. Interface for kernel density estimation from WESTPA data sets (https://westpa.github.io/westpa/).

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
class kde.KDE(data, kernel='gaussian', weights=None, bw=1)
```

Parameters:

| Parameter | Data type | Description |
| --------- | --------- | ----------- |
| `data`    | `numpy.ndarray` | The values of the samples in ℝ2 or ℝ (gaussian kernel) or ℝ+ (gamma kernel) |
| `kernel`  | `string` | The Kernel. Options are `"gaussian"` and `"gamma"`. The Gaussian kernel is given by:<br> _p(x) = 1/√(2πσ) exp(-x<sup>2</sup>/(2σ<sup>2</sup>))_<br><br> The gamma kernel is given by:<br> _p(x) = 1/[Γ(k) θ<sup>k</sup>] x<sup>k-1</sup> exp(-x/θ)_<br><br>with _θ_, _k_ chosen such that the mode of _p_ corresponds with the value of each sample, and the square root of the variance of _p_ is equal to `bw` (described below) |
| `weights` | `numpy.ndarray` or `None` | The weights of the samples. If `None`, the samples are uniformly weighted. |
| `bw`      | `float` | The bandwidth of the kernel (σ for gaussian kernel, or square-root of variance of gamma distribution for gamma kernel) |

            
Methods:

| Method | Description |
| ------ | ----------- |
| `set_kernel_type(kernel)` | Set the kernel to `kernel`. Options are `"gaussian"` and `"gamma"`. |
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

Following initialization, call the `go` method as `<WKDE class instance>.go(points)` to evaluate the kernel density estimate at each point in `points`.  A gaussian kernel is set automatically; to use another kernel, use the `set_kernel_type` method (see documentation for `kde.KDE`) followed by the `evaluate` method.

To interact with WESTPA data from the command line, run `python kde/w_kde.py`; include the `-h` or `--help` flag for more information.
