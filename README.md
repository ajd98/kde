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
| `data`    | `numpy.ndarray` | The values of the samples in R2 or R (gaussian kernel) or R+ (gamma kernel) |
| `kernel`  | `string` | The Kernel. Options are `"gaussian"` and `"gamma"` |
| `weights` | `numpy.ndarray` or `None` | The weights of the samples. If `None`, the samples are uniformly weighted. |
| `bw`      | `float` | The bandwidth of the kernel (σ for gaussian kernel, or square-root of variance of gamma distribution for gamma kernel) |
            
Methods:
| Method | Description |
| ------ | ----------- |
| set_kernel_type(kernel) | Set the kernel to kernel. Options are "gaussian" and "gamma". |
| evaluate(p) | Evaluate the kernel density estimate at each position of p, an n-by-k numpy array, where k is the number of features of the samples. |
```

