# Stosh: Object-Oriented Python Interface to Stan

A simple, object-oriented Python interface to Stan's sampling capabilities using the `stan::run` C++ API.

## Overview

Stosh provides a clean object-oriented interface for compiling and running Stan models. It connects directly to Stan's C++ sampling engine through a shared library interface.

## Installation

From the project directory:

```bash
pip install -e .
```

## Prerequisites

1. **Stan development environment**: You need Stan's makefiles and build system set up
2. **GNU make**: Required for compilation
3. **C++ compiler**: Required for building Stan models

## Basic Usage

### Object-Oriented Interface

```python
import stosh

# Compile a Stan model
model = stosh.compile("mymodel.stan")

# Load data and run sampling
model.load_data("mymodel.data.json", seed=12345)
results = model.hmc_nuts(
    num_chains=4,
    warmup=1000,
    samples=2000
)

print(f"Results saved to: {results['output_dir']}")
```

### Using Context Manager

```python
import stosh

# Automatic cleanup when done
with stosh.compile("mymodel.stan") as model:
    model.load_data("mymodel.data.json")
    results = model.hmc_nuts(num_chains=2, samples=1000)
    print(f"Output: {results['output_dir']}")
# Model automatically cleaned up here
```

## API Reference

### `stosh.compile(stan_file, force=False)`

Compile a Stan program file to a shared library.

**Parameters:**
- `stan_file` (str): Path to Stan program file (e.g., "mymodel.stan")
- `force` (bool): Force recompilation even if .so file exists and is newer

**Returns:**
- `CompiledModel`: Object for loading data and sampling

**Example:**
```python
model = stosh.compile("bernoulli.stan")
model = stosh.compile("mymodel.stan", force=True)  # Force recompilation
```

### `CompiledModel.load_data(data=None, seed=1)`

Load data into the compiled Stan model.

**Parameters:**
- `data` (str or None): Path to JSON data file, or None for models without data
- `seed` (int): Random seed for initialization

**Example:**
```python
model.load_data("bernoulli.data.json", seed=42)
model.load_data(seed=123)  # No data file for simple models
```

### `CompiledModel.hmc_nuts(**kwargs)`

Run HMC-NUTS sampling on the model.

**Parameters:**
- `**kwargs`: Sampling parameters

**Common sampling parameters:**
- `num_chains` (int): Number of chains (default: 1)
- `warmup` (int): Warmup iterations (default: 1000)  
- `samples` (int): Sampling iterations (default: 1000)
- `thin` (int): Thinning interval (default: 1)
- `stepsize` (float): Step size (default: 1.0)
- `max_depth` (int): Max tree depth (default: 10)
- `metric_type` (str): "unit_e", "diag_e", "dense_e" (default: "diag_e")
- `delta` (float): Target acceptance rate (default: 0.8)
- `refresh` (int): Progress update frequency (default: 100)

**Returns:**
- `dict`: Contains `'output_dir'` with path to results

**Example:**
```python
results = model.hmc_nuts(
    num_chains=4,
    warmup=1000,
    samples=2000,
    stepsize=0.1,
    delta=0.95
)
```

### Properties

- `CompiledModel.name`: Model name (if available)

## Error Handling

Stosh raises `StoshError` for all compilation and sampling errors:

```python
try:
    model = stosh.compile("mymodel.stan")
    model.load_data("data.json")
    results = model.hmc_nuts(num_chains=4)
except stosh.StoshError as e:
    print(f"Error: {e}")
```

## Example Workflow

1. **Write Stan model** (`mymodel.stan`)
2. **Compile and run**:

```python
import stosh

# Compile the model
model = stosh.compile("mymodel.stan")

# Load data (optional)
model.load_data("mymodel.data.json", seed=42)

# Run sampling
results = model.hmc_nuts(
    num_chains=4,
    warmup=1000,
    samples=2000,
    delta=0.95
)

print(f"Output directory: {results['output_dir']}")
```

## Building Stan Models

Stosh uses the existing Stan makefiles to compile models:

- Compilation chain: `mymodel.stan` → `mymodel.hpp` → `mymodel_model.so`
- Uses GNU make with shared library flags
- Automatically finds and loads the compiled `.so` file

## Dependencies

- Python 3.8+
- Stan development environment with makefiles
- GNU make
- No external Python packages required
