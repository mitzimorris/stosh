"""
Stosh: Object-oriented Python interface to Stan

Provides compile() function that returns CompiledModel objects for Stan sampling.
"""

import ctypes
import os
import subprocess
import shutil
from typing import Optional, Dict, Any, Union
from pathlib import Path


class StoshError(Exception):
    """Exception for Stosh errors"""
    pass


class CompiledModel:
    """
    A compiled Stan model with sampling capabilities.
    
    This object wraps a compiled Stan model (.so file) and provides
    methods to load data and run HMC-NUTS sampling.
    
    Use the compile() function to create instances of this class.
    """
    
    def __init__(self, so_path: str):
        """Initialize compiled model (internal use only)"""
        self._so_path = so_path
        self._lib = None
        self._model_handle = None
        self._load_library()
        
    def _load_library(self):
        """Load the shared library and set up function signatures"""
        try:
            self._lib = ctypes.CDLL(self._so_path)
            self._setup_function_signatures()
        except OSError as e:
            raise StoshError(f"Failed to load compiled model: {e}")
    
    def _setup_function_signatures(self):
        """Set up ctypes function signatures for the C API"""
        # stan::run::load_model
        self._lib.stosh_load_model.argtypes = [
            ctypes.c_char_p, ctypes.c_uint, ctypes.c_char_p, ctypes.c_size_t
        ]
        self._lib.stosh_load_model.restype = ctypes.c_void_p
        
        # stan::run::hmc_nuts  
        self._lib.stosh_run_samplers.argtypes = [
            ctypes.c_void_p, 
            ctypes.POINTER(ctypes.c_char_p), 
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_int, 
            ctypes.c_char_p, ctypes.c_size_t,
            ctypes.c_char_p, ctypes.c_size_t
        ]
        self._lib.stosh_run_samplers.restype = ctypes.c_int
        
        # Model cleanup
        self._lib.stosh_free_model.argtypes = [ctypes.c_void_p]
        self._lib.stosh_free_model.restype = None
        
        # Model name (optional)
        if hasattr(self._lib, 'stosh_get_model_name'):
            self._lib.stosh_get_model_name.argtypes = [ctypes.c_void_p]
            self._lib.stosh_get_model_name.restype = ctypes.c_char_p
    
    def load_data(self, data: Union[str, Dict[str, Any], None] = None, seed: int = 1):
        """
        Load data into the Stan model.
        
        Args:
            data: Path to JSON data file, dictionary of data, or None
            seed: Random seed for model initialization
            
        Raises:
            StoshError: If data loading fails
        """
        if self._model_handle is not None:
            # Free existing model before loading new data
            self._lib.stosh_free_model(self._model_handle)
            self._model_handle = None
        
        # Convert data to string path if needed
        if isinstance(data, dict):
            # TODO: Could write dict to temporary JSON file here
            raise StoshError("Dictionary data not yet supported - please provide JSON file path")
        
        data_str = str(data) if data is not None else ""
        
        # Prepare error buffer
        error_msg = ctypes.create_string_buffer(1024)
        
        # Call C API to load model with data
        self._model_handle = self._lib.stosh_load_model(
            data_str.encode('utf-8'),
            ctypes.c_uint(seed),
            error_msg,
            ctypes.sizeof(error_msg)
        )
        
        if not self._model_handle:
            raise StoshError(f"Failed to load data: {error_msg.value.decode('utf-8')}")
    
    def hmc_nuts(self, **kwargs) -> Dict[str, str]:
        """
        Run HMC-NUTS sampling on the loaded model.
        
        Args:
            **kwargs: Sampling parameters (num_chains, warmup, samples, etc.)
            
        Returns:
            Dict containing output directory path
            
        Raises:
            StoshError: If sampling fails or no data has been loaded
        """
        if self._model_handle is None:
            raise StoshError("No data loaded. Call load_data() first.")
        
        # Convert kwargs to C API format
        keys, values = self._kwargs_to_c_arrays(kwargs)
        
        # Prepare output buffers
        output_dir = ctypes.create_string_buffer(1024)
        error_msg = ctypes.create_string_buffer(1024)
        
        # Call C API for sampling
        result = self._lib.stosh_run_samplers(
            self._model_handle,
            keys,
            values, 
            len(kwargs),
            output_dir,
            ctypes.sizeof(output_dir),
            error_msg,
            ctypes.sizeof(error_msg)
        )
        
        if result != 0:
            raise StoshError(f"Sampling failed: {error_msg.value.decode('utf-8')}")
        
        return {'output_dir': output_dir.value.decode('utf-8')}
    
    def _kwargs_to_c_arrays(self, kwargs: Dict[str, Any]):
        """Convert Python kwargs to C-compatible key-value arrays"""
        if not kwargs:
            return None, None
        
        # Convert values to strings
        string_kwargs = {key: self._value_to_string(value) for key, value in kwargs.items()}
        
        # Create C arrays
        keys = (ctypes.c_char_p * len(string_kwargs))()
        values = (ctypes.c_char_p * len(string_kwargs))()
        
        for i, (key, value) in enumerate(string_kwargs.items()):
            keys[i] = key.encode('utf-8')
            values[i] = value.encode('utf-8')
        
        return keys, values
    
    def _value_to_string(self, value: Any) -> str:
        """Convert Python values to strings for C API"""
        if isinstance(value, bool):
            return "true" if value else "false"
        else:
            return str(value)
    
    @property
    def name(self) -> Optional[str]:
        """Get the model name if available"""
        if self._model_handle and hasattr(self._lib, 'stosh_get_model_name'):
            name_ptr = self._lib.stosh_get_model_name(self._model_handle)
            if name_ptr:
                return ctypes.string_at(name_ptr).decode('utf-8')
        return None
    
    def __del__(self):
        """Clean up model resources when object is destroyed"""
        if hasattr(self, '_model_handle') and self._model_handle and hasattr(self, '_lib'):
            self._lib.stosh_free_model(self._model_handle)
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        if self._model_handle:
            self._lib.stosh_free_model(self._model_handle)
            self._model_handle = None


def compile(stan_file: str, force: bool = False) -> CompiledModel:
    """
    Compile a Stan model file to a shared library and return a CompiledModel object.
    
    Args:
        stan_file: Path to .stan file (e.g., "examples/bernoulli.stan")
        force: If True, force recompilation even if .so file exists and is newer
        
    Returns:
        CompiledModel object wrapping the compiled shared library
        
    Raises:
        StoshError: If compilation fails
    """
    stan_path = Path(stan_file).resolve()
    
    if not stan_path.exists():
        raise StoshError(f"Stan file does not exist: {stan_file}")
    
    if not stan_path.suffix == '.stan':
        raise StoshError(f"File must have .stan extension: {stan_file}")
    
    # Find the directory containing the makefile
    # Priority: STAN_ROOT env var, then parent of this Python package
    if 'STAN_ROOT' in os.environ:
        makefile_dir = Path(os.environ['STAN_ROOT']).resolve()
    else:
        python_dir = Path(__file__).parent.resolve()
        makefile_dir = python_dir.parent
    
    # Check that makefile exists
    makefile_path = makefile_dir / "makefile"
    if not makefile_path.exists():
        raise StoshError(f"Makefile not found at: {makefile_path}")
    
    # Expected output paths
    model_name = stan_path.stem
    so_path = stan_path.parent / f"{model_name}_model.so"
    
    # Check if we need to compile
    if not force and so_path.exists():
        if so_path.stat().st_mtime > stan_path.stat().st_mtime:
            print(f"Using existing compiled model: {so_path}")
            return CompiledModel(str(so_path))
    
    print(f"Compiling Stan model: {stan_file}")
    print(f"Using makefile directory: {makefile_dir}")
    
    # Calculate the target name for make - this should be the path relative to makefile_dir
    try:
        # Get the relative path from makefile directory to the .stan file
        relative_stan_path = stan_path.relative_to(makefile_dir)
        # Create the target: path/to/model_model.so
        target_path = relative_stan_path.parent / f"{model_name}_model.so"
        make_target = str(target_path)
        
    except ValueError:
        # .stan file is not under the makefile directory
        # Try to use the relative path from current working directory
        try:
            cwd = Path.cwd()
            relative_to_cwd = stan_path.relative_to(cwd)
            target_path = relative_to_cwd.parent / f"{model_name}_model.so"
            make_target = str(target_path)
            print(f"Warning: .stan file is outside makefile directory. Using target: {make_target}")
        except ValueError:
            # Fall back to just the model name
            make_target = f"{model_name}_model.so"
            print(f"Warning: Using fallback target: {make_target}")
    
    # Run make command from the makefile directory
    try:
        cmd = ["make", make_target]
        
        print(f"Running: {' '.join(cmd)} (from {makefile_dir})")
        
        result = subprocess.run(
            cmd,
            cwd=makefile_dir,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            error_msg = f"Compilation failed:\nCommand: {' '.join(cmd)}\nWorking directory: {makefile_dir}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            raise StoshError(error_msg)
        
        if not so_path.exists():
            raise StoshError(f"Compilation succeeded but output file not found: {so_path}")
        
        print(f"Compilation successful: {so_path}")
        return CompiledModel(str(so_path))
        
    except FileNotFoundError:
        raise StoshError("Make command not found. Ensure GNU make is installed and in PATH.")
    except Exception as e:
        raise StoshError(f"Compilation error: {e}")
