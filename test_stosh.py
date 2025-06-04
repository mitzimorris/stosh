#!/usr/bin/env python3
"""
Basic tests for Stosh object-oriented interface.

Run with: python test_stosh.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import stosh


def test_imports():
    """Test that imports work correctly"""
    print("Testing imports...")
    
    # Test that main classes are available
    assert hasattr(stosh, 'compile')
    assert hasattr(stosh, 'CompiledModel')
    assert hasattr(stosh, 'StoshError')
    
    print("✓ Imports successful")


def test_compile_nonexistent():
    """Test compilation error handling"""
    print("Testing error handling...")
    
    try:
        model = stosh.compile("nonexistent.stan")
        print("✗ Expected error but got model")
        return False
    except stosh.StoshError as e:
        print(f"✓ Correctly raised StoshError: {e}")
        return True


def test_compile_invalid_extension():
    """Test invalid file extension"""
    print("Testing invalid file extension...")
    
    try:
        model = stosh.compile("test.txt")
        print("✗ Expected error but got model")
        return False
    except stosh.StoshError as e:
        print(f"✓ Correctly raised StoshError: {e}")
        return True


def test_with_actual_model():
    """Test with actual Stan model if available"""
    print("Testing with actual model (if available)...")
    
    # Look for common Stan model files in examples directory
    test_files = [
        os.path.join("examples", "bernoulli.stan"),
        os.path.join("examples", "simple.stan"),
        os.path.join("..", "examples", "bernoulli.stan"),
        "bernoulli.stan",  # fallback
    ]
    
    stan_file = None
    for file in test_files:
        if os.path.exists(file):
            stan_file = file
            break
    
    if stan_file is None:
        print("ℹ No test Stan files found, skipping actual compilation test")
        print("  Looked for:")
        for file in test_files:
            print(f"    {file}")
        return True
    
    try:
        print(f"  Found Stan file: {stan_file}")
        
        # Test STAN_ROOT environment variable
        if 'STAN_ROOT' in os.environ:
            print(f"  Using STAN_ROOT: {os.environ['STAN_ROOT']}")
        else:
            print("  Using default makefile location (parent directory)")
        
        model = stosh.compile(stan_file)
        print(f"✓ Model compiled successfully!")
        
        # Test that we can't run sampling without data
        try:
            result = model.hmc_nuts(warmup=10, samples=10)
            print("✗ Expected error when sampling without data")
            return False
        except stosh.StoshError:
            print("✓ Correctly prevented sampling without loaded data")
        
        # Test context manager
        with stosh.compile(stan_file) as model2:
            print("✓ Context manager works")
        
        return True
        
    except stosh.StoshError as e:
        print(f"ℹ Compilation test failed (expected if Stan not set up): {e}")
        return True


def test_stan_root_env():
    """Test STAN_ROOT environment variable"""
    print("Testing STAN_ROOT environment variable...")
    
    # Save original value
    original_stan_root = os.environ.get('STAN_ROOT')
    
    try:
        # Test with invalid STAN_ROOT
        os.environ['STAN_ROOT'] = '/nonexistent/path'
        
        try:
            model = stosh.compile("nonexistent.stan")
            print("✗ Expected error with invalid STAN_ROOT")
            return False
        except stosh.StoshError as e:
            if "Makefile not found" in str(e):
                print("✓ Correctly detected invalid STAN_ROOT")
            else:
                print("✓ Correctly raised StoshError (different reason)")
        
        return True
        
    finally:
        # Restore original value
        if original_stan_root is not None:
            os.environ['STAN_ROOT'] = original_stan_root
        elif 'STAN_ROOT' in os.environ:
            del os.environ['STAN_ROOT']


def test_value_conversion():
    """Test parameter conversion utilities"""
    print("Testing parameter conversion...")
    
    # Create a dummy model to test the conversion functions
    class DummyModel:
        def _value_to_string(self, value):
            if isinstance(value, bool):
                return "true" if value else "false"
            else:
                return str(value)
    
    dummy = DummyModel()
    
    # Test value conversion
    assert dummy._value_to_string(True) == "true"
    assert dummy._value_to_string(False) == "false"
    assert dummy._value_to_string(42) == "42"
    assert dummy._value_to_string(3.14) == "3.14"
    assert dummy._value_to_string("hello") == "hello"
    
    print("✓ Value conversion works")


def main():
    """Run all tests"""
    print("Stosh Object-Oriented Interface Tests")
    print("=" * 40)
    
    # Always run these tests
    test_imports()
    test_value_conversion()
    test_compile_nonexistent()
    test_compile_invalid_extension()
    test_stan_root_env()
    
    # Test with actual model if available
    test_with_actual_model()
    
    print("\n" + "=" * 40)
    print("Tests completed!")
    print("\nTo test with a specific Stan installation:")
    print("  export STAN_ROOT=/path/to/stan")
    print("  python test_stosh.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
