#!/usr/bin/env python3
"""
Example usage of Stosh object-oriented interface.

This script demonstrates how to compile a Stan model and run sampling
using the new object-oriented API.

Prerequisites:
1. Stan model file (e.g., bernoulli.stan)
2. Data file (e.g., bernoulli.data.json) - optional
3. GNU make and Stan development environment set up

Run from the directory containing your .stan file:
    python example.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import stosh


def main():
    print("Stosh Object-Oriented Interface Example")
    print("=" * 45)

    bern_stanfile = os.path.join('examples', 'bernoulli.stan')
    bern_datafile = os.path.join('examples', 'bernoulli.data.json')
    
    try:
        # Example 1: Compile and use model
        print("\n1. Compiling Stan model...")
        model = stosh.compile(bern_stanfile)
            
        print(f"   Compiled successfully!")
        
        print("\n2. Loading data...")
        model.load_data(bern_datafile, seed=12345)
        print(f"   Data loaded. Model name: {model.name}")
        
        print("\n3. Running HMC-NUTS sampling...")
        result = model.hmc_nuts(
            num_chains=2,
            warmup=100,
            samples=100,
            refresh=50
        )
        
        print(f"   Sampling completed!")
        print(f"   Results saved to: {result['output_dir']}")
        
    except stosh.StoshError as e:
        print(f"   Error: {e}")
        return 1
    
    # Example 2: Using context manager
    try:
        print("\n4. Using context manager...")
        with stosh.compile(bern_stanfile) as model:
            model.load_data(bern_datafile, seed=42)
            result = model.hmc_nuts(
                num_chains=1,
                warmup=50,
                samples=50,
                stepsize=0.1,
                max_depth=8,
                delta=0.9,
                refresh=0
            )
            print(f"   Results saved to: {result['output_dir']}")
        print("   Model automatically cleaned up!")
        
    except stosh.StoshError as e:
        print(f"   Error: {e}")
        return 1

    rosenbrock_stanfile = os.path.join('examples', 'rosenbrock.stan')
    
    # Example 3: Model without data
    try:
        print("\n5. Model without data...")
        model = stosh.compile(rosenbrock_stanfile)  # Model with no data block
        model.load_data(seed=123)  # No data file needed
        
        result = model.hmc_nuts(
            warmup=25,
            samples=25,
            refresh=0
        )
        print(f"   Results saved to: {result['output_dir']}")
        
    except stosh.StoshError as e:
        print(f"   Info: {e}")
        print("   (This is expected if model file doesn't exist)")
    
    print("\n" + "=" * 45)
    print("Example completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
