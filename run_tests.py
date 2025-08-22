#!/usr/bin/env python3
"""
Test runner for the energy fault prediction system
"""
import os
import sys
import pytest

def main():
    """Run the test suite"""
    # Create tests directory if it doesn't exist
    os.makedirs('tests', exist_ok=True)
    
    # Run pytest with verbose output
    pytest.main(['-v', 'tests/'])

if __name__ == "__main__":
    main()