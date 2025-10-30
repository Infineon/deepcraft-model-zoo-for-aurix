# Central Scripts

This folder contains scripts for common functions needed across various AI model projects in the model zoo, including:

- **Helper Functions**: Shared utilities for model conversion, validation, and analysis
- **Requirements Management**: Centralized dependency specifications and validation
- **Testing Utilities**: Common test functions and validation scripts

## Key Components

- **`helper_functions.py`**: Main utility library containing functions for:
  - ONNX model conversion and export (TensorFlow ↔ PyTorch ↔ ONNX)
  - Data preprocessing and tensor operations
  - Docker container management for model compilation

- **`requirements.txt`**: Centralized Python dependencies for the entire model zoo ecosystem

- **`test_requirements.py`**: Comprehensive test suite using pytest to verify:
  - Package installation and compatibility
  - Critical package imports
  - Version compatibility checks
  - Package conflict detection

## Dependencies

Core dependencies include:

- Deep learning frameworks (TensorFlow, PyTorch)
- Model conversion tools (ONNX, tf2onnx)
- Data science stack (NumPy, Pandas, Matplotlib, Seaborn)
- Development tools (Jupyter, Flask)
- Testing tools (pytest)

See `requirements.txt` for the complete list with version specifications.

## License

Please see our [LICENSE](../LICENSE) for copyright and license information.
