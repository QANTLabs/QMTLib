# QMLLib - Quantum Machine Learning Python Library

QMLLib is a Python library designed for training and evaluating quantum and hybrid quantum-classical machine learning models. It integrates with popular quantum computing frameworks such as Qiskit, Amazon Braket, and Cirq, enabling the development of quantum-enhanced machine learning algorithms.

## Features

- **Quantum-Classical Models**: Build and evaluate hybrid models that combine quantum circuits with classical machine learning techniques.
- **Support for Multiple Quantum Platforms**: Compatible with Qiskit, Amazon Braket, and Cirq for seamless quantum model execution on different quantum computing platforms.
- **Extensive ML Integration**: Use popular machine learning libraries like `scikit-learn` and `pandas` in conjunction with quantum models to enhance your workflows.
- **Quantum Circuits for ML**: Easily integrate quantum circuits into your machine learning pipelines using Qiskit’s quantum computing capabilities.

Here’s the updated installation section with local build instructions:

---

## Installation

### Using `pip`

To install the library from PyPI:

```bash
pip install qmt
```

Alternatively, if you want to install the latest version from the GitHub repository:

```bash
pip install git+https://github.com/QANTLabs/QMT.git
```

To install the library locally from your machine (for development or testing purposes), run:

```bash
pip install -e .
```

This will install QMT from the current directory where the `setup.py` file is located.

---

### Dependencies

QMT requires Python 3.7 or higher and includes the following dependencies:

- `qiskit`, `qiskit_machine_learning`, `qiskit_aer`, `qiskit_algorithms`, `qiskit_ibm_runtime`
- `amazon-braket-sdk`, `qiskit_braket_provider`, `cirq`
- `numpy`, `scipy`, `matplotlib`, `pandas`, `scikit-learn`

These dependencies will be automatically installed when you install the package.

## Usage

### Quantum ML Example

Here’s an example of how to use the QMT library for a classification task using the `QuantumMLFramework`:

```python
from qmt import QuantumMLFramework
import numpy as np

# Example usage code
def quantum_ml_example():
    """Example of using the QuantumMLFramework for a classification task."""
    
    # Generate synthetic data
    np.random.seed(42)
    num_samples = 50
    feature_dim = 2
    
    # Generate simple binary classification data
    X = np.random.uniform(0, np.pi, (num_samples, feature_dim))
    y = np.array([0 if np.sum(np.sin(x)) >= 0 else 1 for x in X])
    
    # Split into train and test
    train_size = int(0.8 * num_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Create and configure the quantum ML framework
    qml = QuantumMLFramework(
        device='ibm_qiskit',  # Specify device platform
        task_type='classification',
        n_qubits=feature_dim,
        shots=1024,
        backend_name='aer_simulator',
        feature_map_type='ZZFeatureMap',
        ansatz_type='TwoLocal',
        optimizer_name='COBYLA',
        optimizer_kwargs={'maxiter': 10},
        verbose=True
    )
    
    # Build, train and evaluate the model
    qml.build_model(input_dim=feature_dim)
    qml.fit(X_train, y_train)
    score = qml.score(X_test, y_test)
    
    # Plot training progress
    qml.plot_training_progress()
    
    return qml, score


if __name__ == "__main__":
    # Run classification example with IBM Qiskit
    print("Running Quantum Classification Example with IBM Qiskit...")
    qml_cls, cls_score = quantum_ml_example()
```

This example demonstrates how to create a quantum machine learning classification task using the QMT library. The example uses IBM Qiskit for quantum computation, but similar implementations can be made for AWS Braket and Google Cirq.

### Running the Example

1. **Run the example**:

   ```bash
   python ./examples/classification.py
   ```

   This will execute the classification task, train the model, and display the results.

## Contribution
Contributions are welcome! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
