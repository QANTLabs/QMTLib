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
