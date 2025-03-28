from setuptools import setup, find_packages

setup(
    name="qmt",  # Change to your library name
    version="0.1",
    packages=find_packages(include=["qmt", "qmt.*"]),
    install_requires=[
        "qiskit",
        "qiskit_aer",
        "qiskit_machine_learning",
        "qiskit_algorithms",
        "qiskit_ibm_runtime",
        "amazon-braket-sdk",
        "qiskit_braket_provider",
        "cirq",
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "scikit-learn"
    ],  # Add dependencies if needed
    author="AsithaKKD",
    author_email="asithaindrajithk9@gmail.com",
    description="A Python library for training and evaluating quantum and hybrid quantum-classical machine learning models.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/QANTLabs/QMT",  # Update with your repo
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Quantum Computing"
    ],
    python_requires='>=3.7',
    entry_points={
        "console_scripts": [
            "qmt-train=qmt.cli:train",
            "qmt-test=qmt.cli:test"
        ]
    },
    include_package_data=True,
)
