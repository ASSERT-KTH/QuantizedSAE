"""
Setup script for quantized SAE package.
"""

from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="quantized-sae",
    version="0.1.0",
    description="Quantized Sparse Autoencoder implementations for efficient representation learning",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="sparse autoencoder quantization machine learning",
    project_urls={
        "Source": "https://github.com/yourusername/quantizedSAE",
        "Tracker": "https://github.com/yourusername/quantizedSAE/issues",
    },
)