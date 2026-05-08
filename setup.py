"""Setup script for medical-diagnostic-assistant."""

from setuptools import find_packages, setup

setup(
    name="medical-diagnostic-assistant",
    version="1.0.0",
    description="Multi-condition medical diagnostic assistant using ensemble ML methods",
    author="Mani",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "streamlit>=1.28.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "medical-dashboard=src.dashboard:main",
        ],
    },
)
