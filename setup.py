from setuptools import setup, find_packages

setup(
    name="neural-predictive-maintenance",
    version="1.0.0",
    author="Your Name",
    description="AI-powered bearing fault detection via vibration signal analysis",
    packages=find_packages(where="."),
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "torch>=2.0",
        "scikit-learn>=1.3",
        "streamlit>=1.28",
        "plotly>=5.14",
        "pandas>=2.0",
        "pywavelets>=1.4",
        "shap>=0.42",
        "matplotlib>=3.7",
        "seaborn>=0.12",
    ],
)
