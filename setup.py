"""Setup configuration for TurbTwin package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    requirements = [
        line.strip()
        for line in requirements_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
else:
    requirements = [
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.1.0",
        "tensorflow>=2.15.0",
        "scipy>=1.11.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ]

setup(
    name="turbtwin",
    version="1.0.0",
    author="Narjisse Kabbaj, Naila Marir, Mohamed F. El-Amin",
    author_email="nmarir@effatuniversity.edu.sa",
    description="Physics-Informed Machine Learning Digital Twin for Thermal Turbulent Jet Temperature Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/TurbTwin",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/TurbTwin/issues",
        "Documentation": "https://github.com/yourusername/TurbTwin#readme",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "turbtwin=turbtwin.__main__:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
