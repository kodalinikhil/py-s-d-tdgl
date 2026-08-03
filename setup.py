from pathlib import Path

from setuptools import find_packages, setup


root = Path(__file__).parent
version = {"__file__": str(root / "tdgl/version.py")}
exec((root / "tdgl/version.py").read_text(), version)

setup(
    name="py-s-d-tdgl",
    version=version["__version__"],
    description="Finite-volume TDGL for single- and multi-component superconductors",
    long_description=(root / "README.md").read_text(),
    long_description_content_type="text/markdown",
    author="Nikhil Kodali and pyTDGL contributors",
    url="https://github.com/kodalinikhil/py-s-d-tdgl",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.9,<3.15",
    install_requires=[
        "cloudpickle",
        "h5py",
        "IPython",
        "joblib",
        "matplotlib",
        "meshpy",
        "numba",
        "numpy",
        "pint",
        "scipy",
        "shapely",
        "tqdm",
    ],
    extras_require={
        "dev": ["black", "isort", "pre-commit", "pytest", "pytest-cov"],
        "docs": [
            "enum_tools",
            "nbsphinx",
            "pillow",
            "sphinx==5.3.0",
            "sphinx-argparse",
            "sphinx-autodoc-typehints",
            "sphinx-rtd-theme>=0.5.2",
            "sphinx_toolbox",
            "sphinxcontrib-bibtex",
        ],
        "pardiso": ["pypardiso"],
        "umfpack": ["swig", "scikit-umfpack"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    keywords="superconductivity TDGL vortex multiband",
)
