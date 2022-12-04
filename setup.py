import os
import sys
from setuptools import setup, find_packages

dirname = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(dirname, "caustic"))

with open(os.path.join(dirname, "README.md"), encoding="utf-8") as f:
    readme = f.read()

setup(
    name="caustic",
    version="0.1.0dev",
    author="Fran Bartolić",
    author_email="fb90@st-andrews.ac.uk",
    url="https://github.com/fbartolic/caustic",
    license="MIT",
    packages=find_packages(where="src"),
    meta_path=os.path.join("src", "caustic", "__init__.py"),
    description="JAX-based library for single lens photometric and astrometric gravitational microlensing",
    long_description=readme,
    long_description_content_type="text/markdown",
    install_requires=["jax", "jaxlib"],
    extra_require={
        "test": ["pytest>=3.6"],
        "docs": [
            "sphinx>=3.3",
            "sphinx-book-theme",
            "matplotlib",
            "numpyro",
            "arviz",
        ],
    },
    package_data={"": ["README.md", "LICENSE"]},
    include_package_data=True,
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
    ],
    zip_safe=True,
)
