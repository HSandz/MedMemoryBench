"""
Setup script for ReMEM package.
This file is kept for backward compatibility, but the primary configuration
is in pyproject.toml following modern Python packaging standards.
"""

from setuptools import setup, find_packages

# Read dependencies from requirements.txt
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="remem",
    version="0.1.0",
    description="ReMEM: A brain-inspired retrieval-augmented generation system with episodic memory",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Yiheng Shu, Padmaja Jonnalagedda",
    license="Apache License 2.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
