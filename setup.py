#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

REQUIREMENTS = ["datasets", "transformers", "torch"]

TEST_REQUIREMENTS = ["pytest"]
QUALITY_REQUIREMENTS = ["black", "ruff"]
TRAINING_REQUIREMENTS = ["lightning", "wandb", "torchmetrics"]

EXTRAS_REQUIREMENTS = {
    "dev": TEST_REQUIREMENTS + QUALITY_REQUIREMENTS,
    "test": TEST_REQUIREMENTS,
    "training": TRAINING_REQUIREMENTS,
}
setup(
    author="Bas Krahmer",
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="LEAF: Linguistic Environmental Analysis of Food products",
    install_requires=REQUIREMENTS,
    license="MIT",
    include_package_data=True,
    name="leaf",
    packages=find_packages(include=["src", "src.*"]),
    test_suite="tests",
    tests_require=TEST_REQUIREMENTS,
    extras_require=EXTRAS_REQUIREMENTS,
    url="https://github.com/baskrahmer/LEAF",
    version="0.0.0",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    zip_safe=False,
)
