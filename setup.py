#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

REQUIREMENTS = ["datasets~=2.18", "transformers~=4.41", "torch~=2.2", "numpy~=1.26"]

TEST_REQUIREMENTS = ["pytest"]
QUALITY_REQUIREMENTS = ["black", "ruff"]
TRAINING_REQUIREMENTS = ["lightning~=2.2", "wandb~=0.16", "torchmetrics~=1.3"]
VISUALIZATION_REQUIREMENTS = ["matplotlib", "seaborn"]
OPENAI_REQUIREMENTS = ["openai", "tiktoken"]

EXTRAS_REQUIREMENTS = {
    "dev": TEST_REQUIREMENTS + QUALITY_REQUIREMENTS + VISUALIZATION_REQUIREMENTS + TRAINING_REQUIREMENTS,
    "test": TEST_REQUIREMENTS,
    "training": TRAINING_REQUIREMENTS,
    "visualization": VISUALIZATION_REQUIREMENTS,
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
