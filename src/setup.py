from setuptools import find_packages, setup

setup(
    name="pysynthbio",
    packages=find_packages(exclude=["source", "docs", "tests"]),
    install_requires=[
        "numpy>=2.2.5",
        "pandas>=2.2.3",
        "requests",
        "keyring>=23.0.0",
    ],
)
