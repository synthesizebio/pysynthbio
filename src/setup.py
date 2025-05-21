from setuptools import find_packages, setup

setup(
    name="pysynthbio",
    packages=find_packages(exclude=["source", "docs", "tests"]),
)
