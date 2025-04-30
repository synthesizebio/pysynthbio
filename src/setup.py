from setuptools import setup, find_packages

setup(
    name="pysynthbio",
    packages=find_packages(exclude=["source", "docs", "tests"]),
)
