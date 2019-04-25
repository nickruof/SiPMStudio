from setuptools import setup, find_packages

setup(
    name="SiPMStudio",
    version="0.0.0",
    author="Nicholas Ruof",
    author_email="nickruof@uw.edu",
    packages=find_packages(),
    install_requires=[
        "numpy", "scipy", "pandas"
    ]
    )
