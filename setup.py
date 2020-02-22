from setuptools import setup, find_packages

setup(
    name="connect_four_bot",
    version="0.1.0",
    author="Thorben jensen",
    author_email="jensen.thorben@gmail.com",
    packages=find_packages(),
    install_requires=["gym>=0.14", "numpy>=1.17.0"],
    extras_require={"dev": ["black", "pylama", "pytest", "rope"]},
)
