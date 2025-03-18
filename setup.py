from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="diffco",  # Replace with your package name
    version="1.1.0",  # Replace with your version
    author="Yuheng Zhi",
    author_email="yzhi@ucsd.edu",
    description="Differentiable Proxy Model for Collision Checking in Robot Motion Generation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ucsdarclab/diffco",  # Replace with your repository URL
    packages=find_packages(),  # Automatically find and include packages
    install_requires=requirements,  # Use requirements from requirements.txt
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: BSD-3-Clause",
        "Operating System :: OS Independent",
    ],
)