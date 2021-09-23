from setuptools import setup, find_packages

with open("README.md", 'r') as file_handle:
    long_description = file_handle.read()

setup(
    name = "MMFlow",
    version = "0.0.1",
    author = "Xinqiang Ding",
    author_email = "xqding@umich.edu",
    description = "MMFlow",
    long_description = long_description,
    long_description_content_type = "text/x-rst",
    url = "",
    packages = find_packages(),
    install_requires=['numpy>=1.21.0',
                      'torch>=1.8.0',
                      'openmm>=7.5.1'],
    license = 'MIT',
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
