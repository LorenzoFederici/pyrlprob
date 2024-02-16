from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name='pyrlprob',
    version='2.1.0',
    author='Lorenzo Federici',
    author_email = 'federicilorenzo94@gmail.com',
    description = 'Train Gym-derived environments in Python/C++ through Ray RLlib',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/LorenzoFederici/pyrlprob',
    classifiers = [
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ],
    install_requires = [
        'dm-tree',
        'gym==0.15.3',
        'lz4',
        'matplotlib',
        'numpy==1.23.5',
        'pandas',
        'pybind11',
        'pytest',
        'ray==1.6.0',
        'scipy',
        'scikit-image',
        'opencv-python; platform_machine=="x86_64"',
        'tabulate',
        'tensorflow',
        'tensorboardx; platform_machine=="x86_64"',
        'torch',
        'typing',
        'PyYAML',
        'protobuf==3.20.3'
    ],
    packages = find_packages(),
    python_requires = '<3.10',
    include_package_data = True)