from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name='pyrlprob',
    version='1.1.9',
    author='Lorenzo Federici',
    author_email = 'federicilorenzo94@gmail.com',
    description = 'Train easily OpenAI-Gym environments through Ray-RLlib',
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
        'gym',
        'lz4',
        'matplotlib',
        'numpy',
        'pandas',
        'pytest',
        'ray==1.6.0',
        'scipy',
        'scikit-image',
        'opencv-python',
        'tabulate',
        'tensorflow',
        'tensorboardx',
        'torch',
        'typing',
        'PyYAML'
    ],
    packages = find_packages(),
    python_requires = '>=3.7',
    include_package_data = True)