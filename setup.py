from setuptools import setup, find_packages
setup(name='pyrlprob',
    version='1.0.0',
    author='Lorenzo Federici',
    author_email = 'federicilorenzo94@gmail.com',
    description = 'Train easily OpenAI-Gym environments through Ray-RLlib',
    long_description = 'file: README.md',
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/LorenzoFederici/pyrlprob',
    classifiers = [
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ],
    install_requires = [
        'gym',
        'importlib',
        'matplotlib',
        'numpy',
        'pandas',
        'ray==1.6.0',
        'scipy',
        'tensorflow',
        'torch',
        'typing',
        'PyYAML'
    ],
    packages = find_packages(),
    python_requires = '>=3.7',
    include_package_data = True)