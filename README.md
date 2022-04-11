<div align="center"> <h1><b> pyRLprob </b></div>

<div align="center">

![GitHub Repo stars](https://img.shields.io/github/stars/LorenzoFederici/pyrlprob?style=social)
![GitHub last commit](https://img.shields.io/github/last-commit/LorenzoFederici/pyrlprob)
![GitHub Release Date](https://img.shields.io/github/release-date/LorenzoFederici/pyrlprob)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/LorenzoFederici/pyrlprob)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyrlprob)

</div>

PyRLprob is an open-source python library for easy training, evaluation, and postprocessing of [Gym](https://gym.openai.com/)-based environments, written in python or c++, through [Ray-RLlib](https://docs.ray.io/en/master/rllib.html) reinforcement learning library.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the latest stable release of pyRLprob, with all its dependencies:

```bash
pip install pyrlprob
```

To test if the package is installed correctly, run the following tests:


```python
import pyrlprob
from pyrlprob.tests import test_train, test_train_eval

test_train()
test_train_eval()
```

If the code exits without errors, a folder named `results/` with the test results will be created in your current directory.

## User Guide
Coming soon...

## Credits
pyRLprob has been created by [Lorenzo Federici](https://github.com/LorenzoFederici) in 2021.
For any problem, clarification or suggestion, you can contact the author at [lorenzo.federici@uniroma1.it](mailto:lorenzo.federici@uniroma1.it).

## License
The package is under the [MIT](https://choosealicense.com/licenses/mit/) license.

