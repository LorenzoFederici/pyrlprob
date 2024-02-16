<p align="center">
  <img align="center" src="https://github.com/LorenzoFederici/pyrlprob/blob/main/logo.png?raw=true" width="500" />
</p>

<div align="center">

![GitHub Repo stars](https://img.shields.io/github/stars/LorenzoFederici/pyrlprob?style=social)
![GitHub last commit](https://img.shields.io/github/last-commit/LorenzoFederici/pyrlprob)
![GitHub Release Date](https://img.shields.io/github/release-date/LorenzoFederici/pyrlprob)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/LorenzoFederici/pyrlprob)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyrlprob)

</div>

PyRLprob is an open-source python library for training, evaluation, and postprocessing of [Gym](https://gym.openai.com/)-based environments, written in Python, through [Ray-RLlib](https://docs.ray.io/en/master/rllib.html) reinforcement learning library.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the latest stable release of pyRLprob, with all its dependencies:

```bash
pip install pyrlprob
```

To test if the package is installed correctly, run the following tests:


```python
from pyrlprob.tests import *

test_train_eval_py()
```

If the code exits without errors, a folder named `results/` with the test results will be created in your current directory.

## User Guide
[Latest user guide](https://drive.google.com/file/d/1bNs2g50cxtmAGhhB1_Kf3hX8pdkbCplZ/view?usp=share_link).

## Credits
pyRLprob has been created by [Lorenzo Federici](https://github.com/LorenzoFederici) in 2021.
For any problem, clarification or suggestion, you can contact the author at [lorenzof@arizona.edu](mailto:lorenzof@arizona.edu).

## License
The package is under the [MIT](https://choosealicense.com/licenses/mit/) license.

