# pyRLprob

pyRLprob is a Python library for easy training, evaluation, and postprocessing of [OpenAI-Gym](https://gym.openai.com/) environments through [Ray-RLlib](https://docs.ray.io/en/master/rllib.html) Reinforcement Learning library.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the latest stable release of pyRLprob, with all its dependencies:

```bash
pip install pyrlprob
```

To test if the package was installed correctly, run the following test:


```python
import pyrlprob
from pyrlprob.tests import test

test()
```

If the code exits without errors, you should see a folder named `results/` with the test results in your current directory.

## Credits
pyRLprob has been created by Lorenzo Federici in 2021.
For any problem, doubt, clarification or suggestion, you can contact me at [lorenzo.federici@uniroma1.it](mailto:lorenzo.federici@uniroma1.it).

## License
The package is under the [MIT](https://choosealicense.com/licenses/mit/) license.

