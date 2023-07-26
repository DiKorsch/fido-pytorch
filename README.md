# FIDO Algorithm with improved Concrete Dropout

## Installation
Install the requirements via `pip`
```
python -m pip install -r requirements.txt
```
or with `conda`:

```
conda install torch~=1.13 torchmetrics~=0.7 torchvision~=0.14 matplotlib
```
## Usage

```python
import sys
sys.path.insert(0, "<path to this repo>/src")
import numpy as np
import torch as th

from fido.module import FIDO
from fido import configs as fido_configs

clf = ... # your model
im = dataset[0] # image from your dataset

with th.no_grad():
    predicted_class = clf.predict(im[None])[0].argmax()

optimized = True # setting it to True enables our improved implementation

mask_config = fido_configs.MaskConfig(mask_size=None,
                                      infill_strategy="blur",
                                      optimized=optimized)

fido = FIDO.new(im, mask_config, device=im.device)

fido_config = fido_configs.FIDOConfig(
    learning_rate=1e1,
    iterations=30,
    batch_size=8,
    l1=1e-3, tv=1e-2
)
fido.fit(im, predicted_class, clf, config=fido_config)

print(fido.ssr_logit_p)
print(fido.sdr_logit_p)
```


## License
This work is licensed under a [GNU Affero General Public License][agplv3].

[![AGPLv3][agplv3-image]][agplv3]

[agplv3]: https://www.gnu.org/licenses/agpl-3.0.html
[agplv3-image]: https://www.gnu.org/graphics/agplv3-88x31.png

## Citation
You are welcome to use our code in your research! If you do so please cite it as:

```
@inproceedings{Korsch23:CSD,
    author = {Dimitri Korsch and Maha Shadaydeh and Joachim Denzler},
    booktitle = {German Conference on Pattern Recognition (GCPR)},
    title = {Simplified Concrete Dropout - Improving the Generation of Attribution Masks for Fine-grained Classification},
    year = {2023},
}
```
