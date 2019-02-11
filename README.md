# Heirarchical Dynamics Encoder
Heirarchical dynamics encoder (HDE) is a deep learning-based
framework to learn multiple hierarchical nonlinear kinetic slow modes.
They are built on top of transfer operator theory, the variational 
approach to conformational dynamics (VAC), and use a neural network as a
featurizer to provide an optimal nonlinear basis set for VAC, which
finds an optimal linear combination of the bases.

HDEs are similar to (and inspired by) [VAMPNets](https://github.com/markovmodel/deeptime)
and [VDEs](https://github.com/msmbuilder/vde) with some key differences. 
VAMPNets seek to replace the entire MSM construction pipeline of
featurization, dimensionality reduction, and state assignment. 
In the one-dimensional limit, HDEs are formally equivalent to 
VDEs with an exclusive autocorrelation loss, subject to Gaussian noise.
VDEs however, cannot currently generalize to multiple dimensions due to
the lack of an orthogonality constraint on the learned eigenfunctions. 

## Requirements 

**HDE** depends on the following libraries:

 - numpy 
 - scipy
 - keras 
 - tensorflow 
 - scikit-learn

## Installation 

With the necessary requirements all you need to do is clone the
repository and pip install. 

```bash
$ git clone https://github.com/hsidky/hde.git
$ pip install ./hde
```

## Examples 

Below is an example that demonstrates basic usage of **HDE**. Here we are using 
[PyEMMA](http://emma-project.org/latest/) to extract features from a trajectory. 

For the examples presented in the original paper see the paper_notebooks folder.
For other detailed examples see the examples and notebooks folders.

```python 
import pyemma as py
from hde import HDE 

features = py.coordinates.featurizer('system.pdb')
features.add_backbone_torsions(cossin=True)
data = py.coordinates.load('trajectory.pdb', features=features)

n_components = 3 # Number of eigenfunctions to learn.
model = HDE(
    features.dimension(), 
    n_components=n_components, 
    n_epochs=20, 
    lag_time=100,
    batch_normalization=True
)

slow_modes = model.fit_transform(data)
```

## <a name="ack"></a> Acknowledgements 

The **HDE** method and code is inspired by and built upon [VAMPNets](https://github.com/markovmodel/deeptime) and 
[VDEs](https://github.com/msmbuilder/vde).

## License 

**HDE** is provided under an MIT license that can be found in the LICENSE file. By using, distributing, or contributing to this project, you agree to the terms and conditions of this license.

## Cite

If you use this code in your work, please cite: 
