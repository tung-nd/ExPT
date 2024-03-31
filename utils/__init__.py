from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel, PeriodicKernel, CosineKernel, LinearKernel, PolynomialKernel, RQKernel, SpectralMixtureKernel
from design_bench.datasets.continuous.ant_morphology_dataset import AntMorphologyDataset
from design_bench.datasets.continuous.dkitty_morphology_dataset import DKittyMorphologyDataset
from design_bench.datasets.discrete.tf_bind_10_dataset import TFBind10Dataset
from design_bench.datasets.discrete.tf_bind_8_dataset import TFBind8Dataset

NAME_TO_FULL_DATASET = {
    'AntMorphology-Exact-v0': AntMorphologyDataset,
    'DKittyMorphology-Exact-v0': DKittyMorphologyDataset,
    'TFBind8-Exact-v0': TFBind8Dataset,
    'TFBind10-Exact-v0': TFBind10Dataset,
}

TASK_ABBREVIATIONS = {
    'ant': 'AntMorphology-Exact-v0',
    'dkitty': 'DKittyMorphology-Exact-v0',
    'tf8': 'TFBind8-Exact-v0',
    'tf10': 'TFBind10-Exact-v0',
}

SYNTHETIC = ['rbf', 'matern', 'linear', 'cosine', 'periodic', 'piecewise', 'polynomial', 'rq', 'deep_kernel']

REAL = ['dkitty', 'ant', 'tf8', 'tf10']

DISCRETE = ['tf8', 'tf10']

CONTINUOUS = [t for t in REAL if t not in DISCRETE]

KERNEL_NAME_MAP = {
    'rbf': RBFKernel,
    'matern': MaternKernel,
    'periodic': PeriodicKernel,
    'cosine': CosineKernel, # no length scale prior
    'linear': LinearKernel, # no length scale prior
    'polynomial': PolynomialKernel, # no length scale prior,
    'rq': RQKernel,
    'mixture': SpectralMixtureKernel
}