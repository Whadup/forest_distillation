from setuptools import setup, find_packages
from setuptools import Extension, setup
from Cython.Build import cythonize
with open('README.md') as _f:
    _README_MD = _f.read()

_VERSION = '0.1'



ext_modules = [
    Extension(
        "random_forest_distillation.kernels",
        ["random_forest_distillation/kernels.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    ),
    # Extension(
    #     "random_forest_robustness.c_lower_bound_manager",
    #     ["random_forest_robustness/c_lower_bound_manager.pyx"],
    #     extra_compile_args=['-fopenmp'],
    #     extra_link_args=['-fopenmp'],
    #     language="c++"
    # )
]

setup(
    name='random_forest_distillation',
    version=_VERSION,
    description='Approximate a sklearn RandomForestClassifier using a single Decision Tree.',
    long_description=_README_MD,
    classifiers=[
    ],
    url='https://github.com/Whadup/random_forest_distillation',  
    download_url='https://github.com/Whadup/random_forest_distillation/tarball/{}'.format(_VERSION), 
    author='Lukas Pfahler', 
    author_email='lukas.pfahler@tu-dortmund.de',  
    packages=find_packages(include=['random_forest_distillation*']),  
    test_suite="testing",
    setup_requires=["pytest-runner", "Cython", "scikit-learn>=1.0", "numpy", "tqdm"],
    ext_modules = cythonize(ext_modules, language_level = "3"),
    tests_require=["pytest", "pytest-cov"],
    include_package_data=True,
    license='MIT',
    python_requires=">3.7",
    keywords='scikit-learn random forest to decision tree distillation'
)

