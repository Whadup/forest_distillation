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
    name='random_forest_robustness', # TODO: rename. 
    version=_VERSION,
    description='An empty project base.',
    long_description=_README_MD,
    classifiers=[
        # TODO: typing.
        "Typing :: Typed"
    ],
    url='https://github.com/Whadup/random_forest_distillation',  # TODO.
    download_url='https://github.com/Whadup/random_forest_distillation/tarball/{}'.format(_VERSION),  # TODO.
    author='Lukas Pfahler',  # TODO.
    author_email='lukas.pfahler@tu-dortmund.de',  # TODO.
    packages=find_packages(include=['random_forest_distillation*']),  # TODO.
    test_suite="testing",
    setup_requires=["pytest-runner"],
    ext_modules = cythonize(ext_modules, language_level = "3"),
    tests_require=["pytest", "pytest-cov"],
    include_package_data=True,
    license='MIT',  
    keywords='empty project TODO keywords'
)

