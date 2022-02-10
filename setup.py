# yapf: disable

from setuptools import setup, find_packages
from setuptools import Extension, find_packages, setup
import sys
import os

packages = find_packages()
requirements = [
    "torch>=1.7.1",
    "librosa",
    "torchaudio>=0.7.2",
    "astropy",
    "editdistance",
    "pyctcdecode",
    "kenlm"
]

VERSION = {}  # type: ignore
with open("__version__.py", "r") as version_file:
    exec(version_file.read(), VERSION)

if sys.platform == "darwin":
    extra_compile_args = ["-stdlib=libc++", "-O3"]
else:
    extra_compile_args = ["-std=c++11", "-O3"]

class NumpyExtension(Extension):
    """Source: https://stackoverflow.com/a/54128391"""

    def __init__(self, *args, **kwargs):
        self.__include_dirs = []
        super().__init__(*args, **kwargs)

    @property
    def include_dirs(self):
        import numpy

        return self.__include_dirs + [numpy.get_include()]

    @include_dirs.setter
    def include_dirs(self, dirs):
        self.__include_dirs = dirs

extensions = [
    NumpyExtension(
        "fairseq.data.data_utils_fast",
        sources=["fairseq/data/data_utils_fast.pyx"],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),

]

setup(
    name="k_wav2vec",
    version=VERSION["version"],
    description="Multimodal Speech Sentiment Recognition Library",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    url="https://github.com/JoungheeKim/",
    author="JoungHee Kim",
    author_email="onlyl4youu@gmail.com",
    license="Apache-2.0",
    packages=find_packages(include=["fairseq", "fairseq_cli", 'preprocess']),
    install_requires=requirements,
    python_requires=">=3.6.0",
    setup_requires=["pytest-runner"],
    #tests_require=["pytest"],
    package_data={},
    include_package_data=True,
    dependency_links=[],
    #cmdclass=cmdclass,
    ext_modules=extensions,
    zip_safe=False,
)