from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='hde',
    version='0.1.0',
    description='Heirarchical Dynamics Encoder',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/hsidky/hde',
    author='Hythem Sidky, Wei Chen',
    author_email='hythem@sidky.io',
    classifiers=[  
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Chemistry',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ],
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    python_requires='>=3.5',
    install_requires=['numpy', 'scipy', 'scikit-learn', 'tensorflow', 'keras']
)