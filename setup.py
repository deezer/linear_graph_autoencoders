from setuptools import setup
from setuptools import find_packages

setup(name='linear_gae',
      description='Keep It Simple: Graph Autoencoders Without Graph Convolutional Networks',
      author='Deezer Research',
      install_requires=['networkx==2.2',
                        'numpy',
                        'scikit-learn',
                        'scipy==1.*',
                        'tensorflow==1.*'],
      package_data={'linear_gae': ['README.md']},
      packages=find_packages())