"""Setup."""

import os
from setuptools import setup, find_packages


here = os.path.abspath(os.path.dirname(__file__))
try:
    README = open(os.path.join(here, 'README.md'), encoding='utf-8').read()
except IOError:
    README = ''

try:
    install_requires = open(os.path.join(
        here, 'requirements.txt'), encoding='utf-8').read().split("\n")
except IOError:
    install_requires = []


setup(
    name='ravens_torch',
    version='0.1',
    description='PyTorch adaptation of Ravens - a collection of simulated tasks in PyBullet for learning vision-based robotic manipulation.',
    long_description='\n\n'.join([README]),
    long_description_content_type='text/markdown',
    url='https://github.com/thomaschabal/transporter-nets-torch',
    packages=find_packages(),
    install_requires=install_requires,
)
