import os
import sys
import platform
from setuptools import setup, find_packages


# "setup.py publish" shortcut.
if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist')
    os.system('twine upload dist/*')
    if platform.system() == 'Windows':
        os.system('powershell rm –path dist, model_zoo_pytorch.egg-info –recurse –force')
    else:
        os.system('rm -rf dist model_zoo_pytorch.egg-info')
    sys.exit()

install_requires = ['torch>=1.8.1']

setup(
    name='model_zoo',
    version=0.1,
    description="Model Zoo",
    url='',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
    python_requires='>=3.8.10'
)
