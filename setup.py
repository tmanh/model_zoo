from setuptools import setup, find_packages


install_requires = []  # 'torch>=1.8.1'

setup(
    name='tma_model_zoo',
    version=0.1,
    description="TMA Model Zoo",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
    python_requires='>=3.7.0',
    license='MIT',
)
