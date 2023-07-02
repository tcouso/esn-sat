from setuptools import setup, find_packages

setup(
    name='foreutils',
    version='0.1.0',
    author='Tomás Couso Coddou',
    author_email='tomascousocoddou@gmail.com',
    description='Utilities for signal forecasting of satelite images with echo-state networks (ESN)',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
