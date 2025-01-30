from setuptools import setup

setup(
    name='MI-identifiability',
    version='1.0.0',
    description='A package to test identifiability of MI criteria',
    url='https://github.com/MelouxM/MI-identifiability',
    author='Maxime Méloux',
    author_email='maxime.meloux@protonmail.com',
    license='MIT',
    packages=['mi_identifiability'],
    install_requires=[
        'tqdm~=4.66.5',
        'matplotlib~=3.8.3',
        'numpy~=1.26.4',
        'scipy~=1.12.0',
        'pandas~=2.2.1',
        'torch~=2.4.1',
        'networkx~=3.2.1',
        'torchvision~=0.19.1',
    ]
)
