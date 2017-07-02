# -*- coding: utf-8 -*-

from setuptools import setup
from setuptools import find_packages

version = '0.0.1'


setup(
    name='eva',
    packages=find_packages(exclude=['tests']),
    package_data={
        'eva': [],
    },
    install_requires=[
        'nltk==3.2.4',
        'numpy==1.12.1',
        'pandas==0.20.1',
        'python-crfsuite==0.9.2',
        'regex==2017.5.26',
        'scikit-learn==0.18.1',
        'scipy==0.19.0',
        'boltons==17.1.0',
        'requests==2.18.1'
    ],
    zip_safe=False,
    version=version,
    description='Chatbot EVA',
    author='Victor Ferraz',
    author_email='vfsf@cin.ufpe.br',
    url='https://github.com/victorfsf/eva',
    keywords=[
        'eva',
        'nlp',
        'ml',
        'ai',
        'natural language processing',
        'machine learning',
        'artificial intelligence',
        'chatbot',
        'chat',
        'chatter',
        'chatterbot',
        'virtual assistant',
        'python3',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU Affero General Public License v3',
        'Natural Language :: Portuguese (Brazilian)',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
