
from distutils.core import setup

setup(
    name = 'tfmf',
    version = '0.1.0',
    author = 'Tymoteusz Wolodzko',
    author_email = 'twolodzko+tfmf@gmail.com',
    packages = ['tfmf'],
    license = 'LICENSE.txt',
    description = 'Matrix factorization with implicit and explicit ratings.',
    install_requires = [
        "numpy>=1.14.1",
        "scikit-learn>=0.13.1",
        "tqdm>=4.19.6",
        "tensorflow>=1.8.0",
        "scipy>=1.0.0" 
    ],
)