# -*- coding: utf-8 -*-
from distutils.core import setup

exec(open('sugartensor/_version.py').read())
setup(
    name='sugartensor',
    packages=['sugartensor'],
    version=__version__,
    description='A slim tensorflow wrapper that provides syntactic sugar for tensor variables.',
    author='Namju Kim at Jamonglabs Co.,Ltd.',
    author_email='buriburisuri@gmail.com',
    url='https://github.com/buriburisuri/sugartensor',
    download_url='https://github.com/buriburisuri/sugartensor/tarball/' + __version__,
    keywords=['tensorflow', 'sugar', 'sugartensor', 'slim', 'wrapper'],
    classifiers=[],
    license='MIT',
)
