# -*- coding: utf-8 -*-
from distutils.core import setup
from pip.req import parse_requirements
from pip.download import PipSession

# dependency
install_reqs = parse_requirements('requirements.txt', session=PipSession())

setup(
    name='sugartensor',
    version='0.0.1.0',
    packages=['sugartensor',
              'sugartensor.sg_data'
              ],
    url='https://github.com/buriburisuri/sugartensor',
    license='MIT',
    author='Namju Kim at Jamonglabs Co.,Ltd.',
    author_email='buriburisuri@gmail.com',
    description='A slim tensorflow wrapper that provides syntactic sugar for tensor variables.',
    install_reqs=[str(ir.req) for ir in install_reqs],
)
