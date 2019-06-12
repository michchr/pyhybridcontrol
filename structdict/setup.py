
from setuptools import setup
from setuptools import Extension


setup_kwargs = dict(
      name='structdict',
      version='0.1',
      description='StructDict',
      long_description="",
      author='Chris Michalak',
      author_email='chris.zulu@hotmail.com',
      license='MIT',
      url='',
      packages=['structdict'],
      package_dir={'structdict': '../structdict'},
     )

setup_kwargs['ext_modules'] = [
                Extension("structdict._accessors",
                          sources = ["./_accessors.c"],
                          undef_macros=['Py_LIMITED_API'],
                          py_limited_api = False)]

setup(**setup_kwargs)