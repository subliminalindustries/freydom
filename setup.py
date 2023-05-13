from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='freydom',
    version='0.1.0',
    description='Microwave auditory effect vocal content isolator',
    long_description=readme,
    author='Daniel R. Azulay',
    author_email='daniel@danielazulay.eu',
    url='https://github.com/drazulay/freydom',
    license=license,
    packages=find_packages(exclude=('tests', 'docs', 'data'))
)
