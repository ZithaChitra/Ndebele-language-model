from setuptools import setup, find_packages
from codecs import open
from os import path


__version__ = "0.1.0"

here = path.abspath(path.dirname(__file__))

# get dependencies and installs
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
dependency_links = [x.strip().replace('git+', '') for x in all_reqs if x.startswith('git+')]

setup(
    name='Ndebele (nd) languuage model',
    version=__version__,
    description='A model of the Ndebele Language',
    url='https://github.com/ZithaChitra/Ndebele-language-model',
    download_url='https://github.com/ZithaChitra/Ndebele-language-model/tree/main/nd_lang_model',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    author='Zitha Chitra',
    install_requires=install_requires,
    setup_requires=['numpy>=1.10', 'scipy>=0.17'],
    dependency_links=dependency_links,
    author_email='realblessingchitra@gmail.com'
)



