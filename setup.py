# -*- coding: utf-8 -*-
try:
    from setuptools import setup
    from setuptools.command.install import install
except ImportError:
    from distutils.core import setup
import pip
from pip.req import parse_requirements
from pip.download import PipSession

install_first = [
    'numpy'
]
git_submodules = [
    'pylearn2',
    'jobman'
]


class CustomInstall(install):
    """Customized setuptools install command.

    Installs git submodules and packages that need to be installed first."""

    def run(self):
        import os
        from distutils.sysconfig import get_python_lib

        for package in install_first:
            pip.main(['install', package])

        install.do_egg_install(self)

        current_dir = os.path.dirname(os.path.realpath(__file__))
        for submodule in git_submodules:
            pth_path = os.path.join(get_python_lib(), submodule + ".pth")
            with open(pth_path, 'w') as pth:
                pth.write(os.path.join(current_dir, submodule) + os.linesep)


def requirements():
    install_reqs = parse_requirements('requirements.txt', session=PipSession())
    return [str(ir.req) for ir in install_reqs]


config = {
    'description': 'Genretron',
    'author': ['Carmine Paolino'],
    'url': 'https://github.com/crmne/Genretron',
    'download_url': 'https://github.com/crmne/Genretron.git',
    'author_email': ['carmine@paolino.me'],
    'version': '1.0.0',
    'install_requires': requirements(),
    'packages': ['genretron'],
    # 'scripts': ['bin/train'],
    'name': 'genretron',
    # 'test_suite': 'nose.collector',
    'cmdclass': {'install': CustomInstall}
}

setup(**config)
