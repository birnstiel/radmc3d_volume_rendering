"""
Setup file for package `dsharp_helper`.
"""
import setuptools  # NOQA
# from numpy.distutils.core import Extension
import pathlib
import warnings

PACKAGENAME = 'radmc3d_volume_rendering'

# the directory where this setup.py resides
HERE = pathlib.Path(__file__).parent

if __name__ == "__main__":
    from numpy.distutils.core import setup

    extensions = []

    def setup_function(extensions):
        setup(
            name=PACKAGENAME,
            description='python routines to volume render data with RADMC3D',
            version='0.0.1',
            long_description=(HERE / "README.md").read_text(),
            long_description_content_type='text/markdown',
            url='https://github.com/birnstiel/radmc3d_volume_rendering',
            author='Til Birnstiel',
            author_email='til.birnstiel@lmu.de',
            license='GPLv3',
            packages=[PACKAGENAME],
            package_dir={PACKAGENAME: 'radmc3d_volume_rendering'},
            package_data={PACKAGENAME: [
                'Makefile',
                'userdef_module.f90',
                ]},
            include_package_data=True,
            install_requires=[
                'scipy',
                'numpy',
                'matplotlib',
                'astropy',
                'pandas',
                'sphinx',
                'nbsphinx',
                'tqdm'],
            zip_safe=False,
            ext_modules=extensions
            )
    try:
        setup_function(extensions)
    except BaseException:
        warnings.warn('Could not compile extensions, code will be slow')
        setup_function([])
