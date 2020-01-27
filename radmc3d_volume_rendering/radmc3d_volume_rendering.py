from pathlib import Path
import shutil
import os
import pkg_resources

import numpy as np
import matplotlib.pyplot as plt

from disklab import radmc3d as r3

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument(dest='model', choices=['sphere', 'disk'])
# args = parser.parse_args()

src_dir = None


def locate_src_dir(dir=None):
    global src_dir
    if dir is not None:
        src_dir = os.path.expanduser(dir)
    elif src_dir is not None:
        print('src_dir already set. Change it by setting the `dir` keyword')
    else:
        try:
            executable = Path(shutil.which('radmc3d'))

            if executable.is_symlink():
                executable = executable.resolve()

        except TypeError:
            executable = Path('.')

        if (executable.parent.parent / 'src').is_dir():
            src_dir = executable.parent.parent / 'src'

    if src_dir is None:
        print('could not find RADMC3D source directory, specify it with the `dir` keyword.')
    else:
        print(f'RADMC3D source dir set to \'{src_dir}\'')


def make(path):
    path = Path(path)
    cwd = Path().cwd()

    try:
        os.chdir(path)
        r3.radmc3d(f'SRC={src_dir}', executable='make')
    except Exception as e:
        raise e
    finally:
        os.chdir(cwd)


def write_sources(path):
    """Write out the Makefile and source file for compiling radmc3d

    Args:
        path (str|pathlib.Path): path into which the setup should be written

    Returns:
        None, but write files into `path`

    """
    path = Path(path)

    if not path.is_dir():
        path.mkdir()

    makefile_path = pkg_resources.resource_filename(__name__, 'Makefile')
    usermodule_path = pkg_resources.resource_filename(__name__, 'userdef_module.f90')

    shutil.copy(makefile_path, path)
    shutil.copy(usermodule_path, path)


def radmc3d_setup(input_data, path='radmc3d_rendering_setup'):
    """Write out the setup of radmc3d.

    This includes writing the setup files (`write_setup_files`) and a guess
    of the transfer fuction (`write_transfer_options`). You still need to write
    the source files, compile and run.

    Args:
        input_data (str | dict): either a path to a npz file that contains the data
            or a dictionary that contains the same data entries. These are

                - ``r_i`` radial interfaces
                - ``t_i`` theta (polar angle) interfaces
                - ``p_i`` phi (azimuthal angle) interfaces
                - ``rho`` gas density of shape (r, theta, phi)

        path (str): path of the directory where the setup will be written.

    Returns: None
    """
    if isinstance(input_data, dict):
        r_i  = input_data['r_i']
        t_i  = input_data['t_i']
        p_i  = input_data['p_i']
        rhog = input_data['rho']
    elif isinstance(input_data, str):
        with np.load(input_data) as fid:
            r_i  = fid['r_i']
            t_i  = fid['t_i']
            p_i  = fid['p_i']
            rhog = fid['rho']
    else:
        raise ValueError('input_data must be dict or path to existing file')

    n_t  = len(t_i) - 1
    #
    # NON-SCIENTIFIC RESCALING:
    #
    rhog /= rhog[:, (n_t + 1) // 2, :].mean(-1)[:, None, None]
    #
    # for the NON-SCIENTIFIC OPACITIES: find limits
    #
    ir = np.argmax(rhog[:, (n_t + 1) // 2, :].max(-1) / rhog[:, (n_t + 1) // 2, :].min(-1))
    vmin = rhog[ir, (n_t + 1) // 2, :].min(-1)
    vmax = rhog[ir, (n_t + 1) // 2, :].max(-1)

    mean = vmax
    sigma = 0.5 * (vmax - vmin)

    # writes out the properties for the density to source function scaling
    write_transfer_options(mean=mean, sigma=sigma, path=path)

    # writes out density, radmc3d.inp and other needed stuff for radmc3d
    write_setup_files(r_i, t_i, p_i, rhog, path=path)


def write_setup_files(r_i, t_i, p_i, rhog, path='.'):
    """Writes out the radmc3d input files:

        - ``gas_density.inp``
        - ``wavelength_micron.inp``
        - ``radmc3d.inp``
        - ``amr_grid.inp``
        - ``transfer.inp``

    Args:
        r_i (array): radial grid interfaces
        t_i (array): theta grid interfaces
        p_i (array): phi grid interfaces
        rhog (array): density array of shape (nr, nt, np)
        path (str | pathlib.Path): path to folder where data is written

    None
    """
    path = Path(path)
    #
    # Write the wavelength file
    #

    Lambda = np.linspace(0, 20, 20)
    with open(path / 'wavelength_micron.inp', 'w') as fid:
        fid.write(str(len(Lambda)) + '\n')
        for value in Lambda:
            fid.write(f'{value:13.6e}\n')

    n_r  = len(r_i) - 1
    n_t  = len(t_i) - 1
    n_p  = len(p_i) - 1
    #
    # Write the grid file
    #
    with open(path / 'amr_grid.inp', 'w') as fid:
        fid.write('1\n')                      # iformat
        fid.write('0\n')                      # AMR grid style  (0=regular grid, no AMR)
        fid.write('100\n')                    # Coordinate system
        fid.write('0\n')                      # gridinfo
        fid.write('1 1 1\n')                  # Include x,y,z coordinate
        fid.write(f'{n_r} {n_t} {n_p}\n')        # Size of grid
        for value in r_i:
            fid.write(f'{value:13.6e}\n')
        for value in t_i:
            fid.write(f'{value:13.6e}\n')
        for value in p_i:
            fid.write(f'{value:13.6e}\n')
    #
    # Write the density file
    #
    with open(path / 'gas_density.inp', 'w') as fid:
        fid.write('1\n')                     # Format number
        fid.write(f'{n_r * n_t * n_p}\n')    # Nr of cells
        fid.write('1\n')                     # Format number
        data = rhog.ravel(order='F')
        data.tofile(fid, sep='\n', format='%13.6e')
        fid.write('\n')
    #
    # Write the radmc3d.inp control file
    #
    with open(path / 'radmc3d.inp', 'w') as fid:
        fid.write('incl_userdef_srcalp = 1')


def write_transfer_options(mean=1.0, sigma=10.0, path='.'):
    """Write out the parameters for the transfer function.

    Args:
        mean (float): mean of the transfer function
        sigma (float): standard deviation of the transfer function
        path (str|pathlib.Path): path to folder where output will be written to.

    Returns:
        None

    """
    path = Path(path)

    with open(path / 'transfer.inp', 'w') as fid:
        fid.write(f'transfer_density_mean = {mean:13.6e}\n')
        fid.write(f'transfer_density_sigm = {sigma:13.6e}\n')


def callit(command=None, executable='./radmc3d', path=os.curdir):
    """
    Once the code is compiled and all input present, call the raytracing.

    command : str
        the radmc3d command to be executed

    executable : str
        the command to run radmc3d (=path to the radmc3d executable)

    path : str
        path to the working directory where radmc3d is called

    """
    if command is None:
        command = 'image lambda 10 sizeradian 2 posang 180 projection 1 locobsau 0 0 .5 pointau -1 0 0 nofluxcons npix 400'

    r3.radmc3d(command, executable=executable, path=path)


def plotit(path='.', log=False, **kwargs):
    im = r3.read_image(filename=str(Path(path) / 'image.out'))
    plt.gca().clear()

    vmin = kwargs.pop('vmin', im.image.min())
    vmax = kwargs.pop('vmin', im.image.max())
    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'Reds'

    if log:
        plt.imshow(np.log10(im.image), vmin=np.log10(vmin), vmax=np.log10(vmax), **kwargs)
    else:
        plt.imshow(im.image, vmin=vmin, vmax=vmax, **kwargs)


locate_src_dir(dir=None)
