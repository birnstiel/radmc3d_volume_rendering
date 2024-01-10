from pathlib import Path
import shutil
import os
import pkg_resources
import warnings
from types import SimpleNamespace
import subprocess

from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as const

pc = const.pc.cgs.value


class Renderer():

    def __init__(self, src_dir=None, path=None):
        self.src_dir = None
        self.locate_src_dir(src_dir)
        self.path = path

        if path is None:
            self.path = Path('radmc3d_setup')

        self.r_i = None
        self.t_i = None
        self.p_i = None
        self.rhog = None

    def locate_src_dir(self, src_dir=None):
        """
        Locates the RADMC3D source directory.

        Args:
            dir (str, optional): The directory path to the RADMC3D source directory. Defaults to None.

        Returns:
            None
        """
        if src_dir is not None:
            src_dir = os.path.expanduser(src_dir)
        elif self.src_dir is not None:
            warnings.warn(
                'src_dir already set. Change it by setting the `src_dir` keyword in `locate_src_dir`')
            return
        else:
            try:
                executable = Path(shutil.which('radmc3d'))

                if executable.is_symlink():
                    executable = executable.resolve()

            except TypeError:
                executable = Path('.')

            if (executable.parent.parent / 'src').is_dir():
                # Add your code here
                pass
                src_dir = executable.parent.parent / 'src'

        self.src_dir = src_dir

        if src_dir is None:
            print(
                'could not find RADMC3D source directory, specify it with the `src_dir` keyword.')
        else:
            print(f'RADMC3D source dir set to \'{src_dir}\'')

    def make(self):
        """
        Compiles the RADMC3D source code with the userdef_module.f90 file.
        """
        path = Path(self.path)
        callit(command=f'SRC={self.src_dir}',
               executable='make', path=self.path, verbose=2)

    def write_sources(self):
        """Write out the Makefile and source file for compiling radmc3d into `self.path`"""
        path = Path(self.path)

        if not path.is_dir():
            path.mkdir()

        makefile_path = pkg_resources.resource_filename(__name__, 'Makefile')
        usermodule_path = pkg_resources.resource_filename(
            __name__, 'userdef_module.f90')

        shutil.copy(makefile_path, path)
        shutil.copy(usermodule_path, path)

    def read_data(self, input_data):
        """Reads the data from file or dictionary `input_data`

            Args:
            input_data (str | dict): either a path to a npz file that contains the data
                or a dictionary that contains the same data entries. These are

                    - ``r_i`` radial interfaces
                    - ``t_i`` theta (polar angle) interfaces
                    - ``p_i`` phi (azimuthal angle) interfaces
                    - ``rho`` gas density of shape (r, theta, phi)


        Returns: None
        """
        path = Path(self.path)

        if isinstance(input_data, dict):
            self.r_i = input_data['r_i']
            self.t_i = input_data['t_i']
            self.p_i = input_data['p_i']
            self.rhog = input_data['rho']
        elif isinstance(input_data, str):
            with np.load(input_data) as fid:
                self.r_i = fid['r_i']
                self.t_i = fid['t_i']
                self.p_i = fid['p_i']
                self.rhog = fid['rho']
        else:
            raise ValueError(
                'input_data must be dict or path to existing file')

    @property
    def n_t(self):
        if self.t_i is None:
            return None
        return len(self.t_i) - 1

    def radmc3d_setup(self):
        """Write out the setup of radmc3d into `self.path`.

        This includes writing the setup files (`write_setup_files`) and a guess
        of the transfer fuction (`write_transfer_options`). You still need to write
        the source files, compile and run.
        """
        if any(v is None for v in [self.r_i, self.t_i, self.p_i, self.rhog]):
            raise ValueError('No data loaded. Use `read_data` first.')

        n_t = self.n_t
        #
        # NON-SCIENTIFIC RESCALING:
        #
        rhog = self.rhog / self.rhog[:,
                                     (n_t + 1) // 2, :].mean(-1)[:, None, None]
        #
        # for the NON-SCIENTIFIC OPACITIES: find limits
        #
        ir = np.argmax(rhog[:, (n_t + 1) // 2, :].max(-1) /
                       rhog[:, (n_t + 1) // 2, :].min(-1))
        vmin = rhog[ir, (n_t + 1) // 2, :].min(-1)
        vmax = rhog[ir, (n_t + 1) // 2, :].max(-1)

        mean = vmax
        sigma = 0.5 * (vmax - vmin)

        # writes out the properties for the density to source function scaling
        self.write_transfer_options(mean=mean, sigma=sigma)

        # writes out density, radmc3d.inp and other needed stuff for radmc3d
        self.write_setup_files(rhog)

    def write_setup_files(self, rhog):
        """Writes out the radmc3d input files:

            - ``gas_density.inp``
            - ``wavelength_micron.inp``
            - ``radmc3d.inp``
            - ``amr_grid.inp``
            - ``transfer.inp``

        Args:
            rhog (array): density array of shape (nr, nt, np)
            path (str | pathlib.Path): path to folder where data is written

        """
        path = Path(self.path)
        #
        # Write the wavelength file
        #

        Lambda = np.linspace(0, 20, 20)
        with open(path / 'wavelength_micron.inp', 'w') as fid:
            fid.write(str(len(Lambda)) + '\n')
            for value in Lambda:
                fid.write(f'{value:13.6e}\n')

        n_r = len(self.r_i) - 1
        n_t = len(self.t_i) - 1
        n_p = len(self.p_i) - 1
        #
        # Write the grid file
        #
        with open(path / 'amr_grid.inp', 'w') as fid:
            fid.write('1\n')                      # iformat
            # AMR grid style  (0=regular grid, no AMR)
            fid.write('0\n')
            fid.write('100\n')                    # Coordinate system
            fid.write('0\n')                      # gridinfo
            fid.write('1 1 1\n')                  # Include x,y,z coordinate
            fid.write(f'{n_r} {n_t} {n_p}\n')        # Size of grid
            for value in self.r_i:
                fid.write(f'{value:13.6e}\n')
            for value in self.t_i:
                fid.write(f'{value:13.6e}\n')
            for value in self.p_i:
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

    def write_transfer_options(self, mean=1.0, sigma=10.0):
        """Write out the parameters for the transfer function.

        Args:
            mean (float): mean of the transfer function
            sigma (float): standard deviation of the transfer function

        Returns:
            None
        """
        path = Path(self.path)

        with open(path / 'transfer.inp', 'w') as fid:
            fid.write(f'transfer_density_mean = {mean:13.6e}\n')
            fid.write(f'transfer_density_sigm = {sigma:13.6e}\n')

    def callit(self, **kwargs):
        kwargs['path'] = kwargs.get('path', self.path)
        kwargs['command'] = kwargs.get('command', None)
        kwargs['executable'] = kwargs.get('executable', './radmc3d')
        callit(**kwargs)

    def plotit(self, **kwargs):
        kwargs['path'] = kwargs.get('path', self.path)
        plotit(**kwargs)


def callit(command=None, executable='./radmc3d', path=os.curdir, verbose=0, total=None):
    """
    Run radmc3d command and show progress bar instead.

    commandd : str
        the command to run, e.g. 'mctherm'
        defaults to an image at 10 micron

    executable : str
        the command to run radmc3d (possibly with path to the executable)

    path : str
        path to the working directory where radmc3d is called

    verbose : bool
        if 0, then all output except the photon packges are shown
        if 1, just the progress is shown.
        if 2, all output is shown

    total : None | int
        total number of photon packages, if known
    """
    if command is None:
        command = 'image lambda 10 sizeradian 2 posang 180 projection 1 locobsau 0 0 .5 pointau -1 0 0 nofluxcons npix 400'

    # join executable and command
    command = f'{executable} {command}'

    # run the command
    p = subprocess.Popen(command, cwd=path, shell=True,
                         stdout=subprocess.PIPE, text=True)
    output = []

    # get total number of photon packages
    if 'nphot' in command:
        total = int(command.split('nphot')[1].split()[1])

    # show progress bar
    if verbose < 2:
        pbar = tqdm(total=total, unit='photons')

    for line in p.stdout:
        if ('Photon nr' in line) and (verbose < 2):
            pbar.update(1000)
        elif verbose:
            print(line, end='')
        output += [line]
    rc = p.wait()
    if verbose < 2:
        pbar.close()

    return rc, ''.join(output)


def plotit(path='.', log=False, **kwargs):
    im = read_image(filename=str(Path(path) / 'image.out'))
    plt.gca().clear()

    vmin = kwargs.pop('vmin', im.image.min())
    vmax = kwargs.pop('vmin', im.image.max())
    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'Reds'

    if log:
        plt.imshow(np.log10(im.image), vmin=np.log10(
            vmin), vmax=np.log10(vmax), **kwargs)
    else:
        plt.imshow(im.image, vmin=vmin, vmax=vmax, **kwargs)


def read_image(ext=None, filename=None):
    """
    Reads the rectangular telescope image produced by RADMC3D. The file name of
    the image is assumed to be image.out if no keyword is given. If keyword
    `ext` is given, the filename  'image_'+ext+'.out' is used. If keyword
    `filename` is given, it is used as the file name.

    Keywords:
    ---------

    ext : string
        Filname extension of the image file, see above

    filename : string
        file name of the image file

    Output:
    -------

    Returns a namespace containing the image data with the following attributes:
    nx,ny,nrfr,sizepix_x,sizepix_y,image,flux,x,y,lamb,radian,stokes

    The image is in erg/(s cm^2 Hz ster)

    """
    from numpy import fromfile, product, arange
    import glob
    #
    # Read from normal file, so make filename
    #
    if filename is None:
        if ext is None:
            filename = 'image.out'
        else:
            filename = 'image_' + str(ext) + '.out'
    fstr = glob.glob(str(filename))
    if len(fstr) == 0:
        print('Sorry, cannot find ' + filename)
        print('Presumably radmc3d exited without succes.')
        print('See above for possible error messages of radmc3d!')
        raise NameError('File not found')
    funit = open(filename)
    #
    # Read the image
    #
    iformat = fromfile(funit, dtype='int', count=1, sep=' ')[0]
    if iformat < 1 or iformat > 4:
        raise NameError('ERROR: File format of ' +
                        filename + ' not recognized.')
    if iformat == 1 or iformat == 3:
        radian = False
    else:
        radian = True
    if iformat == 1 or iformat == 2:
        stokes = False
    else:
        stokes = True

    nx, ny = fromfile(funit, dtype=int, count=2, sep=' ')
    nf = fromfile(funit, dtype=int, count=1, sep=' ')[0]
    sizepix_x, sizepix_y = fromfile(funit, dtype=float, count=2, sep=' ')
    lamb = fromfile(funit, dtype=float, count=nf, sep=' ')
    if nf == 1:
        lamb = lamb[0]
    if stokes:
        image_shape = [4, nx, ny, nf]
    else:
        image_shape = [nx, ny, nf]
    image = fromfile(funit, dtype=float, count=product(
        image_shape), sep=' ').reshape(image_shape, order='F')
    funit.close()
    #
    # If the image contains all four Stokes vector components,
    # then it is useful to transpose the image array such that
    # the Stokes index is the third index, so that the first
    # two indices remain x and y
    #
    if stokes:
        if nf > 1:
            image = image[[1, 2, 0, 3]]
        else:
            image = image[[1, 2, 0]]
    #
    # Compute the flux in this image as seen at 1 pc
    #
    flux = 0.0
    if stokes:
        for ix in arange(nx):
            for iy in arange(ny):
                flux = flux + image[ix, iy, 0, :]
    else:
        for ix in arange(nx):
            for iy in arange(ny):
                flux = flux + image[ix, iy, :]
    flux = flux * sizepix_x * sizepix_y
    if not radian:
        flux = flux / pc**2
    #
    # Compute the x- and y- coordinates
    #
    x = ((arange(nx) + 0.5) / (nx * 1.) - 0.5) * sizepix_x * nx
    y = ((arange(ny) + 0.5) / (ny * 1.) - 0.5) * sizepix_y * ny
    #
    # Return all
    #
    return SimpleNamespace(
        nx=nx,
        ny=ny,
        nrfr=nf,
        sizepix_x=sizepix_x,
        sizepix_y=sizepix_y,
        image=image.squeeze(),
        flux=flux,
        x=x,
        y=y,
        lamb=lamb,
        radian=radian,
        stokes=stokes)
