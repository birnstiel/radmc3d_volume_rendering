import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as c

from disklab import radmc3d as r3

au = c.au.cgs.value

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument(dest='model', choices=['sphere', 'disk'])
# args = parser.parse_args()


def main():
    """
    Load the data and call the raytracing
    """
    with np.load('fargo3d_data.npz') as fid:
        r_i  = fid['r_i'] * au
        t_i  = fid['t_i']
        p_i  = fid['p_i']
        rhog = fid['rho']

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

    set_opts(mean=mean, sigma=sigma)
    radmc3d_setup(r_i, t_i, p_i, rhog)
    r3.radmc3d('', executable='make')
    callit()
    plotit()


def radmc3d_setup(r_i, t_i, p_i, rhog):
    #
    # Write the wavelength file
    #
    Lambda = np.linspace(0, 20, 20)
    with open('wavelength_micron.inp', 'w') as fid:
        fid.write(str(len(Lambda)) + '\n')
        for value in Lambda:
            fid.write(f'{value:13.6e}\n')

    n_r  = len(r_i) - 1
    n_t  = len(t_i) - 1
    n_p  = len(p_i) - 1
    #
    # Write the grid file
    #
    with open('amr_grid.inp', 'w') as fid:
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
    with open('gas_density.inp', 'w') as fid:
        fid.write('1\n')                     # Format number
        fid.write(f'{n_r * n_t * n_p}\n')    # Nr of cells
        fid.write('1\n')                     # Format number
        data = rhog.ravel(order='F')
        data.tofile(fid, sep='\n', format='%13.6e')
        fid.write('\n')
    #
    # Write the radmc3d.inp control file
    #
    with open('radmc3d.inp', 'w') as fid:
        fid.write('incl_userdef_srcalp = 1')


def set_opts(mean=1.0, sigma=10.0):
    with open('transfer.inp', 'w') as fid:
        fid.write(f'transfer_density_mean = {mean:13.6e}\n')
        fid.write(f'transfer_density_sigm = {sigma:13.6e}\n')


def callit():
    # r3.radmc3d('image lambda 10 sizeradian 0.2  incl 45 projection 1 locobsau 5 5 5 pointau 1 0 0 nofluxcons', executable='./radmc3d')
    r3.radmc3d('image lambda 10 sizeradian 2 posang 180 projection 1 locobsau 0 0 .5 pointau -1 0 0 nofluxcons npix 400', executable='./radmc3d')


def plotit(log=False, **kwargs):
    plt.gca().clear()
    im = r3.read_image()

    vmin = kwargs.pop('vmin', im.image.min())
    vmax = kwargs.pop('vmin', im.image.max())
    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'Reds'

    if log:
        plt.imshow(np.log10(im.image.T), vmin=np.log10(vmin), vmax=np.log10(vmax), **kwargs)
    else:
        plt.imshow(im.image.T, vmin=vmin, vmax=vmax, **kwargs)


if __name__ == '__main__':
    main()
