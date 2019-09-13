[![Documentation Status](https://readthedocs.org/projects/dsharp-helper/badge/?version=latest)](https://dsharp-helper.readthedocs.io/en/latest/?badge=latest)

# DSHARP Helper

Helps to downloading, accessing, and plotting of DSHARP data. The same regulations as in the [official data release](https://almascience.eso.org/almadata/lp/DSHARP/) apply with regards to citations.

Plotting the continuum, the radial profile of the continuum, and the SED of a disk is as easy as:

    import dsharp_helper as dh

    disk = 'IM Lup'

    dh.plot_DHSARP_continuum(disk=disk)
    dh.plot_profile(disk)
    dh.plot_sed(disk)

<img width="30%" src=docs/source/notebooks/IMLup_cont.png>
<img width="30%" src=docs/source/notebooks/IMLup_prof.png>
<img width="30%" src=docs/source/notebooks/IMLup_sed.png>

<br />

Furthermore, the information on the sources is available as `dh.sources`. See also the [notebooks folder](docs/source/notebooks/).
