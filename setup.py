from setuptools import setup
from setuptools import find_packages
from six import PY2

# arguments for setup which are used for python2 and beyond
kw23 = {"name": 'S2MSI',
        "version": '2016.01.26.6',
        "url": '',
        "license": 'Creative Commons Attribution-NonCommercial 4.0 International Public License',
        "author": 'Andre Hollstein',
        "author_email": 'andre.hollstein@gfz-potsdam.de',
        "description": '',
        }

if PY2 is False:  # not python2
    setup(install_requires=['numpy',
                            'dill',
                            'glymur',
                            'scipy',
                            'gdal',
                            'pygrib',
                            'matplotlib',
                            'scikit-image',
                            'psutil',
                            'pillow',
                            'glymur',
                            'numba',
                            ],
          packages=['S2MSI',
                    'S2MSI.GranuleInfo',
                    'S2MSI.GranuleDEM',
                    'S2MSI.S2Mask',
                    'S2MSI.S2Image',
                    'S2MSI.cB',
                    'S2MSI.Tools',
                    'S2MSI.GrassTools',
                    ],
          data_files=[("data", ["S2MSI/GranuleInfo/data/S2_tile_data_lite.json",
                                "S2MSI/GranuleInfo/data/S2_tile_data_full.json",
                                "S2MSI/cB/data/cld_mask_20160321_s2.h5",
                                "S2MSI/Tools/data/Sentinel-2A MSI Spectral Responses.xlsx",
                                ])],
          **kw23
          )
else:  # for python2
    # for python 2.7, only grass scripts are installed # scripts=['S2MSI/GrassTools2/dem_shadow_maps.py']
    setup(install_requires=['numpy', 'matplotlib', 'scipy',
                            ],
          scripts=['S2MSI/GrassTools2/dem_shadow_maps.py'],
          packages=['S2MSI',
                    'S2MSI.GrassTools2',
                    ],
          data_files=[],
          **kw23)
