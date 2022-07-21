"""
pyarpes plugin for data files from the PEARL beamline (Swiss Light Source).

provides PearlEndstation class.

coordinate transforms
---------------------

according to the [pyarpes documentation](https://arpes.readthedocs.io/en/latest/spectra.html):

| PEARL | pyarpes |
|:-----:|:-------:|
|   x   |    z    |
|   y   |    x    |
|   z   |    y    |
| theta |  theta  |
| tilt  |  beta   |
|  phi  |   chi   |
| alpha |   phi   |
|   0   |   psi   |
|  -90° |  alpha  |

"""

import functools
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import h5py
import numpy as np
import xarray as xr

from arpes.endstations import (
    SingleFileEndstation,
    HemisphericalEndstation,
    SynchrotronEndstation,
)

__all__ = ("PearlEndstation",)


class PearlEndstation(SingleFileEndstation, SynchrotronEndstation, HemisphericalEndstation):
    """
    pyarpes plugin for data files from the PEARL beamline (Swiss Light Source)

    usage
    -----

    ~~~~~~{.py}
    from arpes.io import load_data
    from arpes.endstations.plugin.pearl import PearlEndstation
    load_data("pshell-20220223-144834-ManipulatorScan.h5", location=PearlEndstation,
              scan=1, region=1, dataset='ScientaImage')
    ~~~~~~

    the `scan`, `region` and `dataset` arguments are optional. the values shown above are the default values.
    """

    PRINCIPAL_NAME = 'SLS-PEARL'
    ALIASES = ['PEARL', 'X03DA', 'SLS-X03DA']

    _TOLERATED_EXTENSIONS = {'.h5'}
    _SEARCH_PATTERNS = [
        # regex matching names like
        # "data_Conrad_4.pxt" and "data_Oct19_1.pxt"
        #
        # the file number is injected into the `{}` pattern.
        r'pshell-[0-9]+-[0-9]+-.+'

        # You can provide as many as you need.
    ]

    RENAME_KEYS = {
        "ManipulatorTheta": "theta",
        "ManipulatorTilt": "beta",
        "ManipulatorPhi": "chi",
        "ManipulatorX": "z",
        "ManipulatorY": "x",
        "ManipulatorZ": "y",
        "ManipulatorTempA": "temperature_cryotip",
        "ManipulatorTempB": "temperature",

        "ScientaSlices": "phi",
        "ScientaChannels": "eV",
        "ScientaCenterEnergy": "daq_center_energy",
        "ScientaLowEnergy": "sweep_low_energy",
        "ScientaHighEnergy": "sweep_high_energy",
        "StepSize": "sweep_step",
        "NumIterations": "n_sweeps",
        "LensMode": "lens_mode",
        "PassEnergy": "pass_energy",
        # "AcquisitionMode": "acquisition_mode",
        "ScientaDwellTime": "dwell_time",
        "AnalyserSlit": "slit",
        "RegionName": "fixed_region_name",

        "MonoEnergy": "hv",
        "MonoGrating": "grating_lines_per_mm",
        "ExitSlit": "exit_slit",

        "RingCurrent": "beam_current",
        "RefCurrent": "photon_flux",
        "SampleCurrent": "photocurrent",
        "ChamberPressure": "pressure",
    }

    TRANSFORM_FUNCS = {
        "ManipulatorTheta": np.deg2rad,
        "ManipulatorTilt": np.deg2rad,
        "ManipulatorPhi": np.deg2rad,
        "ScientaSlices": np.deg2rad,
        "SampleCurrent": lambda x: x * 1e9,
        "RefCurrent": lambda x: x * 1e9,
        "RingCurrent": lambda x: x * 1e6,
        # "MonoGrating": lambda x: int(str(x).split("_")[1])
        }
    
    ATTR_TRANSFORMS = {
        "acquisition_mode": lambda l: l.lower(),
        "lens_mode": lambda l: {
            "lens_mode": None,
            "lens_mode_name": l,
        },
        "region_name": lambda l: {
            "daq_region_name": l,
            "daq_region": l,
        },
    }

    RESOLUTION_TABLE = None

    ANALYZER_INFORMATION = {
        "analyzer": "EW4000",
        "analyzer_name": "Scienta EW4000",
        "parallel_deflectors": False,
        "perpendicular_deflectors": True,
        "analyzer_radius": 200,
        "analyzer_type": "hemispherical",
    }
    SLIT_ORIENTATION = "vertical"

    def load_single_frame(self, frame_path: str = None, scan_desc: dict = None,
                          scan: int = 1, region: int = 1, dataset: str = None, **kwargs):
        """
        load a single dataset from a PEARL pshell file

        Parameters
        ----------

        frame_path: path of the pshell.h5 file

        scan_desc: not used

        scan: scan number, default = 1

        region: region number, default = 1 (or missing group level)

        dataset: name of dataset to load, must be one of
                 'ScientaImage', 'ScientaSpectrum', 'ScientaAngleDistribution', 'ScientaEnergyDistribution'.
                 the default is ScientaImage if present, else ScientaSpectrum.
        """

        dataset_name = dataset
        h5 = h5py.File(frame_path, "r")
        scan_group = self.find_scan_group(h5, scan)
        region_group = self.find_region_group(scan_group, region)
        attrs_group = self.find_attrs_group(scan_group)

        # find dataset
        if not dataset_name:
            dataset_name = 'ScientaImage' if 'ScientaImage' in region_group.keys() else 'ScientaSpectrum'
        dataset = self.read_h5_dataset(region_group[dataset_name])

        # metadata
        attrs = {**self.ANALYZER_INFORMATION}
        self.read_general_metadata(h5, attrs)

        # coordinates
        coords = {}
        dims = []
        for _w in scan_group.attrs['Writables']:
            writable_name = _w.decode()
            original_data = self.read_h5_dataset(scan_group[writable_name])
            coords_name, coords_data = self.transform_coords(writable_name, original_data)
            dims.append(coords_name)
            coords[coords_name] = coords_data

        self.fix_scienta_coords(dataset, dataset_name, attrs_group, coords, dims)

        attrs['psi'] = 0.
        attrs["chi_offset"] = 0.
        attrs['alpha'] = np.deg2rad(-90)
        attrs['scan'] = scan
        attrs['region'] = region

        for attr in attrs_group.keys():
            original_data = self.read_h5_dataset(attrs_group[attr])
            attrs[attr] = original_data
            transformed_name, transformed_data = self.transform_coords(attr, original_data)
            attrs[transformed_name] = transformed_data

        # important fixed coordinates
        for c in self.ENSURE_COORDS_EXIST:
            if c not in dims:
                if c in attrs:
                    coords[c] = np.mean(attrs[c])

        xdata = xr.DataArray(dataset, coords=coords, dims=dims, name=dataset_name, attrs=attrs)
        xdataset = xr.Dataset({"spectrum": xdata}, attrs=xdata.attrs)
        return xdataset

    @staticmethod
    def read_general_metadata(h5: h5py.File, attrs: Dict[str, Any]):
        """
        read metadata from the general group of a pshell HDF5 file.
        """

        try:
            general = h5['general']
        except KeyError:
            pass
        else:
            try:
                attrs['experimenter'] = ", ".join([v.decode() for v in general['authors']])
            except KeyError:
                pass
            try:
                attrs['pgroup'] = general['pgroup'][()].decode()
            except KeyError:
                pass
            try:
                attrs['sample_name'] = general['sample'][()].decode()
            except KeyError:
                pass

    @staticmethod
    def fix_scienta_coords(dataset: np.array, dataset_name: str, attrs_group: h5py.Group,
                           coords: Dict[str, np.array], dims: List[str]):
        """
        attach channels and/or slices scales to Scienta dataset
        """

        channels = functools.partial(np.linspace,
                                     attrs_group['ScientaChannelBegin'][0],
                                     attrs_group['ScientaChannelEnd'][0])
        slices = functools.partial(np.linspace,
                                   attrs_group['ScientaSliceBegin'][0],
                                   attrs_group['ScientaSliceEnd'][0])

        if dataset_name == 'ScientaSpectrum':
            # dims = ['scan', 'energy']
            if 'eV' not in dims:
                dims.append('eV')
                coords['eV'] = channels(dataset.shape[-1])

        elif dataset_name == 'ScientaImage':
            # dims = ['angle', 'energy', 'scan']
            if 'eV' not in dims:
                dims.insert(0, 'eV')
                coords['eV'] = channels(dataset.shape[1])
            if 'phi' not in dims:
                dims.insert(0, 'phi')
                coords['phi'] = slices(dataset.shape[0])

        elif dataset_name == 'ScientaAngleDistribution':
            if 'phi' not in dims:
                dims.append('phi')
                coords['phi'] = slices(dataset.shape[-1])

        elif dataset_name == 'ScientaEnergyDistribution':
            if 'eV' not in dims:
                dims.append('eV')
                coords['eV'] = channels(dataset.shape[-1])

    @staticmethod
    def find_scan_group(h5: h5py.File, scan_number: int = 1) -> h5py.Group:
        """
        find scan group in pshell hdf5 file.
        """

        scan_pattern = r"scan ?{}".format(scan_number)
        for k in h5.keys():
            if re.match(scan_pattern, k, re.IGNORECASE):
                scan_group = h5[k]
                break
        else:
            raise ValueError(f"can't find scan group {scan_number}")
        return scan_group

    @staticmethod
    def find_region_group(scan_group: h5py.Group, region_number: int = 1) -> h5py.Group:
        """
        find region group

        the region group can be identical to scan group if it is not explicit in the file.
        """

        region_pattern = r"region ?{}".format(region_number)
        for k in scan_group.keys():
            if re.match(region_pattern, k, re.IGNORECASE):
                region_group = scan_group[k]
                break
        else:
            if region_number == 1:
                region_group = scan_group
            else:
                raise ValueError(f"can't find scan region {region_number} in file {scan_group.file.filename}")
        return region_group

    @staticmethod
    def find_attrs_group(scan_group):
        """
        find beamline attributes group

        depending on the file version, this can be named `attr`, `attrs`, `diags` or `snaps`.
        """
        attrs_groups = ['attr', 'attrs', 'diags', 'snaps']
        for k in attrs_groups:
            if k in scan_group.keys():
                attrs_group = scan_group[k]
                break
        else:
            raise ValueError(f"can't find attrs group in file {scan_group.file.filename}")
        return attrs_group

    @staticmethod
    def read_h5_dataset(dataset: h5py.Dataset) -> np.array:
        """
        read data from h5py dataset into numpy array

        to actually get the data from the file, we have to slice it.
        if the dataset contains strings, we first have to decode them.
        this function handles both cases and returns a plain numpy array of numeric or (string) object type.
        """
        if np.issubdtype(dataset.dtype, np.number):
            return dataset[:]
        else:
            return dataset.asstr()[:]

    def transform_coords(self, dataset_name: str, dataset: np.array) -> Tuple[str, np.array]:
        """
        transform coordinates to pyarpes units and names
        

        Parameters
        ----------
        dataset_name : str
            PEARL-name of dataset/channel.
        dataset : Array-like
            Dataset as numpy array or h5py-proxy.

        Returns
        -------
        transformed_name : TYPE
            pyarpes-name.
        transformed_dataset : TYPE
            dataset in pyarpes standard units.

        """

        try:
            transformed_name = self.RENAME_KEYS[dataset_name]
        except KeyError:
            transformed_name = dataset_name

        try:
            f = self.TRANSFORM_FUNCS[dataset_name]
        except KeyError:
            transformed_dataset = dataset[:]
        else:
            transformed_dataset = f(dataset[:])

        return transformed_name, transformed_dataset
