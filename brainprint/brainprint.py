"""
Definition of the brainprint analysis execution functions.
"""

import shutil
import warnings
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
from lapy import TriaMesh, TetMesh, shapedna

from . import __version__
from .asymmetry import compute_asymmetry
from .surfaces import create_surfaces, read_vtk
from .tetra_brainprint import calc_volume_eigenmodes
from .utils.utils import (
    create_output_paths,
    export_brainprint_results,
    test_freesurfer,
    validate_environment,
    validate_subject_dir,
)

warnings.filterwarnings("ignore", ".*negative int.*")


def apply_eigenvalues_options(
    eigenvalues: np.ndarray,
    triangular_mesh: TriaMesh,
    norm: str = "none",
    reweight: bool = False,
) -> np.ndarray:
    """
    Applies BrainPrint analysis configuration options to the ShapeDNA
    eigenvalues.

    Parameters
    ----------
    eigenvalues : np.ndarray
        ShapeDNA derived eigenvalues
    triangular_mesh : TriaMesh
        Surface representation
    norm : str, optional
        Eigenvalues normalization method, by default "none"
    reweight : bool, optional
        Whether to reweight eigenvalues or not, by default False

    Returns
    -------
    np.ndarray
        Fixed eigenvalues
    """
    if not triangular_mesh.is_oriented():
        triangular_mesh.orient_()
    if norm != "none":
        eigenvalues = shapedna.normalize_ev(
            geom=triangular_mesh,
            evals=eigenvalues,
            method=norm,
        )
    if reweight:
        eigenvalues = shapedna.reweight_ev(eigenvalues)
    return eigenvalues


def compute_surface_brainprint(
    path: Path,
    return_eigenvectors: bool = True,
    num: int = 50,
    norm: str = "none",
    reweight: bool = False,
    use_cholmod: bool = False,
) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
    """
    Returns the BrainPrint eigenvalues and eigenvectors for the given surface.

    Parameters
    ----------
    path : Path
        *.vtk* surface path
    return_eigenvectors : bool, optional
        Whether to store eigenvectors or not, by default True
    num : int, optional
        Number of eigenvalues to compute, by default 50
    norm : str, optional
        Eigenvalues normalization method, by default "none"
    reweight : bool, optional
        Whether to reweight eigenvalues or not, by default False
    use_cholmod : bool, optional
        If True, attempts to use the Cholesky decomposition for improved execution
        speed. Requires the ``scikit-sparse`` library. If it can not be found, an error
        will be thrown.
        If False, will use slower LU decomposition. This is the default.

    Returns
    -------
    Tuple[np.ndarray, Union[np.ndarray, None]]
        Eigenvalues, eigenvectors (if returned)
    """
    triangular_mesh = read_vtk(path)
    shape_dna = shapedna.compute_shapedna(
        triangular_mesh,
        k=num,
        lump=False,
        aniso=None,
        aniso_smooth=10,
        use_cholmod=use_cholmod,
    )

    eigenvectors = None
    if return_eigenvectors:
        eigenvectors = shape_dna["Eigenvectors"]

    eigenvalues = shape_dna["Eigenvalues"]
    eigenvalues = apply_eigenvalues_options(
        eigenvalues, triangular_mesh, norm, reweight
    )
    eigenvalues = np.concatenate(
        (
            np.array(triangular_mesh.area(), ndmin=1),
            np.array(triangular_mesh.volume(), ndmin=1),
            eigenvalues,
        )
    )
    return eigenvalues, eigenvectors


def compute_volume_brainprint(
    label: str,
    destination: Path = None,
    num: int = 50,
    norm: str = 'none',
    norm_factor=1
):

    # subject id
    sid = str(destination).split('/')[-1:][0]

    # create output folder if doesn't exist
    Path(
        destination / "eigenvectors"
    ).mkdir(parents=True, exist_ok=True)

    tetra_file = destination / f"surfaces/aseg.final.{label}.tetra.vtk"
    nifti_input_filename = destination / f"temp/aseg.{label}.mgz"
    nifti_output_filename = destination / f"eigenvectors/{sid}.brainprint.emodes-vol.{label}.nii.gz"
    output_eval_filename = destination / f"eigenvectors/{sid}.brainprint.evals-vol.{label}.txt"
    output_emode_filename = destination / f"eigenvectors/{sid}.brainprint.emodes-vol.{label}.txt"

    eigenvalues = calc_volume_eigenmodes(
        tetra_file,
        nifti_input_filename, nifti_output_filename,
        output_eval_filename, output_emode_filename,
        num, norm, norm_factor
    )

    eigenvalues = np.concatenate(
        (
            np.array(np.nan, ndmin=1),
            np.array(np.nan, ndmin=1),
            eigenvalues,
        )
    )

    return eigenvalues

def compute_brainprint(
    surfaces: Dict[str, Path],
    destination: Path = None,
    tetrahedral: bool = False,
    keep_eigenvectors: bool = False,
    num: int = 50,
    norm: str = "none",
    reweight: bool = False,
    use_cholmod: bool = False,
) -> Tuple[Dict[str, np.ndarray], Union[Dict[str, np.ndarray], None]]:
    """
    Computes ShapeDNA descriptors over several surfaces.

    Parameters
    ----------
    surfaces : Dict[str, Path]
        Dictionary mapping from labels to *.vtk* paths
    keep_eigenvectors : bool, optional
        Whether to also return eigenvectors or not, by default False
    num : int, optional
        Number of eigenvalues to compute, by default 50
    norm : str, optional
        Eigenvalues normalization method, by default "none"
    reweight : bool, optional
        Whether to reweight eigenvalues or not, by default False
    use_cholmod : bool, optional
        If True, attempts to use the Cholesky decomposition for improved execution
        speed. Requires the ``scikit-sparse`` library. If it can not be found, an error
        will be thrown. If False, will use slower LU decomposition. This is the default.

    Returns
    -------
    Tuple[Dict[str, np.ndarray], Union[Dict[str, np.ndarray], None]]
        Surface label to eigenvalues, surface label to eigenvectors (if
        *keep_eigenvectors* is True)
    """
    eigenvalues = dict()
    eigenvectors = dict() if keep_eigenvectors else None
    for surface_label, surface_path in surfaces.items():
        if tetrahedral:
            try:
                volume_eigenvalues = compute_volume_brainprint(
                    surface_label,
                    destination=destination,
                    num=num,
                    norm=norm,
                    norm_factor=1
                )
            except Exception as e:
                message = (
                    "BrainPrint analysis raised the following exception:\n"
                    "{exception}".format(exception=e)
                )
                warnings.warn(message)
                eigenvalues[surface_label] = ["NaN"] * (num + 2)            
            else:
                if len(volume_eigenvalues) == 0:
                    eigenvalues[surface_label] = ["NaN"] * (num + 2)
                else:
                    eigenvalues[surface_label] = volume_eigenvalues

        else:
            try:
                (
                    surface_eigenvalues,
                    surface_eigenvectors,
                ) = compute_surface_brainprint(
                    surface_path,
                    num=num,
                    norm=norm,
                    reweight=reweight,
                    return_eigenvectors=keep_eigenvectors,
                    use_cholmod=use_cholmod,
                )
            except Exception as e:
                message = (
                    "BrainPrint analysis raised the following exception:\n"
                    "{exception}".format(exception=e)
                )
                warnings.warn(message)
                eigenvalues[surface_label] = ["NaN"] * (num + 2)
            else:
                if len(surface_eigenvalues) == 0:
                    eigenvalues[surface_label] = ["NaN"] * (num + 2)
                else:
                    eigenvalues[surface_label] = surface_eigenvalues
                if keep_eigenvectors:
                    eigenvectors[surface_label] = surface_eigenvectors

    return eigenvalues, None if tetrahedral else eigenvectors


def run_brainprint(
    subjects_dir: Path,
    subject_id: str,
    destination: Path = None,
    custom_seg: Path = None,
    custom_ind: Path = None,
    num: int = 50,
    skip_cortex: bool = False,
    tetrahedral: bool = False,
    smooth: bool = False,
    ncpu: int = 1,
    keep_eigenvectors: bool = False,
    norm: str = "none",
    reweight: bool = False,
    asymmetry: bool = False,
    asymmetry_distance: str = "euc",
    keep_temp: bool = False,
    use_cholmod: bool = False,
):
    """
    Runs the BrainPrint analysis.

    Parameters
    ----------
    subjects_dir : Path
        FreeSurfer's subjects directory
    subject_id : str
        The subject identifier, as defined within the FreeSurfer's subjects
        directory
    destination : Path, optional
        If provided, will use this path as the results root directory, by
        default None
    custom_seg : Path, optional
        If provided, will use this segmentation for subcortical surfaces, by
        default None
    custom_ind : Path, optional
        A custom list of indices should provided in case a custom segmentation
        is used, by default None
    num : int, optional
        Number of eigenvalues to compute, by default 50
    norm : str, optional
        Eigenvalues normalization method, by default "none"
    reweight : bool, optional
        Whether to reweight eigenvalues or not, by default False
    smooth : bool, optional
        Whether to smooth subcortical surfaces or not, by default False
    ncpu : int, optional
        Number of CPUs to use during construction of surfaces, by default 1
    skip_cortex : bool, optional
        _description_, by default False
    tetrahedral : bool, optional
        Whether to use tetrahedral surfaces, by default False
    keep_eigenvectors : bool, optional
        Whether to also return eigenvectors or not, by default False
    asymmetry : bool, optional
        Whether to calculate asymmetry between lateral structures, by default
        False
    asymmetry_distance : str, optional
        Distance measurement to use if *asymmetry* is set to True, by default
        "euc"
    keep_temp : bool, optional
        Whether to keep the temporary files directory or not, by default False
    use_cholmod : bool, optional
        If True, attempts to use the Cholesky decomposition for improved execution
        speed. Requires the ``scikit-sparse`` library. If it can not be found, an error
        will be thrown. If False, will use slower LU decomposition. This is the default.

    Returns
    -------
    Tuple[Dict[str, np.ndarray], Union[Dict[str, np.ndarray], None],
    Union[Dict[str, float], None]]
        Eigenvalues, eigenvectors, distances
    """
    validate_environment()
    test_freesurfer()
    subject_dir = validate_subject_dir(subjects_dir, subject_id)
    destination = create_output_paths(
        subject_dir=subject_dir,
        destination=destination,
    )

    # surfaces = create_surfaces(
    #     subject_dir, destination, custom_seg=custom_seg, custom_ind=custom_ind,
    #     skip_cortex=skip_cortex, smooth=smooth, ncpu=ncpu
    # )

    surfaces = {
        'Left-ThalamusWhole': Path('derivatives/brainprint_tetra/sub-C030/surfaces/aseg.final.Left-ThalamusWhole.vtk'),
        'Right-ThalamusWhole': Path('derivatives/brainprint_tetra/sub-C030/surfaces/aseg.final.Right-ThalamusWhole.vtk')
    }

    # Test implementation of volume-based eigenmodes
    eigenvalues, eigenvectors = compute_brainprint(
        surfaces,
        destination=destination,
        tetrahedral=tetrahedral,
        num=num,
        norm=norm,
        reweight=reweight,
        keep_eigenvectors=keep_eigenvectors,
        use_cholmod=use_cholmod,
    )

    distances = None
    if asymmetry:
        distances = compute_asymmetry(
            eigenvalues,
            distance=asymmetry_distance,
            skip_cortex=skip_cortex,
            custom_ind=custom_ind
        )

    csv_name = "{subject_id}.brainprint.csv".format(subject_id=subject_id)
    csv_path = destination / csv_name

    export_brainprint_results(csv_path, eigenvalues, eigenvectors, distances)
    print("Returning matrices for eigenvalues, eigenvectors, and (optionally) distances.")
    print("The eigenvalue matrix contains area and volume as first two rows.")

    if not keep_temp:
        shutil.rmtree(destination / "temp")    

    # return eigenvalues, eigenvectors, distances
    return eigenvalues, eigenvectors, distances


class Brainprint:
    __version__ = __version__

    def __init__(
        self,
        subjects_dir: Path,
        num: int = 50,
        custom_seg: Path = None,
        custom_ind: Path = None,
        skip_cortex: bool = False,
        tetrahedral: bool = False,
        smooth: bool = False,
        ncpu: int = 1,
        keep_eigenvectors: bool = False,
        norm: str = "none",
        reweight: bool = False,
        asymmetry: bool = False,
        asymmetry_distance: str = "euc",
        keep_temp: bool = False,
        environment_validation: bool = True,
        freesurfer_validation: bool = True,
        use_cholmod: bool = False,
    ) -> None:
        """
        Initializes a new :class:`Brainprint` instance.

        Parameters
        ----------
        subjects_dir : Path
            FreeSurfer's subjects directory
        num : int, optional
            Number of eigenvalues to compute, by default 50
        norm : str, optional
            Eigenvalues normalization method, by default "none"
        reweight : bool, optional
            Whether to reweight eigenvalues or not, by default False
        custom_seg : Path, optional
            If provided, will use this segmentation for subcortical surfaces, by
            default None
        custom_ind : Path, optional
            A custom list of indices should provided in case a custom segmentation
            is used, by default None            
        skip_cortex : bool, optional
            _description_, by default False
        smooth : bool, optional
            Whether to smooth subcortical surfaces or not, by default False
        ncpu : int, optional
            Number of CPUs to use during construction of surfaces, by default 1                        
        keep_eigenvectors : bool, optional
            Whether to also return eigenvectors or not, by default False
        asymmetry : bool, optional
            Whether to calculate asymmetry between lateral structures, by
            default False
        asymmetry_distance : str, optional
            Distance measurement to use if *asymmetry* is set to True, by
            default "euc"
        keep_temp : bool, optional
            Whether to keep the temporary files directory or not, by default False
        use_cholmod : bool, optional
            If True, attempts to use the Cholesky decomposition for improved execution
            speed. Requires the ``scikit-sparse`` library. If it can not be found, an
            error will be thrown. If False, will use slower LU decomposition. This is
            the default.
        """
        self.subjects_dir = subjects_dir
        self.num = num
        self.norm = norm
        self.custom_seg = custom_seg
        self.custom_ind = custom_ind
        self.skip_cortex = skip_cortex
        self.tetrahedral = tetrahedral
        self.smooth = smooth
        self.ncpu = ncpu
        self.reweight = reweight
        self.keep_eigenvectors = keep_eigenvectors
        self.asymmetry = asymmetry
        self.asymmetry_distance = asymmetry_distance
        self.keep_temp = keep_temp
        self.use_cholmod = use_cholmod

        self._subject_id = None
        self._destination = None
        self._eigenvalues = None
        self._eigenvectors = None
        self._distances = None

        if environment_validation:
            validate_environment()
        if freesurfer_validation:
            test_freesurfer()

    def run(self, subject_id: str, destination: Path = None) -> Dict[str, Path]:
        self._eigenvalues = self._eigenvectors = self._distances = None
        subject_dir = validate_subject_dir(self.subjects_dir, subject_id)
        destination = create_output_paths(
            subject_dir=subject_dir,
            destination=destination,
        )

        surfaces = create_surfaces(
            subject_dir, destination, custom_seg=self.custom_seg, custom_ind=self.custom_ind,
            skip_cortex=self.skip_cortex, smooth=self.smooth, ncpu=self.ncpu
        )

        self._eigenvalues, self._eigenvectors = compute_brainprint(
            surfaces,
            tetrahedral=self.tetrahedral,
            num=self.num,
            norm=self.norm,
            reweight=self.reweight,
            keep_eigenvectors=self.keep_eigenvectors,
            use_cholmod=self.use_cholmod,
        )

        if self.asymmetry:
            self._distances = compute_asymmetry(
                self._eigenvalues,
                distance=self.asymmetry_distance,
                skip_cortex=self.skip_cortex,
                custom_ind=self.custom_ind
            )

        self.cleanup(destination=destination)
        return self.export_results(destination=destination, subject_id=subject_id)

    def export_results(self, destination: Path, subject_id: str) -> None:
        csv_name = "{subject_id}.brainprint.csv".format(subject_id=subject_id)
        csv_path = destination / csv_name
        return export_brainprint_results(
            csv_path, self._eigenvalues, self._eigenvectors, self._distances
        )

    def cleanup(self, destination: Path) -> None:
        if not self.keep_temp:
            shutil.rmtree(destination / "temp")
