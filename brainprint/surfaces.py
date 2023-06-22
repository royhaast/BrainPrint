"""
Utility module holding surface generation related functions.
"""

import os
import uuid
import json
import subprocess
from pathlib import Path
from typing import Dict, List

from joblib import Parallel, delayed, dump, load
from lapy import TriaMesh

from .utils.utils import run_shell_command

def create_aseg_surface(
    subject_dir: Path, destination: Path, label: str, indices: List[int],
    custom_seg: Path = None, smooth: bool = False, ncpu: int = 1
) -> Path:
    """
    Creates a surface from the aseg and label files.
    """

    # Use custom label file if specified
    if custom_seg.is_file():
        aseg_path = custom_seg
    else:
        aseg_path = subject_dir / "mri/aseg.mgz"

    norm_path = subject_dir / "mri/norm.mgz"
    temp_name = "temp/aseg.{label}".format(label=label)
    indices_mask = destination / f"{temp_name}.mgz"
    # binarize on selected labels (creates temp indices_mask)
    # always binarize first, otherwise pretess may scale aseg if labels are
    # larger than 255 (e.g. aseg+aparc, bug in mri_pretess?)
    binarize_template = "mri_binarize --i {source} --match {match} --o {destination}"
    binarize_command = binarize_template.format(
        source=aseg_path, match=" ".join(indices), destination=indices_mask
    )
    run_shell_command(binarize_command)

    label_value = "1"
    # if norm exist, fix label (pretess)
    if norm_path.is_file():
        pretess_template = (
            "mri_pretess {source} {label_value} {norm_path} {destination}"
        )
        pretess_command = pretess_template.format(
            source=indices_mask,
            label_value=label_value,
            norm_path=norm_path,
            destination=indices_mask,
        )
        run_shell_command(pretess_command)

    # runs marching cube to extract surface
    surface_name = "{name}.surf".format(name=temp_name)
    surface_path = destination / surface_name
    extraction_template = "mri_mc {source} {label_value} {destination}"
    extraction_command = extraction_template.format(
        source=indices_mask, label_value=label_value, destination=surface_path
    )
    run_shell_command(extraction_command)

    if smooth:
        # optional smoothing
        smoothing_template = "mris_smooth -nw {source} {destination}"
        smoothing_command = smoothing_template.format(
            source=surface_path, destination=surface_path
        )

        run_shell_command(smoothing_command)

    # convert to gifti
    relative_path = f"surfaces/aseg.final.{label}.surf.gii"
    conversion_destination = destination / relative_path
    conversion_template = "mris_convert --to-scanner {source} {destination}"
    conversion_command = conversion_template.format(
        source=surface_path, destination=conversion_destination
    )
    run_shell_command(conversion_command)   

    # convert to vtk
    relative_path = f"surfaces/aseg.final.{label}.vtk"
    conversion_destination = destination / relative_path
    conversion_template = "mris_convert --to-scanner {source} {destination}"
    conversion_command = conversion_template.format(
        source=surface_path, destination=conversion_destination
    )
    run_shell_command(conversion_command) 

    # create tetrahedral surface
    tria_file = conversion_destination
    geo_file = os.path.splitext(conversion_destination)[0] + '.geo'
    tetra_file = os.path.splitext(conversion_destination)[0] + '.tetra.vtk'

    file = str(tria_file).rsplit('/')
    inputGeo = file[len(file)-1]
    
    with open(geo_file, 'w') as writer:
        writer.write('Mesh.Algorithm3D=4;\n')
        writer.write('Mesh.Optimize=1;\n')
        writer.write('Mesh.OptimizeNetgen=1;\n')
        writer.write('Merge "'+inputGeo+'";\n')
        writer.write('Surface Loop(1) = {1};\n')
        writer.write('Volume(1) = {1};\n')
        writer.write('Physical Volume(1) = {1};\n')

    cmd = 'gmsh -3 -o ' + tetra_file + ' ' + geo_file
    output = subprocess.check_output(cmd, shell="True")
    output = output.splitlines()

    cmd = "sed 's/double/float/g;s/UNSTRUCTURED_GRID/POLYDATA/g;s/CELLS/POLYGONS/g;/CELL_TYPES/,$d' " + tetra_file + " > " + tetra_file + "'_fixed'"
    os.system(cmd)
    os.system('mv -f ' + tetra_file + '_fixed ' + tetra_file)

    print("Finished generating surfaces")

    if ncpu != 1:
        return {
            label: conversion_destination
        }
    else:
        return conversion_destination


def create_aseg_surfaces(
    subject_dir: Path, destination: Path, custom_seg: Path = None, 
    custom_ind: Path = None, smooth: bool = False, ncpu: int = 1
) -> Dict[str, Path]:
    # Define aseg labels

    # combined and individual aseg labels:
    # - Left  Striatum: left  Caudate + Putamen + Accumbens
    # - Right Striatum: right Caudate + Putamen + Accumbens
    # - CorpusCallosum: 5 subregions combined
    # - Cerebellum: brainstem + (left+right) cerebellum WM and GM
    # - Ventricles: (left+right) lat.vent + inf.lat.vent + choroidplexus + 3rdVent + CSF
    # - Lateral-Ventricle: lat.vent + inf.lat.vent + choroidplexus
    # - 3rd-Ventricle: 3rd-Ventricle + CSF

    if custom_seg.is_file():
        if custom_ind is None:
            message = "Please provide a list of indices when using a custom segmentation"
            raise RuntimeError(message)
        elif custom_ind.is_file():
            with open(custom_ind, "r") as read_file:
                aseg_labels = json.load(read_file)
                aseg_labels = aseg_labels['indices'][0]
    else:
        aseg_labels = {
            "CorpusCallosum": ["251", "252", "253", "254", "255"],
            "Cerebellum": ["7", "8", "16", "46", "47"],
            "Ventricles": ["4", "5", "14", "24", "31", "43", "44", "63"],
            "3rd-Ventricle": ["14", "24"],
            "4th-Ventricle": ["15"],
            "Brain-Stem": ["16"],
            "Left-Striatum": ["11", "12", "26"],
            "Left-Lateral-Ventricle": ["4", "5", "31"],
            "Left-Cerebellum-White-Matter": ["7"],
            "Left-Cerebellum-Cortex": ["8"],
            "Left-Thalamus-Proper": ["10"],
            "Left-Caudate": ["11"],
            "Left-Putamen": ["12"],
            "Left-Pallidum": ["13"],
            "Left-Hippocampus": ["17"],
            "Left-Amygdala": ["18"],
            "Left-Accumbens-area": ["26"],
            "Left-VentralDC": ["28"],
            "Right-Striatum": ["50", "51", "58"],
            "Right-Lateral-Ventricle": ["43", "44", "63"],
            "Right-Cerebellum-White-Matter": ["46"],
            "Right-Cerebellum-Cortex": ["47"],
            "Right-Thalamus-Proper": ["49"],
            "Right-Caudate": ["50"],
            "Right-Putamen": ["51"],
            "Right-Pallidum": ["52"],
            "Right-Hippocampus": ["53"],
            "Right-Amygdala": ["54"],
            "Right-Accumbens-area": ["58"],
            "Right-VentralDC": ["60"],
        }

    if ncpu != 1:
        surfaces = Parallel(n_jobs=ncpu)(
            delayed(create_aseg_surface)(
                subject_dir, destination, label, indices, custom_seg, smooth, ncpu
            ) for label, indices in aseg_labels.items()
        )   

        return {
            label: conversion_destination for i in surfaces
            for label, conversion_destination in i.items()
        }
    else:
        return {
            label: create_aseg_surface(subject_dir, destination, indices, custom_seg, smooth, ncpu)
            for label, indices in aseg_labels.items()
        }        


def create_cortical_surfaces(subject_dir: Path, destination: Path) -> Dict[str, Path]:
    cortical_labels = {
        "lh-white-2d": "lh.white",
        "rh-white-2d": "rh.white",
        "lh-pial-2d": "lh.pial",
        "rh-pial-2d": "rh.pial",
    }
    return {
        label: surf_to_vtk(
            subject_dir / "surf" / name,
            destination / "surfaces" / f"{name}.vtk",
        )
        for label, name in cortical_labels.items()
    }


def create_surfaces(
    subject_dir: Path, destination: Path, custom_seg: Path, custom_ind: Path,
    skip_cortex: bool = False, smooth: bool = False, ncpu: int = 1
) -> Dict[str, Path]:
    surfaces = create_aseg_surfaces(
        subject_dir, destination, custom_seg, custom_ind, smooth, ncpu
    )
    if not skip_cortex:
        cortical_surfaces = create_cortical_surfaces(subject_dir, destination)
        surfaces.update(cortical_surfaces)
    return surfaces


def read_vtk(path: Path):
    try:
        triangular_mesh = TriaMesh.read_vtk(path)
    except Exception:
        message = "Failed to read VTK from the following path: {path}!".format(
            path=path
        )
        raise RuntimeError(message)
    else:
        if triangular_mesh is None:
            message = "Failed to read VTK from the following path: {path}!".format(
                path=path
            )
            raise RuntimeError(message)
        return triangular_mesh


def surf_to_vtk(source: Path, destination: Path) -> Path:
    """
    Converted a FreeSurfer *.surf* file to *.vtk*.

    Parameters
    ----------
    source : Path
        FreeSurfer *.surf* file
    destination : Path
        Equivalent *.vtk* file

    Returns
    -------
    Path
        Resulting *.vtk* file
    """
    TriaMesh.read_fssurf(source).write_vtk(destination)
    return destination
