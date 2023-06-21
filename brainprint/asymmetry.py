"""
Contains asymmetry estimation functionality.
"""
import json
from pathlib import Path
from typing import Dict

import numpy as np
from lapy import shapedna


def compute_asymmetry(
    eigenvalues, distance: str = "euc", skip_cortex: bool = False, custom_ind: Path = None
) -> Dict[str, float]:
    """
    Computes lateral shape distances from BrainPrint analysis results.

    Parameters
    ----------
    eigenvalues : _type_
        BrainPrint analysis results
    custom_ind : Path, optional
        Custom list of indices, by default None
    distance : str, optional
        ShapeDNA distance, by default "euc"
    skip_cortex : bool, optional
        Whether to skip white matter and pial surfaces, by default False

    Returns
    -------
    Dict[str, float]
        {left_label}_{right_label}, distance
    """
    # Define structures

    # combined and individual aseg labels:
    # - Left  Striatum: left  Caudate + Putamen + Accumbens
    # - Right Striatum: right Caudate + Putamen + Accumbens
    # - CorpusCallosum: 5 subregions combined
    # - Cerebellum: brainstem + (left+right) cerebellum WM and GM
    # - Ventricles: (left+right) lat.vent + inf.lat.vent + choroidplexus + 3rdVent + CSF
    # - Lateral-Ventricle: lat.vent + inf.lat.vent + choroidplexus
    # - 3rd-Ventricle: 3rd-Ventricle + CSF

    if custom_ind:
        if custom_ind.is_file():
            with open(custom_ind, "r") as read_file:
                asymmetry_labels = json.load(read_file)
                asymmetry_labels = asymmetry_labels['symmetry']

            structures_left_right = [ (comp[0], comp[1]) for comp in asymmetry_labels ]
    else:
        structures_left_right = [
            ("Left-Striatum", "Right-Striatum"),
            ("Left-Lateral-Ventricle", "Right-Lateral-Ventricle"),
            (
                "Left-Cerebellum-White-Matter",
                "Right-Cerebellum-White-Matter",
            ),
            ("Left-Cerebellum-Cortex", "Right-Cerebellum-Cortex"),
            ("Left-Thalamus-Proper", "Right-Thalamus-Proper"),
            ("Left-Caudate", "Right-Caudate"),
            ("Left-Putamen", "Right-Putamen"),
            ("Left-Pallidum", "Right-Pallidum"),
            ("Left-Hippocampus", "Right-Hippocampus"),
            ("Left-Amygdala", "Right-Amygdala"),
            ("Left-Accumbens-area", "Right-Accumbens-area"),
            ("Left-VentralDC", "Right-VentralDC"),
        ]

    cortex_2d_left_right = [
        ("lh-white-2d", "rh-white-2d"),
        ("lh-pial-2d", "rh-pial-2d"),
    ]

    structures = structures_left_right
    if not skip_cortex:
        structures += cortex_2d_left_right

    distances = dict()
    for left_label, right_label in structures:
        left_eigenvalues, right_eigenvalues = (
            eigenvalues[left_label][2:],
            eigenvalues[right_label][2:],
        )
        has_nan = np.isnan(left_eigenvalues).any() or np.isnan(right_eigenvalues).any()
        key = f"{left_label}_{right_label}"
        if has_nan:
            message = (
                "NaNs found for {left_label} or {right_label}, "
                "skipping asymmetry computation...".format(
                    left_label=left_label, right_label=right_label
                )
            )
            print(message)
            distances[key] = np.nan
        else:
            distances[key] = shapedna.compute_distance(
                left_eigenvalues,
                right_eigenvalues,
                dist=distance,
            )
    return distances
