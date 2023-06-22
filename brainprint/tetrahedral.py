import os
import numpy as np
import nibabel as nib
from lapy import TetMesh, shapedna
from lapy.solver import Solver
from scipy.interpolate import griddata

def calc_volume(nifti_input_filename):
    """Calculate the physical volume of the ROI in the nifti file.

    Parameters
    ----------
    nifti_input_filename : str
        Filename of input volume where the relevant ROI have voxel values = 1

    Returns
    ------
    ROI_number : int
        Total number of non-zero voxels
    ROI_volume : float
        Total volume of non-zero voxels in physical dimensions   
    """

    # Load data
    ROI_data = nib.load(nifti_input_filename)
    roi_data = ROI_data.get_fdata()

    # Get voxel dimensions in mm
    voxel_dims = np.array(ROI_data.header.get_zooms())
    voxel_vol = np.prod(voxel_dims)

    # Compute volume
    ROI_number = np.count_nonzero(roi_data)
    ROI_volume = ROI_number * voxel_vol

    return ROI_number, ROI_volume

def normalize_vtk(
    tet, nifti_input_filename, normalization_type='none', normalization_factor=1
):
    """Normalize tetrahedral surface.

    Parameters
    ----------
    tet : lapy compatible object
        Loaded vtk object corresponding to a surface tetrahedral mesh
    nifti_input_filename : str
        Filename of input volume where the relevant ROI have voxel values = 1
    normalization_type : str (default: 'none')
        Type of normalization
        number - normalization with respect to the total number of non-zero voxels
        volume - normalization with respect to the total volume of non-zero voxels in physical dimensions   
        constant - normalization with respect to a chosen constant
        others - no normalization
    normalization_factor : float (default: 1)
        Factor to be used in a constant normalization     

    Returns
    ------
    tet_norm : lapy compatible object
        Loaded vtk object corresponding to the normalized surface tetrahedral mesh
    """

    nifti_input_file_head, nifti_input_file_tail = os.path.split(nifti_input_filename)
    nifti_input_file_main, nifti_input_file_ext = os.path.splitext(nifti_input_file_tail)

    ROI_number, ROI_volume = calc_volume(nifti_input_filename)

    # normalization process
    tet_norm = tet
    if normalization_type == 'number':
        tet_norm.v = tet.v/(ROI_number**(1/3))
    elif normalization_type == 'volume':
        tet_norm.v = tet.v/(ROI_volume**(1/3))
    elif normalization_type == 'constant':
        tet_norm.v = tet.v/(normalization_factor**(1/3))
    else:
        pass

    # writing normalized surface to a vtk file
    if normalization_type == 'number' or normalization_type == 'volume' or normalization_type == 'constant':
        surface_output_filename = nifti_input_filename + '_norm=' + normalization_type + '.tetra.vtk'

        f = open(surface_output_filename, 'w')
        f.write('# vtk DataFile Version 2.0\n')
        f.write(nifti_input_file_tail + '\n')
        f.write('ASCII\n')
        f.write('DATASET POLYDATA\n')
        f.write('POINTS ' + str(np.shape(tet.v)[0]) + ' float\n')
        for i in range(np.shape(tet.v)[0]):
            f.write(' '.join(map(str, tet_norm.v[i, :])))
            f.write('\n')
        f.write('\n')
        f.write('POLYGONS ' + str(np.shape(tet.t)[0]) + ' ' + str(5 * np.shape(tet.t)[0]) + '\n')
        for i in range(np.shape(tet.t)[0]):
            f.write(' '.join(map(str, np.append(4, tet.t[i, :]))))
            f.write('\n')
        f.close()

    return tet_norm

def calc_eig(
    tetra_file, nifti_input_filename, output_eval_filename, output_emode_filename, num_modes,
    normalization_type='none', normalization_factor=1
):
    """Calculate the eigenvalues and eigenmodes of the ROI volume in a nifti file.

    Parameters
    ----------
    nifti_input_filename : str
        Filename of input volume where the relevant ROI have voxel values = 1
    output_eval_filename : str  
        Filename of text file where the output eigenvalues will be stored
    output_emode_filename : str  
        Filename of text file where the output eigenmodes (in tetrahedral surface space) will be stored    
    num_modes : int
        Number of eigenmodes to be calculated
    normalization_type : str (default: 'none')
        Type of normalization
        number - normalization with respect to the total number of non-zero voxels
        volume - normalization with respect to the total volume of non-zero voxels in physical dimensions   
        constant - normalization with respect to a chosen constant
        others - no normalization
    normalization_factor : float (default: 1)
        Factor to be used in a constant normalization      

    Returns
    ------
    evals: array (num_modes x 1)
        Eigenvalues
    emodes: array (number of tetrahedral surface points x num_modes)
        Eigenmodes
    """

    # load tetrahedral surface (as a brainspace object)
    tetra = TetMesh.read_vtk(tetra_file) 

    # normalize tetrahedral surface
    tetra_norm = normalize_vtk(tetra, nifti_input_filename, normalization_type, normalization_factor)

    # calculate eigenvalues and eigenmodes
    fem = Solver(tetra_norm)
    evals, emodes = fem.eigs(k=num_modes)
    
    output_eval_file_main, output_eval_file_ext = os.path.splitext(output_eval_filename)
    output_emode_file_main, output_emode_file_ext = os.path.splitext(output_emode_filename)

    if normalization_type == 'number' or normalization_type == 'volume' or normalization_type == 'constant':
        np.savetxt(output_eval_file_main + '_norm=' + normalization_type + output_eval_file_ext, evals)
        np.savetxt(output_emode_file_main + '_norm=' + normalization_type + output_emode_file_ext, emodes)
    else:
        np.savetxt(output_eval_filename, evals)
        np.savetxt(output_emode_filename, emodes)
    
    return evals, emodes

def calc_volume_eigenmodes(
    surface,
    nifti_input_filename, nifti_output_filename, output_eval_filename, 
    output_emode_filename, num_modes, normalization_type='none',
    normalization_factor=1
):
    """Main function to calculate the eigenmodes of the ROI volume in a nifti file.

    Parameters
    ----------
    nifti_input_filename : str
        Filename of input volume where the relevant ROI have voxel values = 1
    nifti_output_filename : str  
        Filename of nifti file where the output eigenmdoes (in volume space) will be stored
    output_eval_filename : str  
        Filename of text file where the output eigenvalues will be stored
    output_emode_filename : str  
        Filename of text file where the output eigenmodes (in tetrahedral surface space) will be stored    
    num_modes : int
        Number of eigenmodes to be calculated
    normalization_type : str (default: 'none')
        Type of normalization
        number - normalization with respect to the total number of non-zero voxels
        volume - normalization with respect to the total volume of non-zero voxels in physical dimensions   
        constant - normalization with respect to a chosen constant
        others - no normalization
    normalization_factor : float (default: 1)
        Factor to be used in a constant normalization
    """

    # calculate eigenvalues and eigenmodes
    evals, emodes = calc_eig(
        surface, nifti_input_filename, output_eval_filename, output_emode_filename,
        num_modes, normalization_type, normalization_factor
        )

    # project eigenmodes in tetrahedral surface space into volume space
    # prepare transformation
    ROI_data = nib.load(nifti_input_filename)
    roi_data = ROI_data.get_fdata()
    inds_all = np.where(roi_data==1)
    xx = inds_all[0]
    yy = inds_all[1]
    zz = inds_all[2]

    points = np.zeros([xx.shape[0],4])
    points[:,0] = xx
    points[:,1] = yy
    points[:,2] = zz
    points[:,3] = 1

    # transform voxel to world coords
    grid = np.dot(
        ROI_data.affine,
        points.T
    )

    # load tetrahedral surface
    tetra = TetMesh.read_vtk(surface)
    points_surface = tetra.v

    # initialize nifti output array
    new_shape = np.array(roi_data.shape)
    if roi_data.ndim>3:
        new_shape[3] = num_modes
    else:
        new_shape = np.append(new_shape, num_modes)
    new_data = np.zeros(new_shape)

    # perform interpolation of eigenmodes from tetrahedral surface space to volume space
    for mode in range(num_modes):
        interpolated_data = griddata(
            points_surface, emodes[:,mode], grid.T[:,:3], method='linear'
        )
        
        for ind in range(0, len(interpolated_data)):
            new_data[xx[ind],yy[ind],zz[ind],mode] = interpolated_data[ind]

    # save to output nifti file
    img = nib.Nifti1Image(
        new_data, ROI_data.affine, header=ROI_data.header
    )
    nib.save(img, nifti_output_filename)

    return evals