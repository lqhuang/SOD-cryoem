from util import run_system_command


def project(map_file, proj, only_output_star=False, **kwargs):
    """
    project(map_file, proj, **kwargs)
    run relion_project with a system process

    Input:
        map_file: input mrc file to be projected / input 3D volume to be projected
        proj: rootname for output projections
        *ang: STAR file with orientations for multiple projections

    Output:
        star: name for output star file
        mrcs: name for output mrcs file

    Parameters:
    relion_project
    +++ RELION: command line arguments (with defaults for optional ones between parantheses) +++
    ====== Options =====
                                    --i : Input map to be projected
                             --o (proj) : Rootname for output projections
                          --ctf (false) : Apply CTF to reference projections
               --ctf_phase_flip (false) : Flip phases of the CTF in the output projections
        --ctf_intact_first_peak (false) : Ignore CTFs until their first peak?
                           --angpix (1) : Pixel size (in Angstroms)
                              --mask () : Mask that will be applied to the input map prior to making projections
                           --ang (None) : STAR file with orientations for multiple projections (if None, assume single projection)
                              --rot (0) : First Euler angle (for a single projection)
                             --tilt (0) : Second Euler angle (for a single projection)
                              --psi (0) : Third Euler angle (for a single projection)
                             --xoff (0) : Origin X-offsets (in pixels) (for a single projection)
                             --yoff (0) : Origin Y-offsets (in pixels) (for a single projection)
                             --zoff (0) : Origin Z-offsets (in pixels) (for a single 3D rotation)
                    --add_noise (false) : Add noise to the output projections (only with --ang)
                      --white_noise (0) : Standard deviation of added white Gaussian noise
                       --model_noise () : Model STAR file with power spectra for coloured Gaussian noise
                 --subtract_exp (false) : Subtract projections from experimental images (in --ang)
         --ignore_particle_name (false) : Ignore the rlnParticleName column (in --ang)
                       --3d_rot (false) : Perform 3D rotations instead of projection into 2D images
                          --maxres (-1) : Maximum resolution (in Angstrom) to consider in Fourier space (default Nyquist)
                              --pad (2) : Padding factor
                         --ctf2 (false) : Apply CTF*CTF to reference projections
                           --NN (false) : Use nearest-neighbour instead of linear interpolation
    """
    if isinstance(map_file, str) is False:
        pass # Not implemented
    options = ['--{0} {1}'.format(str(key), str(value)) for (key, value) in kwargs.items()]
    relion_project = 'relion_project --i {0} --o {1} {2}'.format(
        str(map_file), str(proj), ' '.join(options))
    # try:
    output = run_system_command(relion_project)

    star_file = str(proj) + '.star'
    if only_output_star:
        return star_file
    else:
        mrcs_file = str(proj) + '.mrcs'
        return star_file, mrcs_file

def reconstruct(star_file, output_mrc, threads=1, **kwargs):
    """
    Run relion_reconstruct with a system process

    Input:
        star_file: Input STAE file with the projection images and their orientations
        output_mrc: Name for output mrc file
        threads: Number of threads to use for FFTs

    Parameters:
    relion_reconstruct
    +++ RELION: command line arguments (with defaults for optional ones between parantheses) +++
    ====== General options =====
                             --debug () : Rootname for debug reconstruction files
                   --debug_ori_size (1) : Rootname for debug reconstruction files
                       --debug_size (1) : Rootname for debug reconstruction files
                                 --i () : Input STAR file with the projection images and their orientations
                       --o (relion.mrc) : Name for output reconstruction
                             --sym (c1) : Symmetry group
                           --angpix (1) : Pixel size (in Angstroms)
                          --maxres (-1) : Maximum resolution (in Angstrom) to consider in Fourier space (default Nyquist)
                              --pad (2) : Padding factor
                                --j (1) : Number of threads to use for FFTs
    ====== CTF options =====
                          --ctf (false) : Apply CTF correction
        --ctf_intact_first_peak (false) : Leave CTFs intact until first peak
            --ctf_phase_flipped (false) : Images have been phase flipped
             --only_flip_phases (false) : Do not correct CTF-amplitudes, only flip phases
                      --beamtilt_x (0.) : Beamtilt in the X-direction (in mrad)
                      --beamtilt_y (0.) : Beamtilt in the Y-direction (in mrad)
    ====== Expert options =====
                          --subtract () : Subtract projections of this map from the images used for reconstruction
                           --NN (false) : Use nearest-neighbour instead of linear interpolation before gridding correction
                         --blob_r (1.9) : Radius of blob for gridding interpolation
                           --blob_m (0) : Order of blob for gridding interpolation
                          --blob_a (15) : Alpha-value of blob for gridding interpolation
                            --iter (10) : Number of gridding-correction iterations
                           --refdim (3) : Dimension of the reconstruction (2D or 3D)
                   --angular_error (0.) : Apply random deviations with this standard deviation (in degrees) to each of the 3 Euler angles
                     --shift_error (0.) : Apply random deviations with this standard deviation (in pixels) to each of the 2 translations
                --fom_weighting (false) : Weight particles according to their figure-of-merit (_rlnParticleFigureOfMerit)
                               --fsc () : FSC-curve for regularized reconstruction
                       --3d_rot (false) : Perform 3D rotations instead of backprojections from 2D images
                 --reconstruct_ctf (-1) : Perform a 3D reconstruction from 2D CTF-images, with the given size in pixels
    """
    options = ['--{0} {1}'.format(str(key), str(value)) for (key, value) in kwargs.items()]
    relion_reconstruct = 'relion_reconstruct --i {0} --o {1} --j {2} {3}'.format(
        str(star_file), str(output_mrc), str(threads), ' '.join(options))
    # try:
    output = run_system_command(relion_reconstruct)
