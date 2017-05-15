from util import run_system_command


def reconstruct_fourier(md_file, vol, **kwargs):
    """
    PROGRAM
    xmipp_reconstruct_fourier
    USAGE
    Generate 3D reconstructions from projections using direct Fourier interpolation with arbitrary geometry.
    Kaisser-windows are used for interpolation in Fourier space.
    OPTIONS

    -i <md_file>
            Metadata file with input projections
    [-o <volume_file=rec_fourier.vol>]
            Filename for output volume
    [--iter <iterations=1>]
            Number of iterations for weight correction
    [--sym <symfile=c1>]
            Enforce symmetry in projections
    [--padding <proj=2.0> <vol=2.0>]
            Padding used for projections and volume
    [--prepare_fsc <fscfile>]
            Filename root for FSC files
    [--max_resolution <p=0.5>]
            Max resolution (Nyquist=0.5)
    [--weight]
            Use weights stored in the image metadata
    [--thr <threads=1> <rows=1>]
            Number of concurrent threads and rows processed at time by a thread
    [--blob <radius=1.9> <order=0> <alpha=15>]
            Blob parameters
            radius in pixels, order of Bessel function in blob and parameter alpha

    Common options
    [--gui]
            Show a GUI to launch the program.
    [--more]
            Show additional options.
    EXAMPLES
    For reconstruct enforcing i3 symmetry and using stored weights:
            xmipp_reconstruct_fourier  -i reconstruction.sel --sym i3 --weight
    """
    options = ['--{0} {1}'.format(str(key), str(value))
               for (key, value) in kwargs.items()]
    xmipp_reconstruct_fourier = 'xmipp_reconstruct_fourier -i {0} -o {1} {2}'.format(
        str(md_file), str(vol), ' '.join(options))
    run_system_command(xmipp_reconstruct_fourier)
    return vol
