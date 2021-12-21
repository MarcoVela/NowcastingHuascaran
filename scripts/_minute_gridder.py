#!/usr/bin/env python3
"""
Copyright (C) 2020-2021 Space Science and Engineering Center (SSEC), University of Wisconsin-Madison.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

This file is part of the CSPP Geo Gridded GLM software package. CSPP Geo
Gridded GLM takes GOES GLM Level 2 LCFA files and grids them to the ABI
fixed grid. It does this using the open source glmtools python package by
Eric Bruning. glmtools can be found in the runtime directory of this package
or at https://github.com/deeplycloudy/glmtools.
"""

parse_desc = """Create one minute NetCDF4 grids (and, optionally, AWIPS-compatible tiles) from GLM flash data.

Example usage:

    %(prog)s \\
        --goes-sector conus \\
        --create-tiles \\
        -vv \\
        OR_GLM-L2-LCFA_G17_s20182750032000_e20182750032200_c20182750032225.nc \\
        OR_GLM-L2-LCFA_G17_s20182750032200_e20182750032400_c20182750032426.nc \\
        OR_GLM-L2-LCFA_G17_s20182750032400_e20182750033000_c20182750033025.nc
"""

import numpy as np
from datetime import datetime, timedelta
import os
import sys
import tempfile
import shutil
import atexit
from glob import glob
import socket
import signal
from netCDF4 import Dataset
# from multiprocessing import freeze_support # https://docs.python.org/2/library/multiprocessing.html#multiprocessing.freeze_support
from functools import partial
from lmatools.grid.make_grids import write_cf_netcdf_latlon, write_cf_netcdf_noproj, write_cf_netcdf_fixedgrid
from lmatools.grid.make_grids import dlonlat_at_grid_center, grid_h5flashfiles
from glmtools.grid.make_grids import grid_GLM_flashes
from glmtools.io.glm import parse_glm_filename
from lmatools.grid.fixed import get_GOESR_grid, get_GOESR_coordsys

import logging

log = logging.getLogger(__name__)

import warnings

warnings.filterwarnings("ignore")

import dask

dask.config.set(num_workers=1)


def create_parser():
    import argparse
    prog = os.getenv('PROG_NAME', sys.argv[0])
    parser = argparse.ArgumentParser(prog=prog,
                                     description=parse_desc,
                                     formatter_class=argparse.RawTextHelpFormatter)  # RawTextHelpFormatter preserves our newlines in the example usage message
    parser.add_argument('-v', '--verbose', dest='verbosity', action="count", default=0,
                        help="each occurrence increases verbosity 1 level through ERROR-WARNING-INFO-DEBUG\n"
                             "(default: ERROR)")
    parser.add_argument('-l', '--log', dest="log_fn", default=None,
                        help="specify a log filename.\n"
                             "(default: print to screen).")
    parser.add_argument('-o', '--output-dir', metavar='OUTPUT_DIR',
                        default=os.getcwd(), help="output directory (default: use current directory)")
    parser.add_argument('--goes-sector', default="full", choices=['full', 'conus', 'meso'],
                        help="If sector is meso, ctr_lon and ctr_lat \n"
                             "are interpreted as the ctr_x and ctr_y of the fixed grid.\n"
                             "(default: full)")
    parser.add_argument("-t", "--create-tiles", default=False, action='store_true',
                        help="create AWIPS-compatible tiles (default: off)")
    parser.add_argument('--ctr-lat', metavar='latitude',
                        type=float, help='center latitude (required for meso)')
    parser.add_argument('--ctr-lon', metavar='longitude',
                        type=float, help='center longitude (required for meso)')
    parser.add_argument('-1', "--realtime", default=False, action='store_true',
                        help="enable 'realtime' mode, where we expect only one input file,\n"
                             "find the surrounding trio, and automatically determine if a full minute\n"
                             "of data is available (default: off)")
    parser.add_argument('--system-environment-prefix', default="CG",
                        help="set the system environment prefix for the output grids (default: CG)")
    parser.add_argument('--system-environment-prefix-tiles', default="CSPP_OR",
                        help="set the system environment prefix for the output tiles (default: CSPP_OR)")
    # from Requirements: "Input is one or more GLM LCFA (L2+) files in mission standard format (nominally three 20-second input files)"
    parser.add_argument(dest='filenames', metavar='filename', nargs='+')
    return parser


def get_resolution(args):
    closest_resln = 2.0  # hardcoding resolution to 2.0 for now. see nearest_resolution in make_glm_grids for how we could expose this if we change our minds.
    resln = '{0:4.1f}km'.format(closest_resln).replace(' ', '')
    return resln


# if provided "auto" position, we determine the sensor from the filename
def get_goes_position(filenames):
    if all("_G16_" in f for f in filenames):
        return "east"
    if all("_G17_" in f for f in filenames):
        return "west"

    # we require that all files are from the same sensor and raise an exception if not
    raise ValueError("could not determine GOES position - did you provide a mix of satellites?")


def get_start_end(filenames, start_time=None, end_time=None):
    """Compute start and end time of data based on filenames."""
    base_filenames = [os.path.basename(p) for p in filenames]

    filename_infos = [parse_glm_filename(f) for f in base_filenames]
    # opsenv, algorithm, platform, start, end, created = parse_glm_filename(f)
    filename_starts = [info[3] for info in filename_infos]
    filename_ends = [info[4] for info in filename_infos]
    start_time = min(filename_starts)

    # Used to use max(filename_ends), but on 27 Oct 2020, the filename
    # ends started to report the time of the last event in the file,
    # causing a slight leakage (usually less than a second) into the
    # next minute. This caused two minutes of grids to be produced for every
    # three twenty second files passed to this script.
    # Instead, we now assume every LCFA file is 20 s long, beginning with
    # the start time. No doubt in the future we will see filenames that no
    # longer start on an even minute boundary.
    end_time = max(filename_starts) + timedelta(0, 20)

    if start_time is None or end_time is None:
        raise ValueError("Could not determine start/end time")

    return start_time, end_time


def get_sector_shortstring(args):
    if args.goes_sector == 'full':
        return 'F'
    elif args.goes_sector == 'conus':
        return 'C'
    elif args.goes_sector == 'meso':
        return 'M1'
    else:
        raise RuntimeError("sector not recognized")


def get_outpath_base(args):
    """create a base outpath string to feed glmtools
    
    from glmtools:
            outpath can be a template string; defaults to {'./{dataset_name}'}
        Available named arguments in the template are:
            dataset_name: standard GOES imagery format, includes '.nc'. Looks like
                OR_GLM-L2-GLMM1-M3_G16_s20181830432000_e20181830433000_c20200461148520.nc
            start_time, end_time: datetimes that can be used with strftime syntax, e.g.
                './{start_time:%y/%b/%d}/GLM_{start_time:%Y%m%d_%H%M%S}.nc'
    """
    ordered_filenames = sorted(args.filenames)
    _, _, platform, start_time, _, _ = parse_glm_filename(os.path.basename(ordered_filenames[0]))
    _, _, _, _, end_time, _ = parse_glm_filename(os.path.basename(ordered_filenames[-1]))

    sector_short = get_sector_shortstring(args)
    mode = "M3"  # FIXME: is GLM always in M3?

    # example string: OR_GLM-L2-GLMC-M3_G17_s20182750032000_e20182750033000_c20210431923130.nc
    dsname = "{environment_prefix}_GLM-L2-GLM{sector_short}-{mode}_{platform}_s{start_time}_e{end_time}_c{created_time}.nc".format(
        environment_prefix=args.system_environment_prefix,
        sector_short=sector_short,
        mode=mode,
        platform=platform,
        start_time=start_time.strftime("%Y%j%H%M%S0"),
        end_time=end_time.strftime("%Y%j%H%M%S0"),
        created_time=datetime.utcnow().strftime("%Y%j%H%M%S0")
    )
    return dsname


def grid_setup(args, work_dir=os.getcwd()):
    # When passed None for the minimum event or group counts, the gridder will skip
    # the check, saving a bit of time.
    min_events = None
    min_groups = None

    if args.realtime:
        if len(args.filenames) != 1:
            log.error("realtime mode only accepts one input file")
            exit(1)
        glminfo = parse_glm_filename(os.path.basename(args.filenames[0]))

        globstring = "{}_{}_{}_s{}*".format(glminfo[0], glminfo[1], glminfo[2], glminfo[3].strftime("%Y%j%H%M"))
        fileglob = glob(os.path.join(os.path.dirname(args.filenames[0]), globstring))
        if len(fileglob) != 3:
            print("There are not (yet) three GLM files from this minute. This may be expected. Exiting.")
            exit(0)
        args.filenames = fileglob

    for f in args.filenames:
        if not os.path.exists(f):
            log.error("Tried to grid file that does not exist: {}".format(f))
            exit(1)

    if args.goes_sector == "meso" and (args.ctr_lat == None or args.ctr_lon == None):
        log.error("sector 'meso' requires --ctr-lat & --ctr-lon")
        exit(1)

    try:
        start_time, end_time = get_start_end(args.filenames)
    except:
        log.error("Could not parse start & end times from one or more of the files provided:")
        log.error(", ".join(args.filenames))
        exit(1)

    base_date = datetime(start_time.year, start_time.month, start_time.day)
    proj_name = 'geos'

    outputpath = os.path.join(work_dir, get_outpath_base(args))  # GLMTools expects a template in addition to the path

    goes_position = get_goes_position(args.filenames)

    resln = get_resolution(args)
    view = get_GOESR_grid(position=goes_position,
                          view=args.goes_sector,
                          resolution=resln)
    nadir_lon = view['nadir_lon']
    dx = dy = view['resolution']
    nx, ny = view['pixelsEW'], view['pixelsNS']
    geofixcs, grs80lla = get_GOESR_coordsys(sat_lon_nadir=nadir_lon)

    if 'centerEW' in view:
        x_ctr, y_ctr = view['centerEW'], view['centerNS']
    elif args.goes_sector == 'meso':
        # use ctr_lon, ctr_lat to get the center of the mesoscale FOV
        x_ctr, y_ctr, z_ctr = geofixcs.fromECEF(
            *grs80lla.toECEF(args.ctr_lon, args.ctr_lat, 0.0))
    else:
        # FIXME: is it possible to get here? if so, what should happen?
        raise RuntimeError

    # Need to use +1 here to convert to xedge, yedge expected by gridder
    # instead of the pixel centroids that will result in the final image
    nx += 1
    ny += 1
    x_bnd = (np.arange(nx, dtype='float') - (nx) / 2.0) * dx + x_ctr + 0.5 * dx
    y_bnd = (np.arange(ny, dtype='float') - (ny) / 2.0) * dy + y_ctr + 0.5 * dy
    log.debug(("initial x,y_ctr", x_ctr, y_ctr))
    log.debug(("initial x,y_bnd", x_bnd.shape, y_bnd.shape))
    x_bnd = np.asarray([x_bnd.min(), x_bnd.max()])
    y_bnd = np.asarray([y_bnd.min(), y_bnd.max()])

    geofixcs, grs80lla = get_GOESR_coordsys(sat_lon_nadir=nadir_lon)
    ctr_lon, ctr_lat, ctr_alt = grs80lla.fromECEF(
        *geofixcs.toECEF(x_ctr, y_ctr, 0.0))
    fixed_grid = geofixcs
    log.debug((x_bnd, y_bnd, dx, dy, nx, ny))

    output_writer = partial(write_cf_netcdf_fixedgrid, nadir_lon=nadir_lon)

    gridder = grid_GLM_flashes
    output_filename_prefix = 'GLM'
    grid_kwargs = dict(proj_name=proj_name,
                       base_date=base_date, do_3d=False,
                       dx=dx, dy=dy, frame_interval=60.0,
                       x_bnd=x_bnd, y_bnd=y_bnd,
                       ctr_lat=ctr_lat, ctr_lon=ctr_lon, outpath=outputpath,
                       min_points_per_flash=min_events,
                       output_writer=output_writer, subdivide=1,
                       # subdivide the grid this many times along each dimension
                       output_filename_prefix=output_filename_prefix,
                       output_kwargs={'scale_and_offset': False},
                       spatial_scale_factor=1.0)

    # if args.fixed_grid:
    #    grid_kwargs['fixed_grid'] = True
    #    grid_kwargs['nadir_lon'] = nadir_lon
    # if args.split_events:
    grid_kwargs['clip_events'] = True
    if min_groups is not None:
        grid_kwargs['min_groups_per_flash'] = min_groups
    grid_kwargs['energy_grids'] = ('total_energy',)
    if (proj_name == 'pixel_grid') or (proj_name == 'geos'):
        grid_kwargs['pixel_coords'] = fixed_grid
    grid_kwargs['ellipse_rev'] = -1  # -1 (default) = infer from date in each GLM file
    return gridder, args.filenames, start_time, end_time, grid_kwargs


def get_cspp_gglm_version():
    try:
        version_filename = os.path.join(os.getenv('CSPP_GEO_GGLM_HOME'), ".VERSION.txt")
        return open(version_filename, 'r').read().rstrip()
    except:
        return "unknown"


def add_gglm_attrs(netcdf_filename, input_filenames):
    try:
        nc = Dataset(netcdf_filename, 'a')
        setattr(nc, 'cspp_geo_gglm_version', get_cspp_gglm_version())
        setattr(nc, 'cspp_geo_gglm_production_host', socket.gethostname())
        setattr(nc, 'cspp_geo_gglm_input_files', ",".join([os.path.basename(f) for f in input_filenames]))
        nc.close()
    except:
        log.error("could not add CSPP Geo GGLM attributes to {}".format(netcdf_filename))


def alarm_handler(signum, frame):
    raise OSError("Timeout exceeded!")


if __name__ == '__main__':
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(10 * 60)  # timeout if we're not done after 10 minutes

    #    freeze_support() # nb. I don't think this is needed as we're not making windows execs at this time
    parser = create_parser()
    args = parser.parse_args()

    # Configure logging
    levels = [logging.ERROR, logging.WARN, logging.INFO, logging.DEBUG]
    clamp = lambda n, minn, maxn: max(min(maxn, n),
                                      minn)  # used below to keep us from going off the end of the logging levels
    logging.basicConfig(level=levels[clamp(args.verbosity, 0, len(levels) - 1)], filename=args.log_fn)
    if levels[min(3, args.verbosity)] > logging.DEBUG:
        import warnings

        warnings.filterwarnings("ignore")
    log.info("Starting GLM Gridding")
    log.debug("Starting script with: %s", sys.argv)

    # set up output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # set up temporary dir
    tempdir_path = tempfile.mkdtemp(suffix=None, prefix="tmp-glm-grids-", dir=os.getcwd())
    log.info("working in: {}".format(tempdir_path))
    # clean our temporary dir on exit
    atexit.register(shutil.rmtree, tempdir_path)

    # do the gridding
    gridder, glm_filenames, start_time, end_time, grid_kwargs = grid_setup(args, work_dir=tempdir_path)
    gridder_return = gridder(glm_filenames, start_time, end_time, **grid_kwargs)

    gridded_files = []
    for subgrid in gridder_return:
        for gridded_file in subgrid[1]:
            gridded_files.append(gridded_file)

    # we need to add attributes here due to an issue where satpy (or its dependencies) are
    # holding the input gridded file open until the process exits
    for f in gridded_files:
        add_gglm_attrs(f, glm_filenames)

    # (optionally) do tiling
    if args.create_tiles:

        sector = get_goes_position(glm_filenames)
        if sector == "east":
            sector_id = "GOES_EAST"
        elif sector == "west":
            sector_id = "GOES_WEST"
        else:
            raise RuntimeError("could not determine sector_id")

        from satpy import Scene

        for gridded_file in gridded_files:
            log.info("TILING: {}".format(gridded_files))
            scn = Scene(reader='glm_l2', filenames=[gridded_file])  # n.b. satpy requires a list of filenames
            scn.load([
                'DQF',
                'flash_extent_density',
                'minimum_flash_area',
                'total_energy',
            ])

            scn.save_datasets(writer='awips_tiled',
                              template='glm_l2_radf',
                              sector_id=sector_id,
                              # sector_id becomes an attribute in the output files and may be another legacy kind of thing. I'm not sure how much is is actually used here.
                              source_name="",
                              # You could probably make source_name an empty string. I think it is required by the writer for legacy reasons but isn't actually used for the glm output
                              base_dir=tempdir_path,
                              # base_dir is the output directory. I think blank is the same as current directory.
                              tile_size=(506, 904),
                              # tile_size is set to the size of the GLMF sample tiles we were given and should match the full disk ABI tiles which is what they wanted
                              check_categories=False,
                              # check_categories is there because of that issue I mentioned where DQF is all valid all the time so there is no way to detect empty tiles unless we ignore the "category" products
                              environment_prefix=args.system_environment_prefix_tiles,
                              compress=True)

    # pick up output files from the tempdir
    # output looks like: CG_GLM-L2-GLMC-M3_G17_T03_20200925160040.nc
    log.debug("files in {}".format(tempdir_path))
    log.debug(os.listdir(tempdir_path))
    log.debug("moving output to {}".format(args.output_dir))
    tiled_path = os.path.join(tempdir_path,
                              '{}_GLM-L2-GLM*-M?_G??_T??_*.nc'.format(args.system_environment_prefix_tiles))
    tiled_files = glob(tiled_path)
    for f in tiled_files:
        add_gglm_attrs(f, glm_filenames)
        shutil.move(f, os.path.join(args.output_dir, os.path.basename(f)))
    for f in gridded_files:
        shutil.move(f, os.path.join(args.output_dir, os.path.basename(f)))

    # tempdir cleans itself up via atexit, above
