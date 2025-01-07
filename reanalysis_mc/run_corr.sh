#!/bin/bash

source /APSshare/miniconda/x86_64/etc/profile.d/conda.sh
DEFAULT_ENV=/home/beams/MQICHU/miniconda3/envs/e2411_production

conda activate ${DEFAULT_ENV}

QMAP=./llps_qmap.hdf
# RAWFILE='/gdata/s8id-dmdtn/2021-2/babnigg202107_2/E047_VPAVG_H04_060C10p_att00_Rq0_00020/E047_VPAVG_H04_060C10p_att00_Rq0_00020.bin'
RAWFILE='/gdata/s8id-dmdtn/2021-2/babnigg202107_2/B040_VPAVG_H02_200C10p_att00_Rq0/B040_VPAVG_H02_200C10p_att00_Rq0_00001/B040_VPAVG_H02_200C10p_att00_Rq0_00001.bin'

# run with all frames 
boost_corr -r $RAWFILE -q $QMAP  -v -i 3 -ow -o 'cluster_results'

# run with the first 50,000 frames
boost_corr -r $RAWFILE -q $QMAP  -v -i 3 -ow -o 'cluster_results_part1' -begin_frame 0 -end_frame 50000

# run with the last 50,000 frames
boost_corr -r $RAWFILE -q $QMAP  -v -i 3 -ow -o 'cluster_results_part2' -begin_frame 50000 -end_frame 100000

# move and rename the files to cluster_results
./move_and_rename.sh
