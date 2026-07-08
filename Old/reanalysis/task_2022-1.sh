#!/bin/bash

source /APSshare/miniconda/x86_64/etc/profile.d/conda.sh
DEFAULT_ENV=/home/beams/8IDIUSER/.conda/envs/i2412_llps

conda activate ${DEFAULT_ENV}


INDEX=$1

CORR_BIN=/clhome/8IDIUSER/.conda/envs/lgpu/bin/boost_corr
QMAP=/home/beams10/8IDIUSER/Documents/llps-saxpcs/qmap_reanalysis/babnigg202203_Sq270_Dq27_Sphi8_Dphi1_Lin.h5

# remember to create the output directory before run this script; multiple works
OUTPUT=/home/8ididata/2022-1/babnigg202203/cluster_results_QZ

RECORD_FILE=all_file_2022-1.txt

# # prepare h5 files
# ls /gdata/s8id-dmdtn/2022-1/babnigg202203/E0110_*_*/*.bin >  $RECORD_FILE 
# ls /gdata/s8id-dmdtn/2022-1/babnigg202203/E0111_*_*/*.bin >> $RECORD_FILE 
# ls /gdata/s8id-dmdtn/2022-1/babnigg202203/D0138_*_*/*.bin >> $RECORD_FILE 
# ls /gdata/s8id-dmdtn/2022-1/babnigg202203/G0015_*_*/*.bin >> $RECORD_FILE 
# ls /gdata/s8id-dmdtn/2022-1/babnigg202203/B0076_*_*/*.bin >> $RECORD_FILE 
# ls /gdata/s8id-dmdtn/2022-1/babnigg202203/E0142_*_*/*.bin >> $RECORD_FILE 
# ls /gdata/s8id-dmdtn/2022-1/babnigg202203/B0140_*_*/*.bin >> $RECORD_FILE 

# split the files into 4 files because we will use 4 GPUs to analyze them in parallel.
# 4 files will be generated, with filename pattern as raw_input_0x, x = 0, 1, 2, 3

NUM_FILE=$(wc -l ${RECORD_FILE} | awk '{ print $1}')
NUM_EACH_WORKER=$(( ($NUM_FILE+3)/4 ))
echo $NUM_FILE $NUM_EACH_WORKER
split -l $NUM_EACH_WORKER --numeric-suffixes $RECORD_FILE ${RECORD_FILE}_


# define a GPU worker that takes GPU_ID as a variable; it will analyze the files listed in
# raw_input_GPU_ID

gpu_corr_worker () {
    echo 'starting job for GPU:' $1
    for RAWFILE in $(cat ${RECORD_FILE}_0$1)
    do
		# run with all frames
		boost_corr -r $RAWFILE -q $QMAP  -v -i -2 -ow -o './results/cluster_results'
		# run with the first 50,000 frames
		boost_corr -r $RAWFILE -q $QMAP  -v -i -2 -ow -o './results/cluster_results_part1' -begin_frame 0 -end_frame 50000
		# run with the last 50,000 frames
		boost_corr -r $RAWFILE -q $QMAP  -v -i -2 -ow -o './results/cluster_results_part2' -begin_frame 50000 -end_frame 100000
    done
}


# launch 4 workers
for GPU_ID in 0 1 2 3
do
    # remove all previous output logs
    # rm output_$GPU_ID
    gpu_corr_worker $GPU_ID >> output_$GPU_ID 2>&1 &
done
