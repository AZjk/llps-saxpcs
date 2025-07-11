#!/bin/bash

# ==== USER CONFIGURABLE SECTION ====
# Choose dataset: "2021-2" or "2022-1"
DATASET="2021-2"
# DATASET="2022-1"

BOOST_CORR=boost_corr_bin
export CUDA_VISIBLE_DEVICES=4,5,6,7
# export CUDA_VISIBLE_DEVICES=0,1,3,7

# ==== PATH CONFIGURATION ====
if [[ "$DATASET" == "2021-2" ]]; then
    # QMAP="/home/beams10/8IDIUSER/Documents/llps-saxpcs/reanalysis_2025_0428/202107_2_qmap.hdf"
    # QMAP="/gdata/s8id-dmdtn/2021-2/babnigg202107_2_nexus/babnigg202107_2_nexus_Sq270_Dq27_Sphi8_Dphi1_q0p0025_Lin.hdf"
    QMAP="/gdata/s8id-dmdtn/2021-2/babnigg202107_2_nexus/babnigg202107_2_nexus_Sq270_Dq27_Sphi8_Dphi1_q0p0032_Lin.hdf"
    OUTPUT="/home/8-id-i/2024-3/2024_1228_qz_llps_analysis/2025_04_analysis/2021-2"
    RECORD_FILE="all_file_2021-2.txt"

    # Uncomment to prepare .bin file list
    # ls /gdata/s8id-dmdtn/2021-2/babnigg202107_2/D029_*/D029_*_*/*.bin >  $RECORD_FILE
    # ls /gdata/s8id-dmdtn/2021-2/babnigg202107_2/B039_*/B039_*_*/*.bin >> $RECORD_FILE
    # ls /gdata/s8id-dmdtn/2021-2/babnigg202107_2/B040_*/B040_*_*/*.bin >> $RECORD_FILE
    # ls /gdata/s8id-dmdtn/2021-2/babnigg202107_2/H041_*/H041_*_*/*.bin >> $RECORD_FILE
    # ls /gdata/s8id-dmdtn/2021-2/babnigg202107_2/H042_*/H042_*_*/*.bin >> $RECORD_FILE
    # ls /gdata/s8id-dmdtn/2021-2/babnigg202107_2/T214_*/T214_*_*/*.bin >> $RECORD_FILE
    # ls /gdata/s8id-dmdtn/2021-2/babnigg202107_2/T215_*/T215_*_*/*.bin >> $RECORD_FILE

elif [[ "$DATASET" == "2022-1" ]]; then
    # QMAP="/home/beams10/8IDIUSER/Documents/llps-saxpcs/reanalysis_2025_0428/202203_qmap.hdf"
    # QMAP="/gdata/s8id-dmdtn/2022-1/babnigg202203_nexus/babnigg202203_nexus_Sq270_Dq27_Sphi8_Dphi1_Q0p0025_Lin.hdf"
    QMAP="/gdata/s8id-dmdtn/2022-1/babnigg202203_nexus/babnigg202203_nexus_Sq270_Dq27_Sphi8_Dphi1_Q0p0032_Lin.hdf"
    OUTPUT="/home/8-id-i/2024-3/2024_1228_qz_llps_analysis/2025_04_analysis/2022-2"
    RECORD_FILE="all_file_2022-1.txt"

    # Uncomment to prepare .bin file list
    # ls /gdata/s8id-dmdtn/2022-1/babnigg202203/E0110_*_*/*.bin >  $RECORD_FILE
    # ls /gdata/s8id-dmdtn/2022-1/babnigg202203/E0111_*_*/*.bin >> $RECORD_FILE
    # ls /gdata/s8id-dmdtn/2022-1/babnigg202203/D0138_*_*/*.bin >> $RECORD_FILE
    # ls /gdata/s8id-dmdtn/2022-1/babnigg202203/G0015_*_*/*.bin >> $RECORD_FILE
    # ls /gdata/s8id-dmdtn/2022-1/babnigg202203/B0076_*_*/*.bin >> $RECORD_FILE
    # ls /gdata/s8id-dmdtn/2022-1/babnigg202203/E0142_*_*/*.bin >> $RECORD_FILE
    # ls /gdata/s8id-dmdtn/2022-1/babnigg202203/B0140_*_*/*.bin >> $RECORD_FILE
else
    echo "ERROR: Unknown dataset \"$DATASET\""
    exit 1
fi

# ==== SPLIT FILES FOR PARALLEL PROCESSING ====
NUM_FILE=$(wc -l < "$RECORD_FILE")
NUM_EACH_WORKER=$(( (NUM_FILE + 3) / 4 ))
echo "Total files: $NUM_FILE; Each worker: $NUM_EACH_WORKER"

split -l "$NUM_EACH_WORKER" --numeric-suffixes "$RECORD_FILE" "${RECORD_FILE}_"

# ==== WORKER FUNCTION ====
gpu_corr_worker () {
    GPU_ID=$1
    FILE_CHUNK="${RECORD_FILE}_0${GPU_ID}"

    echo "Starting job for GPU $GPU_ID using file list $FILE_CHUNK"
    while IFS= read -r RAWFILE; do
        echo "Processing $RAWFILE"
        $BOOST_CORR -r "$RAWFILE" -q "$QMAP" -v -i "$GPU_ID" -w -o "$OUTPUT/cluster_results_all"
        $BOOST_CORR -r "$RAWFILE" -q "$QMAP" -v -i "$GPU_ID" -w -o "$OUTPUT/cluster_results_part1" --begin-frame 0 --end-frame 50000
        $BOOST_CORR -r "$RAWFILE" -q "$QMAP" -v -i "$GPU_ID" -w -o "$OUTPUT/cluster_results_part2" --begin-frame 50000 --end-frame 100000
    done < "$FILE_CHUNK"
}

# ==== LAUNCH GPU WORKERS ====
for GPU_ID in 0 1 2 3; do
    gpu_corr_worker "$GPU_ID" &
done

wait
echo "All GPU jobs completed."

