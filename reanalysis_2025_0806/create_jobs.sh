#!/bin/bash

BOOST_CORR=boost_corr_bin
QUEUE_DIR="/home/beams/8IDIUSER/.simple_gpu_scheduler/queue"
mkdir -p "$QUEUE_DIR"

# ==== Dataset Configuration ====
declare -A QMAPS
declare -A OUTPUTS
declare -A RECORD_FILES

# QMAPS["2021-2"]="/gdata/s8id-dmdtn/2021-2/babnigg202107_2_nexus/babnigg202107_2_nexus_Sq270_Dq27_Sphi8_Dphi1_q0p0032_Lin.hdf"
QMAPS["2021-2"]="/gdata/s8id-dmdtn/2021-2/babnigg202107_2_nexus/babnigg202107_2_nexus_Sq270_Dq27_Sphi8_Dphi1_q0p0032_Lin_2.hdf"
OUTPUTS["2021-2"]="/home/8-id-i/2024-3/2024_1228_qz_llps_analysis/2025_08_analysis/2021-2"
RECORD_FILES["2021-2"]="all_file_2021-2.txt"

QMAPS["2022-1"]="/gdata/s8id-dmdtn/2022-1/babnigg202203_nexus/babnigg202203_nexus_Sq270_Dq27_Sphi8_Dphi1_Q0p0032_Lin.hdf"
OUTPUTS["2022-1"]="/home/8-id-i/2024-3/2024_1228_qz_llps_analysis/2025_08_analysis/2022-1"
RECORD_FILES["2022-1"]="all_file_2022-1.txt"

index=0

for DATASET in "${!QMAPS[@]}"; do
    QMAP="${QMAPS[$DATASET]}"
    OUTPUT="${OUTPUTS[$DATASET]}"
    RECORD_FILE="${RECORD_FILES[$DATASET]}"

    # Process record file
    while IFS= read -r RAWFILE; do
        [[ -z "$RAWFILE" ]] && continue

        JOB_FILE="$QUEUE_DIR/job_$(printf "%08d" "$index").sh"

        cat > "$JOB_FILE" <<EOF
#!/bin/bash
$BOOST_CORR -r "$RAWFILE" -q "$QMAP" -v -i 0 -w -o "$OUTPUT/cluster_results_all"
$BOOST_CORR -r "$RAWFILE" -q "$QMAP" -v -i 0 -w -o "$OUTPUT/cluster_results_part1" --begin-frame 0 --end-frame 50000
$BOOST_CORR -r "$RAWFILE" -q "$QMAP" -v -i 0 -w -o "$OUTPUT/cluster_results_part2" --begin-frame 50000 --end-frame 100000
EOF

        chmod +x "$JOB_FILE"
        ((index++))
    done < "$RECORD_FILE"
done

echo "✅ Created $index job files in $QUEUE_DIR"

