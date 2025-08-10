#!/bin/bash

# QMAP="/gdata/dm/8ID/8IDI/2025-1/qzhang202503/data/rigaku3m_qmap_S360_D36_Lin_refined.h5"
QMAP="/gdata/dm/8ID/8IDI/2025-1/qzhang202503/data/rigaku_s360_d36_lin_MC.hdf"
for raw in /gdata/dm/8ID/8IDI/2025-1/qzhang202503/data/E0171_H06-c6c5-4_a0014_f100000_r00001/E0171_H06-c6c5-4_a0014_f100000_r00001.bin.000 /gdata/dm/8ID/8IDI/2025-1/qzhang202503/data/E0172_H06-c6c5-4_a0011_f100000_r00001/E0172_H06-c6c5-4_a0011_f100000_r00001.bin.000 /gdata/dm/8ID/8IDI/2025-1/qzhang202503/data/E0173_H06-c6c5-4_a0009_f100000_r00001/E0173_H06-c6c5-4_a0009_f100000_r00001.bin.000 /gdata/dm/8ID/8IDI/2025-1/qzhang202503/data/E0174_H06-c6c5-4_a0007_f100000_r00001/E0174_H06-c6c5-4_a0007_f100000_r00001.bin.000; do
    echo $raw
    boost_corr_dev -r $raw -q $QMAP -i 0 -v 
done
