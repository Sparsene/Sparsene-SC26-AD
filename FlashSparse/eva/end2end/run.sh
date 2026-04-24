#!/bin/bash

# 1. run gcn
python ./gcn/eva_gcn_fs.py &&

# 2. run agnn
python ./agnn/eva_agnn_fs.py &&