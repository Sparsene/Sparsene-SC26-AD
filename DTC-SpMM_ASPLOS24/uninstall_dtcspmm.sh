#!/bin/bash
set -e
export DTC_HOME=$(pwd)

pip uninstall -y DTCSpMM

cd ${DTC_HOME}/third_party/ && bash clean.sh

cd ${DTC_HOME}/DTC-SpMM && bash clean.sh