#!/bin/bash

# # Insatll RoDe and Sputnik
cd RoDe &&
rm -rf build &&
mkdir build &&
cd build &&
cmake .. &&
make &&
cd .. &&
cd .. 
