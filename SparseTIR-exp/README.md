# Experimental Steps

```bash
docker run --name fty-exp --gpus all -v ~/projects/sparsene:/work/sparsene -itd nvidia/cuda:12.6.1-cudnn-devel-ubuntu20.04 bash
# Must use Ubuntu 20.04!
apt install lsb-release wget software-properties-common gnupg zlib1g-dev git cmake vim

# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh ./Miniconda3-latest-Linux-x86_64.sh

# Create Python 3.9 environment (stick to 3.9!)
conda create -n sptir python==3.9
conda activate sptir

# Install LLVM
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
./llvm.sh 10
apt install libpolly-18-dev
ln -s /usr/bin/llvm-config-10 /usr/bin/llvm-config

# Install SparseTIR
git clone --recursive https://github.com/fty1777/SparseTIR-exp SparseTIR
cd SparseTIR
echo set\(USE_LLVM \"llvm-config --ignore-libllvm --link-static\"\) >> config.cmake
echo set\(HIDE_PRIVATE_SYMBOLS ON\) >> config.cmake
echo set\(USE_CUDA ON\) >> config.cmake
echo set\(USE_CUBLAS ON\) >> config.cmake
echo set\(USE_CUDNN ON\) >> config.cmake
mkdir -p build
cd build
cmake ..
make -j$(nproc)

# Return to root directory
cd ..

# Navigate to python directory, install python binding and dependencies (don't miss the dot!)
cd python
conda activate sptir
pip install . dgl==1.0.0 torchdata==0.7.0 pandas pyyaml torch==2.1.0 "numpy<2" pydantic setuptools==65.5.1 pytest ogb
cd ..

# Experiment path
cd examples/spmm

# Prepare mat_list.txt
python /work/sparsene/exp/find_mats.py <dir> > mat_list.txt

# Run experiment
python exp.py
```