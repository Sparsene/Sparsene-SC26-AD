include(cmake/Cuda.cmake)

# TODO(tgale): Move cuSPARSE, cuBLAS deps to test & benchmark only.
cuda_find_library(CUDART_LIBRARY cudart_static)
cuda_find_library(CUBLAS_LIBRARY cublas_static)
cuda_find_library(CUSPARSE_LIBRARY cusparse_static)
list(APPEND CUDA_LIBS "cudart_static;cublas_static;cusparse_static;culibos;cublasLt_static")
