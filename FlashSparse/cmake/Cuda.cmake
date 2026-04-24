# Helper to find CUDA libraries.
function(cuda_find_library out_path lib_name)
  find_library(${out_path} ${lib_name} PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
    PATH_SUFFIXES lib lib64 REQUIRED)
endfunction()
