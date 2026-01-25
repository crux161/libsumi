#pragma once

#ifdef __CUDACC__
    #define SUMI_CTX __host__ __device__
#else
    #include <cmath>
    #include <cstdint>
    // FIX: Changed from 'inline' to empty, because sumi.h already writes 'inline' explicitly.
    #define SUMI_CTX 
    #define SUMI_DEVICE
    #define SUMI_HOST
#endif
