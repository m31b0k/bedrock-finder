/*
This is a tool that finds the location of a bedrock pattern, using CUDA acceleration.
It is based on ChromeCrusher's version and DaMatrix's java version, see those programs for additional documentation.
*/

#include <stdio.h>

int full_pattern[16*16] = {1,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,
                           1,1,1,1,0,0,0,0,0,0,0,0,0,1,0,0,
                           1,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,
                           1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,
                           0,1,0,1,0,0,0,0,0,0,1,1,0,0,0,0,
                           0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,0,
                           0,0,1,1,0,1,1,0,0,0,0,1,0,1,0,1,
                           0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,
                           0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,
                           0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,
                           1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,
                           0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,
                           0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,
                           0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,1,
                           0,0,0,0,0,1,0,0,0,0,0,0,1,0,1,0,
                           0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0};

int blockSize = 1024; // Change this depending on your GPU

// uncomment this if you are using wildcards, enter 2 for unknown blocks instead of 0 or 1
// #define WILDCARD 2

__device__ void find_full_chunk(int *c, int64_t x, int64_t z) {
  // The seed used is based on chunk coordinates, not the world seed
  // see world/gen/ChunkProviderOverworld.java: provideChunk from Mod Coder Pack 9.30
  int64_t seed = (x*341873128712LL + z*132897987541LL)^0x5DEECE66DLL;
    
  for (int a = 0; a < 16; ++a) {
    for(int b = 0; b < 16; ++b) {
      // this does nextInt 250 times and only takes the rightmost 48 bits
      seed = seed*709490313259657689LL + 1748772144486964054LL & 281474976710655LL;

#ifdef WILDCARD
      if(c[a*16+b] != WILDCARD)
#endif
      if(4 <= (seed >> 17) % 5) {
        if(c[a*16+b] != 1)
          return;
      } else {
        if(c[a*16+b] != 0)
          return;
      }

      seed = seed*5985058416696778513LL + -8542997297661424380LL;
    }
  }
  
  // The chunk matches the pattern
  printf("Found chunk at %lld, %lld; real: %lld, %lld\n", x, z, x*16LL, z*16LL);
  return;
}

__global__ void find_full(int *pattern, int startx, int endx, int startz, int endz, int64_t area) {
  // Get the stride and index of the current thread
  int64_t indexX = blockIdx.x * blockDim.x;
  int64_t indexZ = blockIdx.y * blockDim.y + threadIdx.y;

  find_full_chunk(pattern, indexX + startx, indexZ + startz);
}

int main() {
  if (blockSize > 1024) {
    printf("blockSize too high! Max is 1024");
    return 1;
  } else if (blockSize < 4) {
    printf("blockSize too low! Min is 458");
    return 1;
  }

  // 1875000
  int startx = -1875000;
  int endx = 1875000;
  int startz = -1875000;
  int endz = 1875000;
  if (startx < -1875000 || startz < -1875000) {
      printf("startx or startz is too low! Min is -1875000");
      return 1;
  }
  if (endx > 1875000 || endz > 1875000) {
      printf("endx or endz is too high! Max is 1875000");
  }

  int xSize = endx-startx;
  int zSize = endz-startz;
  int64_t areaSize = static_cast<int64_t>(xSize)*static_cast<int64_t>(zSize);

  int *d_full_pattern;
  cudaMalloc((void**)&d_full_pattern, sizeof(int) * 16 * 16);
  cudaMemcpy(d_full_pattern, full_pattern, sizeof(int) * 16 * 16, cudaMemcpyHostToDevice);

  dim3 numBlocks(xSize, (zSize + 1023) / 1024);
  dim3 numThreads(1, blockSize);

  //int64_t numBlocks = (areaSize + static_cast<int64_t>(blockSize) - 1LL) / static_cast<int64_t>(blockSize);
  //find_full<<<numBlocks, blockSize>>>(d_full_pattern, startx, endx, startz, endz, areaSize);
  find_full<<<numBlocks, numThreads>>>(d_full_pattern, startx, endx, startz, endz, areaSize);

  cudaDeviceSynchronize();

  cudaFree(d_full_pattern);

  return 0;
}
