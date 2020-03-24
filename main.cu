/*
This is a tool that finds the location of a bedrock pattern, using CUDA acceleration.
It is based on ChromeCrusher's version and DaMatrix's java version, see those programs for additional documentation.
*/

#include <stdio.h>

// Located at 123, -456
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

// Located at -98, 76
#define SUBX 8 // Submatrix x size
#define SUBZ 8 // Submatrix z size
int sub_pattern[SUBX*SUBZ] = {0,0,1,0,0,0,1,0,
                                0,0,0,0,0,0,0,0,
                                0,0,1,1,0,0,0,0,
                                0,1,1,0,0,0,0,0,
                                0,0,0,1,0,0,0,0,
                                0,0,0,0,0,1,0,1,
                                1,0,0,0,1,0,0,0,
                                0,0,1,0,0,1,0,1};

// Change this depending on your GPU, it can be between 458 and 1024 (lower limit due to grid y-size limitation, upper limit due to hardware thread limit and cuda thread per block limit)
// Lowering the value means a higher grid y-size
// For best performance, it should be the maximum amount of parallel threads on your GPU
int blockSize = 1024;

// uncomment this if you are using wildcards. In the pattern, enter 2 for unknown blocks instead of 0 or 1
// #define WILDCARD 2

__global__ void find_full(int *c, int startx, int startz) {
  // Get chunk coordinates using thread index
  int x = blockIdx.x * blockDim.x + startx;
  int z = blockIdx.y * blockDim.y + threadIdx.y + startz;

  // The seed used is based on chunk coordinates, not the world seed
  // see world/gen/ChunkProviderOverworld.java: provideChunk from Mod Coder Pack 9.30
  int64_t seed = (x*341873128712LL + z*132897987541LL)^0x5DEECE66DLL;
  
  for (int a = 0; a < 16; ++a) {
    for(int b = 0; b < 16; ++b) {
      // Do nextInt 250 times and only take the rightmost 48 bits
      seed = seed * 709490313259657689LL + 1748772144486964054LL & 0xFFFFFFFFFFFFLL;
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

      seed = seed * 5985058416696778513LL - 8542997297661424380LL;
    }
  }
  
  // None of the blocks were wrong so the chunk matches the pattern
  printf("Found chunk at %d, %d; real: %d, %d\n", x, z, x*16, z*16);
  return;
}

__global__ void find_sub(int *c, int startx, int startz) {
  int x = blockIdx.x * blockDim.x + startx;
  int z = blockIdx.y * blockDim.y + threadIdx.y + startz;

  int64_t seed = (x*341873128712LL + z*132897987541LL)^0x5DEECE66DLL;

  // Store whether there is bedrock or not
  bool chunk[256];

  for(int a = 0; a < 16; ++a) {
    for(int b = 0; b < 16; ++b) {
      seed = seed * 709490313259657689LL + 1748772144486964054LL & 0xFFFFFFFFFFFFLL;
      if(4 <= (seed >> 17) % 5) {
          chunk[a*16+b] = 1;
      } else {
          chunk[a*16+b] = 0;
      }

      seed = seed * 5985058416696778513LL - 8542997297661424380LL;
    }
  }

  bool match;
  for(int m = 0; m <= 16 - SUBX; ++m) {
    for(int n = 0; n <= 16 - SUBZ; ++n) {
      match = true;
      for(int i = 0; i < SUBX && match == true; ++i) {
        for(int j = 0; j < SUBZ && match == true; ++j) {
#ifdef WILDCARD
          if(c[i*SUBZ+j] != WILDCARD)
#endif
          if(c[i*SUBZ+j] != chunk[(m+i)*16+(n+j)]) {
            match = false;
          }
        }
      }
      if(match) {
        printf("Found chunk at %d, %d; real: %d, %d\n", x, z, x*16, z*16);
        return;
      }
    }
  }

  return;
}

int main() {
  if (blockSize > 1024) {
    printf("blockSize too high! Max is 1024");
    return 1;
  } else if (blockSize < 4) {
    printf("blockSize too low! Min is 458");
    return 1;
  }

  // The whole map is -1875000 to 1875000 on both the x and z axis
  int startx = -5000;
  int endx = 5000;
  int startz = -5000;
  int endz = 5000;
  if (startx < -1875000 || startz < -1875000) {
    printf("startx or startz is too low! Min is -1875000");
    return 1;
  }
  if (endx > 1875000 || endz > 1875000) {
    printf("endx or endz is too high! Max is 1875000");
  }

  dim3 numBlocks(endx-startx, (endz-startz + 1023) / 1024);
  dim3 numThreads(1, blockSize);

  //int *d_full_pattern;
  //cudaMalloc((void**)&d_full_pattern, sizeof(int) * 16 * 16);
  //cudaMemcpy(d_full_pattern, full_pattern, sizeof(int) * 16 * 16, cudaMemcpyHostToDevice);
  //find_full<<<numBlocks, numThreads>>>(d_full_pattern, startx, startz);

  int *d_sub_pattern;
  cudaMalloc((void**)&d_sub_pattern, sizeof(int) * SUBX * SUBZ);
  cudaMemcpy(d_sub_pattern, sub_pattern, sizeof(int) * SUBX * SUBZ, cudaMemcpyHostToDevice);
  find_sub<<<numBlocks, numThreads>>>(d_sub_pattern, startx, startz);

  cudaDeviceSynchronize();

  //cudaFree(d_full_pattern);
  cudaFree(d_sub_pattern);

  return 0;
}
