//****************************************************************************
//
// Array of weights:
//
//  0.0  0.2  0.0
//  0.2  0.2  0.2
//  0.0  0.2  0.0
//
// Image (note that we align the array of weights to the center of the box):
//
//    1  2  5  2  0  3
//       -------
//    3 |2  5  1| 6  0       0.0*2 + 0.2*5 + 0.0*1 +
//      |       |
//    4 |3  6  2| 1  4   ->  0.2*3 + 0.2*6 + 0.2*2 +   ->  3.2
//      |       |
//    0 |4  0  3| 4  2       0.0*4 + 0.2*0 + 0.0*3
//       -------
//    9  6  5  0  3  9
//
//         (1)                         (2)                 (3)
//
//****************************************************************************

#include "utils.h"

// GLOBAL VARS ON DEVICE
unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;

// devicel functions:
__device__ int min_int(int a, int b){
  if (a <= b) return a;
  else return b;
}
__device__ int max_int(int a, int b) {
  if (a>=b) return a;
  else return b;
}

__global__
void gaussian_blur(const unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth)
{
  const int2 thread_2D_pos = make_int2(blockIdx.x*blockDim.x + threadIdx.x,
                                        blockIdx.y*blockDim.y + threadIdx.y);
  // don't access memory outside of bounds
  if(thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows) return;
  const int thread_1D_pos = thread_2D_pos.y*numCols + thread_2D_pos.x;


  float result = 0.f;
  //For every value in the filter around the pixel (c, r)
  for (int filter_r = -filterWidth/2; filter_r <= filterWidth/2; ++filter_r) {
    for (int filter_c = -filterWidth/2; filter_c <= filterWidth/2; ++filter_c) {
      //Find the global image position for this filter position
      //clamp to boundary of the image
      int image_r = min_int(max_int(thread_2D_pos.y + filter_r, 0), static_cast<int>(numRows - 1));
      int image_c = min_int(max_int(thread_2D_pos.x + filter_c, 0), static_cast<int>(numCols - 1));

      if (thread_1D_pos == 0) {
        printf("filter_r: %d, filter_c: %d, image_r: %d, image_c: %d \n",filter_r,filter_c,image_r,image_c);
      }

      float image_value = static_cast<float>(inputChannel[image_r * numCols + image_c]);
      float filter_value = filter[(filter_r + filterWidth/2) * filterWidth + filter_c + filterWidth/2];

      result += image_value * filter_value;
    }
  }

  outputChannel[thread_1D_pos] = result;

}

//This kernel takes in an image represented as a uchar4 and splits
//it into three images consisting of only one color channel each
__global__
void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows,
                      int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{


  const int2 thread_2D_pos = make_int2(blockIdx.x*blockDim.x + threadIdx.x,
                                       blockIdx.y*blockDim.y + threadIdx.y);
  // don't access memory outside of bounds
  if(thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows) return;

  const int thread_1D_pos = thread_2D_pos.y*numCols + thread_2D_pos.x;

  redChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].x;
  greenChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].y;
  blueChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].z;

}

//This kernel takes in three color channels and recombines them
//into one image.  The alpha channel is set to 255 to represent
//that this image has no transparency.
__global__
void recombineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows,
                       int numCols)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  //don't try and access memory outside the image
  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  unsigned char red   = redChannel[thread_1D_pos];
  unsigned char green = greenChannel[thread_1D_pos];
  unsigned char blue  = blueChannel[thread_1D_pos];

  //Alpha should be 255 for no transparency
  uchar4 outputPixel = make_uchar4(red, green, blue, 255);

  outputImageRGBA[thread_1D_pos] = outputPixel;
}


void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth)
{

  //allocate memory for the three different channels
  checkCudaErrors(cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage));

  //allocate memory for filter and copy from host to device ptr.
  checkCudaErrors(cudaMalloc((void**)&d_filter,sizeof(float)*filterWidth*filterWidth));
  checkCudaErrors(cudaMemcpy(d_filter,h_filter,sizeof(float)*filterWidth*filterWidth,
                             cudaMemcpyHostToDevice));
}

void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        unsigned char *d_redBlurred,
                        unsigned char *d_greenBlurred,
                        unsigned char *d_blueBlurred,
                        const int filterWidth)
{

  // printf("numRows: %lu, numCols: %lu\n",numRows,numCols);

  //Set reasonable block size (number of threads per block)
  const dim3 blockSize(20,20,1);  // 20*20 = 400 threads per block

  //Compute correct grid size (i.e., number of blocks per kernel launch)
  //from the image size and and block size.
  const dim3 gridSize(numCols/blockSize.x+1,numRows/blockSize.y+1,1);

  //printf("block size x: %d, y: %d \n",blockSize.x,blockSize.y);
  //printf("grid size x: %d, y: %d \n",gridSize.x,gridSize.y);

  //separating the RGBA image into different color channels
  separateChannels<<<gridSize,blockSize>>>(d_inputImageRGBA,numRows,numCols,
                                           d_red,d_green,d_blue);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  //for each color channel
  gaussian_blur<<<gridSize,blockSize>>>(d_red,d_redBlurred,numRows,numCols,
                                        d_filter,filterWidth);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  gaussian_blur<<<gridSize,blockSize>>>(d_green,d_greenBlurred,numRows,numCols,
                                        d_filter,filterWidth);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  gaussian_blur<<<gridSize,blockSize>>>(d_blue,d_blueBlurred,numRows,numCols,
                                        d_filter,filterWidth);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  recombineChannels<<<gridSize, blockSize>>>(d_redBlurred,
                                             d_greenBlurred,
                                             d_blueBlurred,
                                             d_outputImageRGBA,
                                             numRows,
                                             numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

}


//Free all the memory that we allocated
void cleanup() {
  checkCudaErrors(cudaFree(d_red));
  checkCudaErrors(cudaFree(d_green));
  checkCudaErrors(cudaFree(d_blue));
  checkCudaErrors(cudaFree(d_filter));
}
