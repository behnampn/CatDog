#include <iomanip>
#include <chrono>
#include <ctime>
#include <ratio>
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <unistd.h>
#include <bitset>

#include <hip/hip_runtime.h>
#include <hipdnn.h>

#define HIP_CALL(f) { \
  hipError_t err = (f); \
  if (err != hipSuccess) { \
    std::cout \
        << "    Error occurred: " << err << std::endl; \
    std::exit(1); \
  } \
}

#define HIPDNN_CALL(f) { \
  hipdnnStatus_t err = (f); \
  if (err != HIPDNN_STATUS_SUCCESS) { \
    std::cout \
        << "    Error occurred: " << err << std::endl; \
    std::exit(1); \
  } \
}

__global__ void dev_const(float *px, float k) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  px[tid] = k;
}

__global__ void dev_iota(float *px) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  px[tid] = tid % 10;
}

void printCurrDeviceInfo(){
  int id;
  HIP_CALL(hipGetDevice(&id));
  hipDeviceProp_t prop;
  HIP_CALL(hipGetDeviceProperties(&prop, id));
  printf("Device id: %d\n", id);
  printf("Device name: %s\n", prop.name);
  printf("SM/CU counts: %d\n", prop.multiProcessorCount);
  printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
  printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
  printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
}

void print(const float *data, int n, int c, int h, int w) {
  std::vector<float> buffer(1 << 20);
  HIP_CALL(hipMemcpy(
        buffer.data(), data,
        n * c * h * w * sizeof(float),
        hipMemcpyDeviceToHost));
  int a = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < c; ++j) {
      std::cout << "n=" << i << ", c=" << j << ":" << std::endl;
      for (int k = 0; k < h; ++k) {
        for (int l = 0; l < w; ++l) {
          std::cout << std::setw(4) << std::right << buffer[a];
          ++a;
        }
        std::cout << std::endl;
      }
    }
  }
  std::cout << std::endl;
}

void printFirstKElements(const float *data, int n, int c, int h, int w, int K) {
  std::vector<float> buffer(1 << 20);
  HIP_CALL(hipMemcpy(
        buffer.data(), data,
        n * c * h * w * sizeof(float),
        hipMemcpyDeviceToHost));
  for (int i = 0; i < K; ++i) 
    std::cout << std::setw(4) << " "  << buffer[i];
  std::cout << std::endl;
}

int main(int argc, char* argv[]) {
  printCurrDeviceInfo();
  
  hipStream_t stream1, stream2;
  if (argc != 3){
    hipStreamCreateWithFlags( &stream1, hipStreamNonBlocking) ;
    hipStreamCreateWithFlags( &stream2, hipStreamNonBlocking) ;
  }
  else{
    int id;
    HIP_CALL(hipGetDevice(&id));
    hipDeviceProp_t prop;
    HIP_CALL(hipGetDeviceProperties(&prop, id));
    int multiProcessorCount = prop.multiProcessorCount;
    int SM1_num = atoi(argv[1]);
    int SM2_num = atoi(argv[2]);    
    if (SM1_num + SM2_num > multiProcessorCount){
      std::cout<< "ERROR: total number of multiProcessor should be less than " << multiProcessorCount << std::endl;
      return -1;
    } 
    uint64_t cuMaskSize = 2;
    uint64_t cuMaskL1 = (0x1L << SM1_num)-1;
    std::bitset<64> nbit1(cuMaskL1);
    uint64_t cuMaskL2 = ((0x1L << SM2_num) - 1) << nbit1.count();
    uint32_t cuMask1[2];
    uint32_t cuMask2[2];
    cuMask1[0] = cuMaskL1 & 0xFFFFFFFF;
    cuMask2[0] = cuMaskL2 & 0xFFFFFFFF;
    cuMask1[1] = cuMaskL1 >> 32;
    cuMask2[1] = cuMaskL2 >> 32;
    std::cout<<cuMaskL1<<" "<<cuMaskL2<<std::endl;
    std::cout<<cuMask1[0]<<" "<<cuMask1[1]<<" "<<cuMask2[0]<<" "<<cuMask2[1]<<" "<<std::endl;
    hipExtStreamCreateWithCUMask(& stream1, cuMaskSize, cuMask1);
    hipExtStreamCreateWithCUMask(& stream2, cuMaskSize, cuMask2);
  }

  hipdnnHandle_t hipdnn1, hipdnn2;
  HIPDNN_CALL(hipdnnCreate(&hipdnn1));
  HIPDNN_CALL(hipdnnCreate(&hipdnn2));

  HIPDNN_CALL(hipdnnSetStream(
    hipdnn1,
    stream1));
  HIPDNN_CALL(hipdnnSetStream(
    hipdnn2,
    stream2));

  // input
  const int in1_n = 32;
  const int in1_c = 5;
  const int in1_h = 100;
  const int in1_w = 100;
  std::cout << "in1_n: " << in1_n << std::endl;
  std::cout << "in1_c: " << in1_c << std::endl;
  std::cout << "in1_h: " << in1_h << std::endl;
  std::cout << "in1_w: " << in1_w << std::endl;
  std::cout << std::endl;

  hipdnnTensorDescriptor_t in1_desc;
  HIPDNN_CALL(hipdnnCreateTensorDescriptor(&in1_desc));
  HIPDNN_CALL(hipdnnSetTensor4dDescriptor(
        in1_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT,
        in1_n, in1_c, in1_h, in1_w));

  float *in1_data;
  HIP_CALL(hipMalloc(
        &in1_data, in1_n * in1_c * in1_h * in1_w * sizeof(float)));

  const int in2_n = 32;
  const int in2_c = 5;
  const int in2_h = 100;
  const int in2_w = 100;
  std::cout << "in2_n: " << in2_n << std::endl;
  std::cout << "in2_c: " << in2_c << std::endl;
  std::cout << "in2_h: " << in2_h << std::endl;
  std::cout << "in2_w: " << in2_w << std::endl;
  std::cout << std::endl;

  hipdnnTensorDescriptor_t in2_desc;
  HIPDNN_CALL(hipdnnCreateTensorDescriptor(&in2_desc));
  HIPDNN_CALL(hipdnnSetTensor4dDescriptor(
        in2_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT,
        in2_n, in2_c, in2_h, in2_w));

  float *in2_data;
  HIP_CALL(hipMalloc(
        &in2_data, in2_n * in2_c * in2_h * in2_w * sizeof(float)));

  // filter
  const int filt1_k = 10;
  const int filt1_c = in1_c;
  const int filt1_h = 3;
  const int filt1_w = 3;
  std::cout << "filt1_k: " << filt1_k << std::endl;
  std::cout << "filt1_c: " << filt1_c << std::endl;
  std::cout << "filt1_h: " << filt1_h << std::endl;
  std::cout << "filt1_w: " << filt1_w << std::endl;
  std::cout << std::endl;

  hipdnnFilterDescriptor_t filt1_desc;
  HIPDNN_CALL(hipdnnCreateFilterDescriptor(&filt1_desc));
  int filterDim1[] = {filt1_k, filt1_c, filt1_h, filt1_w};
  HIPDNN_CALL(hipdnnSetFilterNdDescriptor(
        filt1_desc, HIPDNN_DATA_FLOAT, HIPDNN_TENSOR_NCHW,
        4, filterDim1));

  float *filt1_data;
  HIP_CALL(hipMalloc(
      &filt1_data, filt1_k * filt1_c * filt1_h * filt1_w * sizeof(float)));

  const int filt2_k = 10;
  const int filt2_c = in2_c;
  const int filt2_h = 3;
  const int filt2_w = 3;
  std::cout << "filt2_k: " << filt2_k << std::endl;
  std::cout << "filt2_c: " << filt2_c << std::endl;
  std::cout << "filt2_h: " << filt2_h << std::endl;
  std::cout << "filt2_w: " << filt2_w << std::endl;
  std::cout << std::endl;

  hipdnnFilterDescriptor_t filt2_desc;
  HIPDNN_CALL(hipdnnCreateFilterDescriptor(&filt2_desc));
  int filterDim2[] = {filt2_k, filt2_c, filt2_h, filt2_w};
  HIPDNN_CALL(hipdnnSetFilterNdDescriptor(
        filt2_desc, HIPDNN_DATA_FLOAT, HIPDNN_TENSOR_NCHW,
        4, filterDim2));

  float *filt2_data;
  HIP_CALL(hipMalloc(
      &filt2_data, filt2_k * filt2_c * filt2_h * filt2_w * sizeof(float)));

  // convolution
  const int pad1_h = 1;
  const int pad1_w = 1;
  const int str1_h = 1;
  const int str1_w = 1;
  const int dil1_h = 1;
  const int dil1_w = 1;
  std::cout << "pad1_h: " << pad1_h << std::endl;
  std::cout << "pad1_w: " << pad1_w << std::endl;
  std::cout << "str1_h: " << str1_h << std::endl;
  std::cout << "str1_w: " << str1_w << std::endl;
  std::cout << "dil1_h: " << dil1_h << std::endl;
  std::cout << "dil1_w: " << dil1_w << std::endl;
  std::cout << std::endl;

  hipdnnConvolutionDescriptor_t conv1_desc;
  HIPDNN_CALL(hipdnnCreateConvolutionDescriptor(&conv1_desc));
  // hipdnnSetConvolution2dDescriptor_v5(...)
  HIPDNN_CALL(hipdnnSetConvolution2dDescriptor(
        conv1_desc,
        pad1_h, pad1_w, str1_h, str1_w, dil1_h, dil1_w,
        HIPDNN_CROSS_CORRELATION, HIPDNN_DATA_FLOAT));

  const int pad2_h = 1;
  const int pad2_w = 1;
  const int str2_h = 1;
  const int str2_w = 1;
  const int dil2_h = 1;
  const int dil2_w = 1;
  std::cout << "pad2_h: " << pad2_h << std::endl;
  std::cout << "pad2_w: " << pad2_w << std::endl;
  std::cout << "str2_h: " << str2_h << std::endl;
  std::cout << "str2_w: " << str2_w << std::endl;
  std::cout << "dil2_h: " << dil2_h << std::endl;
  std::cout << "dil2_w: " << dil2_w << std::endl;
  std::cout << std::endl;

  hipdnnConvolutionDescriptor_t conv2_desc;
  HIPDNN_CALL(hipdnnCreateConvolutionDescriptor(&conv2_desc));
  // hipdnnSetConvolution2dDescriptor_v5(...)
  HIPDNN_CALL(hipdnnSetConvolution2dDescriptor(
        conv2_desc,
        pad2_h, pad2_w, str2_h, str2_w, dil2_h, dil2_w,
        HIPDNN_CROSS_CORRELATION, HIPDNN_DATA_FLOAT));

  // output
  int out1_n;
  int out1_c;
  int out1_h;
  int out1_w;
  
  HIPDNN_CALL(hipdnnGetConvolution2dForwardOutputDim(
        conv1_desc, in1_desc, filt1_desc,
        &out1_n, &out1_c, &out1_h, &out1_w));

  std::cout << "out1_n: " << out1_n << std::endl;
  std::cout << "out1_c: " << out1_c << std::endl;
  std::cout << "out1_h: " << out1_h << std::endl;
  std::cout << "out1_w: " << out1_w << std::endl;
  std::cout << std::endl;

  hipdnnTensorDescriptor_t out1_desc;
  HIPDNN_CALL(hipdnnCreateTensorDescriptor(&out1_desc));
  HIPDNN_CALL(hipdnnSetTensor4dDescriptor(
        out1_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT,
        out1_n, out1_c, out1_h, out1_w));

  float *out1_data;
  HIP_CALL(hipMalloc(
        &out1_data, out1_n * out1_c * out1_h * out1_w * sizeof(float)));

  int out2_n;
  int out2_c;
  int out2_h;
  int out2_w;
  
  HIPDNN_CALL(hipdnnGetConvolution2dForwardOutputDim(
        conv2_desc, in2_desc, filt2_desc,
        &out2_n, &out2_c, &out2_h, &out2_w));

  std::cout << "out1_n: " << out2_n << std::endl;
  std::cout << "out1_c: " << out2_c << std::endl;
  std::cout << "out1_h: " << out2_h << std::endl;
  std::cout << "out1_w: " << out2_w << std::endl;
  std::cout << std::endl;

  hipdnnTensorDescriptor_t out2_desc;
  HIPDNN_CALL(hipdnnCreateTensorDescriptor(&out2_desc));
  HIPDNN_CALL(hipdnnSetTensor4dDescriptor(
        out2_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT,
        out2_n, out2_c, out2_h, out2_w));

  float *out2_data;
  HIP_CALL(hipMalloc(
        &out2_data, out2_n * out2_c * out2_h * out2_w * sizeof(float)));

  // algorithm
  hipdnnConvolutionFwdAlgo_t algo1, algo2;
  HIPDNN_CALL(hipdnnGetConvolutionForwardAlgorithm(
        hipdnn1,
        in1_desc, filt1_desc, conv1_desc, out1_desc,
        HIPDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo1));
  HIPDNN_CALL(hipdnnGetConvolutionForwardAlgorithm(
        hipdnn2,
        in2_desc, filt2_desc, conv2_desc, out2_desc,
        HIPDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo2));

  //algo = HIPDNN_CONVOLUTION_FWD_ALGO_FFT;

  std::cout << "Convolution algorithm stream1: " << algo1 << std::endl;
  std::cout << "Convolution algorithm stream2: " << algo2 << std::endl;
  std::cout << std::endl;

  // workspace
  size_t ws1_size, ws2_size;
  HIPDNN_CALL(hipdnnGetConvolutionForwardWorkspaceSize(
        hipdnn1, in1_desc, filt1_desc, conv1_desc, out1_desc, algo1, &ws1_size));
  HIPDNN_CALL(hipdnnGetConvolutionForwardWorkspaceSize(
        hipdnn2, in2_desc, filt2_desc, conv2_desc, out2_desc, algo2, &ws2_size));

  float *ws1_data, *ws2_data;
  HIP_CALL(hipMalloc(&ws1_data, ws1_size));
  HIP_CALL(hipMalloc(&ws2_data, ws2_size));

  std::cout << "Workspace size stream1: " << ws1_size << std::endl;
  std::cout << "Workspace size stream2: " << ws2_size << std::endl;
  std::cout << std::endl;

  // perform
  float alpha1 = 1.f;
  float alpha2 = 1.f;
  float beta1 = 0.f;
  float beta2 = 0.f;
  dev_iota<<<in1_w * in1_h, in1_n * in1_c, 0, stream1>>>(in1_data);
  dev_iota<<<in2_w * in2_h, in2_n * in2_c, 0, stream2>>>(in2_data);
  dev_const<<<filt1_w * filt1_h, filt1_k * filt1_c, 0, stream1>>>(filt1_data, 1.f);
  dev_const<<<filt2_w * filt2_h, filt2_k * filt2_c, 0, stream2>>>(filt1_data, 1.f);
  //dev_iota<<<filt1_w * filt1_h, filt1_k * filt1_c, 0, stream1>>>(filt1_data);
  //dev_iota<<<filt2_w * filt2_h, filt2_k * filt2_c, 0, stream2>>>(filt1_data);
  HIP_CALL(hipDeviceSynchronize());

  int conv_repeat = 20;
  typedef std::chrono::high_resolution_clock Clock;
  Clock::time_point t_start = Clock::now();
  for (int i = 0; i < conv_repeat; i++){
    HIPDNN_CALL(hipdnnConvolutionForward(
        hipdnn1,
        &alpha1, in1_desc, in1_data, filt1_desc, filt1_data,
        conv1_desc, algo1, ws1_data, ws1_size,
        &beta1, out1_desc, out1_data));
    HIPDNN_CALL(hipdnnConvolutionForward(
        hipdnn2,
        &alpha2, in2_desc, in2_data, filt2_desc, filt2_data,
        conv2_desc, algo2, ws2_data, ws2_size,
        &beta2, out2_desc, out2_data));
    HIP_CALL(hipDeviceSynchronize());
  }
  HIP_CALL(hipDeviceSynchronize());
  Clock::time_point t_final = Clock::now();
  std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t_final - t_start);
  // results
  std::cout << "total conv duration:" << time_span.count() << " seconds" << std::endl;

  std::cout << "in_data:" << std::endl;
  //print(in1_data, in1_n, in1_c, in1_h, in1_w);
  
  std::cout << "filt_data:" << std::endl;
  //print(filt1_data, filt1_k, filt1_c, filt1_h, filt1_w);
  
  std::cout << "out_data:" << std::endl;
  //print(out1_data, out1_n, out1_c, out1_h, out1_w);
  //printFirstKElements(out1_data, out1_n, out1_c, out1_h, out1_w, 20);

  // finalizing
  HIP_CALL(hipFree(ws1_data));
  HIP_CALL(hipFree(ws2_data));
  HIP_CALL(hipFree(out1_data));
  HIP_CALL(hipFree(out2_data));
  HIPDNN_CALL(hipdnnDestroyTensorDescriptor(out1_desc));
  HIPDNN_CALL(hipdnnDestroyTensorDescriptor(out2_desc));
  HIPDNN_CALL(hipdnnDestroyConvolutionDescriptor(conv1_desc));
  HIPDNN_CALL(hipdnnDestroyConvolutionDescriptor(conv2_desc));
  HIP_CALL(hipFree(filt1_data));
  HIP_CALL(hipFree(filt2_data));
  HIPDNN_CALL(hipdnnDestroyFilterDescriptor(filt1_desc));
  HIPDNN_CALL(hipdnnDestroyFilterDescriptor(filt2_desc));
  HIP_CALL(hipFree(in1_data));
  HIP_CALL(hipFree(in2_data));
  HIPDNN_CALL(hipdnnDestroyTensorDescriptor(in1_desc));
  HIPDNN_CALL(hipdnnDestroyTensorDescriptor(in2_desc));
  HIPDNN_CALL(hipdnnDestroy(hipdnn1));
  HIPDNN_CALL(hipdnnDestroy(hipdnn2));
  return 0;
}
