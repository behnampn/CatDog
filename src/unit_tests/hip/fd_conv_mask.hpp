#include <hip/hip_runtime.h>
#include <hipdnn.h>

#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <unistd.h>

#define HIP_CALL(f) { \
  hipError_t err = (f); \
  if (err != hipSuccess) { \
    std::cout \
        << "ERROR " << __FILE__ << ":" << __LINE__ << " --> " << hipGetErrorString(err) << std::endl; \
    std::exit(1); \
  } \
}

#define HIPDNN_CALL(f) { \
  hipdnnStatus_t err = (f); \
  if (err != HIPDNN_STATUS_SUCCESS) { \
    std::cout \
        << "ERROR " << __FILE__ << ":" << __LINE__ << " --> " << hipdnnGetErrorString(err) << std::endl; \
    std::exit(1); \
  } \
}

void printDevicesInfo(){
  int nDevices;

  HIP_CALL(hipGetDeviceCount(&nDevices));
  for (int i = 0; i < nDevices; i++) {
    hipDeviceProp_t prop;
    HIP_CALL(hipGetDeviceProperties(&prop, i));
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  }
}

__global__ void dev_const(float *px, float k) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  px[tid] = k;
}

__global__ void dev_iota(float *px) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  px[tid] = tid % 10;
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

int main(int argc, char* argv[]) {
  
  printDevicesInfo();
  int deviceNum = 0;
  HIP_CALL(hipSetDevice(deviceNum));
  HIP_CALL(hipGetDevice(&deviceNum));
  std::cout << "Current Device Numer: " << deviceNum << std::endl << std::endl;   

  hipStream_t stream1;
  if (argc < 2)
	hipStreamCreate(&stream1);
  else{
        uint32_t cuMaskSize = 1;
        uint32_t cuMask = atoi(argv[1]);
        hipExtStreamCreateWithCUMask(& stream1, cuMaskSize, &cuMask);
        //hipStreamSetComputeUnitMask(stream1, mask);
  }
     
  hipdnnHandle_t hipdnn;
  HIPDNN_CALL(hipdnnCreate(&hipdnn));

    HIPDNN_CALL(hipdnnSetStream(
    hipdnn,
    stream1));
  // input
  const int in_n = 32;
  const int in_c = 64;
  const int in_h = 64;
  const int in_w = 64;
  std::cout << "in_n: " << in_n << std::endl;
  std::cout << "in_c: " << in_c << std::endl;
  std::cout << "in_h: " << in_h << std::endl;
  std::cout << "in_w: " << in_w << std::endl;
  std::cout << std::endl;

  hipdnnTensorDescriptor_t in_desc;
  HIPDNN_CALL(hipdnnCreateTensorDescriptor(&in_desc));
  HIPDNN_CALL(hipdnnSetTensor4dDescriptor(
        in_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT,
        in_n, in_c, in_h, in_w));

  float *in_data;
  HIP_CALL(hipMalloc(
        &in_data, in_n * in_c * in_h * in_w * sizeof(float)));

  // filter
  const int filt_k = 1;
  const int filt_c = in_c;
  const int filt_h = 3;
  const int filt_w = 3;
  std::cout << "filt_k: " << filt_k << std::endl;
  std::cout << "filt_c: " << filt_c << std::endl;
  std::cout << "filt_h: " << filt_h << std::endl;
  std::cout << "filt_w: " << filt_w << std::endl;
  std::cout << std::endl;

  hipdnnFilterDescriptor_t filt_desc;
  HIPDNN_CALL(hipdnnCreateFilterDescriptor(&filt_desc));
  int filterDimA[] = {filt_k, filt_c, filt_h, filt_w};
  HIPDNN_CALL(hipdnnSetFilterNdDescriptor(
        filt_desc, HIPDNN_DATA_FLOAT, HIPDNN_TENSOR_NCHW,
        4, filterDimA));

  float *filt_data;
  HIP_CALL(hipMalloc(
      &filt_data, filt_k * filt_c * filt_h * filt_w * sizeof(float)));

  // convolution
  const int pad_h = 1;
  const int pad_w = 1;
  const int str_h = 1;
  const int str_w = 1;
  const int dil_h = 1;
  const int dil_w = 1;
  std::cout << "pad_h: " << pad_h << std::endl;
  std::cout << "pad_w: " << pad_w << std::endl;
  std::cout << "str_h: " << str_h << std::endl;
  std::cout << "str_w: " << str_w << std::endl;
  std::cout << "dil_h: " << dil_h << std::endl;
  std::cout << "dil_w: " << dil_w << std::endl;
  std::cout << std::endl;

  hipdnnConvolutionDescriptor_t conv_desc;
  HIPDNN_CALL(hipdnnCreateConvolutionDescriptor(&conv_desc));
  HIPDNN_CALL(hipdnnSetConvolution2dDescriptor(
        conv_desc,
        pad_h, pad_w, str_h, str_w, dil_h, dil_w,
        HIPDNN_CROSS_CORRELATION, HIPDNN_DATA_FLOAT));

  // output
  int out_n;
  int out_c;
  int out_h;
  int out_w;
  
  HIPDNN_CALL(hipdnnGetConvolution2dForwardOutputDim(
        conv_desc, in_desc, filt_desc,
        &out_n, &out_c, &out_h, &out_w));

  std::cout << "out_n: " << out_n << std::endl;
  std::cout << "out_c: " << out_c << std::endl;
  std::cout << "out_h: " << out_h << std::endl;
  std::cout << "out_w: " << out_w << std::endl;
  std::cout << std::endl;

  hipdnnTensorDescriptor_t out_desc;
  HIPDNN_CALL(hipdnnCreateTensorDescriptor(&out_desc));
  HIPDNN_CALL(hipdnnSetTensor4dDescriptor(
        out_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT,
        out_n, out_c, out_h, out_w));

  float *out_data;
  HIP_CALL(hipMalloc(
        &out_data, out_n * out_c * out_h * out_w * sizeof(float)));

  // algorithm
  hipdnnConvolutionFwdAlgo_t algo;
  HIPDNN_CALL(hipdnnGetConvolutionForwardAlgorithm(
        hipdnn,
        in_desc, filt_desc, conv_desc, out_desc,
        HIPDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));

  //algo = HIPDNN_CONVOLUTION_FWD_ALGO_FFT;

  std::cout << "Convolution algorithm: " << algo << std::endl;
  std::cout << std::endl;

  // workspace
  size_t ws_size;
  HIPDNN_CALL(hipdnnGetConvolutionForwardWorkspaceSize(
        hipdnn, in_desc, filt_desc, conv_desc, out_desc, algo, &ws_size));

  float *ws_data;
  HIP_CALL(hipMalloc(&ws_data, ws_size));

  std::cout << "Workspace size: " << ws_size << std::endl;
  std::cout << std::endl;

  // perform
  float alpha = 1.f;
  float beta = 0.f;
  dev_iota<<<in_w * in_h, in_n * in_c>>>(in_data);
  dev_const<<<filt_w * filt_h, filt_k * filt_c>>>(filt_data, 1.f);
  HIPDNN_CALL(hipdnnConvolutionForward(
      hipdnn,
      &alpha, in_desc, in_data, filt_desc, filt_data,
      conv_desc, algo, ws_data, ws_size,
      &beta, out_desc, out_data));


  // results
  std::cout << "in_data:" << std::endl;
  //print(in_data, in_n, in_c, in_h, in_w);
  
  std::cout << "filt_data:" << std::endl;
  //print(filt_data, filt_k, filt_c, filt_h, filt_w);
  
  std::cout << "out_data:" << std::endl;
  //print(out_data, out_n, out_c, out_h, out_w);

  // finalizing
  HIP_CALL(hipFree(ws_data));
  HIP_CALL(hipFree(out_data));
  HIPDNN_CALL(hipdnnDestroyTensorDescriptor(out_desc));
  HIPDNN_CALL(hipdnnDestroyConvolutionDescriptor(conv_desc));
  HIP_CALL(hipFree(filt_data));
  HIPDNN_CALL(hipdnnDestroyFilterDescriptor(filt_desc));
  HIP_CALL(hipFree(in_data));
  HIPDNN_CALL(hipdnnDestroyTensorDescriptor(in_desc));
  HIPDNN_CALL(hipdnnDestroy(hipdnn));
  return 0;
}

