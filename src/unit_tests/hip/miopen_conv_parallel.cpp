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
//#include <hipdnn.h>
#include <miopen/miopen.h>

#define HIP_CALL(f) { \
  hipError_t err = (f); \
  if (err != hipSuccess) { \
    std::cout \
        << "    Error occurred: " << err << std::endl; \
    std::exit(1); \
  } \
}

#define MI_CALL(f) { \
  miopenStatus_t err = (f); \
  if (err != miopenStatusSuccess) { \
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
          //std::cout << std::setw(4) << std::right << buffer[a];
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
    //std::cout << std::setw(4) << " "  << buffer[i];
  std::cout << std::endl;
}

int main(int argc, char* argv[]) {
  printCurrDeviceInfo();

  //create stream without cu masking if no args prvoided at command line  
  hipStream_t stream1, stream2;
  if (argc != 5 || argc!=3){
    hipStreamCreateWithFlags( &stream1, hipStreamNonBlocking) ;
    hipStreamCreateWithFlags( &stream2, hipStreamNonBlocking) ;
  }

	//create stream with cu masking with cu mask provided in command line
  else
  {
    int id;
    HIP_CALL(hipGetDevice(&id));
    hipDeviceProp_t prop;
    HIP_CALL(hipGetDeviceProperties(&prop, id));
    int multiProcessorCount = prop.multiProcessorCount;
    int SM1_num = atoi(argv[1]);
    int SM2_num = atoi(argv[2]);    
    std::cout<<"Input 1 = "<<argv[1]<<std::endl;
    std::cout<<"Input 2 = "<<argv[2]<<std::endl;
    if (SM1_num + SM2_num > multiProcessorCount)
		{
      std::cout<< "ERROR: total number of multiProcessor should be less than " << multiProcessorCount << std::endl;
      return -1;
    } 
    uint64_t cuMaskSize = 2;
    uint64_t cuMaskL1 = (0x1L << SM1_num)-1;
    std::bitset<64> nbit1(cuMaskL1);
    uint64_t cuMaskL2 = ((0x1L << SM2_num) - 1);
    uint32_t cuMask1[2];
    uint32_t cuMask2[2];
    cuMask1[0] = (cuMaskL1 & 0xFFFFFFFF);
    cuMask2[0] = (cuMaskL2 & 0xFFFFFFFF);
    cuMask1[1] = cuMaskL1 >> 31;
    cuMask2[1] = cuMaskL2 >> 31;
    std::cout<<"cu mask L1 = "<<cuMaskL1<<", cu mask L2= "<<cuMaskL2<<std::endl;
    std::cout<<cuMask1[0]<<" "<<cuMask1[1]<<" "<<cuMask2[0]<<" "<<cuMask2[1]<<" "<<std::endl;
    hipExtStreamCreateWithCUMask(& stream1, cuMaskSize, cuMask1);
    hipExtStreamCreateWithCUMask(& stream2, cuMaskSize, cuMask2);
  }

  int a1,a2;
  if(argc!=5)
  {
      a1 = 0;
      a2 = 0;
  }
  else
  {
      a1 = atoi(argv[3]);
      a2 = atoi(argv[4]);
  }
  miopenHandle_t hipdnn1, hipdnn2;
  MI_CALL(miopenCreate(&hipdnn1));
  MI_CALL(miopenCreate(&hipdnn2));

	//set stream in the hipdnn handle
  MI_CALL(miopenSetStream(
    hipdnn1,
    stream1));
  MI_CALL(miopenSetStream(
    hipdnn2,
    stream2));
	
  //input
  const int in1_n = 32;
  const int in1_c = 5;
  const int in1_h = 64;
  const int in1_w = 64;
  std::cout << "in1_n: " << in1_n << std::endl;
  std::cout << "in1_c: " << in1_c << std::endl;
  std::cout << "in1_h: " << in1_h << std::endl;
  std::cout << "in1_w: " << in1_w << std::endl;
  std::cout << std::endl;

	//create a generic tensor descriptor object by allocating memory needed to hold the data structure
  miopenTensorDescriptor_t in1_desc;
  MI_CALL(miopenCreateTensorDescriptor(&in1_desc));

	//set tensor 4d descriptor initializes the previously created generic tensor descriptor into a 4D tensor. arguments - (tensorDesc, format type, datatype, n,c,h,w)
  MI_CALL(miopenSet4dTensorDescriptor(
        in1_desc, miopenHalf,
        in1_n, in1_c, in1_h, in1_w));

	//allocate space for in1 on gpu
  float *in1_data;
  HIP_CALL(hipMalloc(
        &in1_data, in1_n * in1_c * in1_h * in1_w * sizeof(float)));

  // filter
	//k= number of feature maps/filters = number of output channels
  const int filt1_k = 10;
  const int filt1_c = in1_c;
  const int filt1_h = 3;
  const int filt1_w = 3;
  std::cout << "filt1_k: " << filt1_k << std::endl;
  std::cout << "filt1_c: " << filt1_c << std::endl;
  std::cout << "filt1_h: " << filt1_h << std::endl;
  std::cout << "filt1_w: " << filt1_w << std::endl;
  std::cout << std::endl;

  miopenTensorDescriptor_t filt1_desc;
  MI_CALL(miopenCreateTensorDescriptor(&filt1_desc));
  MI_CALL(miopenSet4dTensorDescriptor(
        filt1_desc, miopenHalf, 
	filt1_k, filt1_c, filt1_h, filt1_w));

  float *filt1_data;
  HIP_CALL(hipMalloc(
      &filt1_data, filt1_k * filt1_c * filt1_h * filt1_w * sizeof(float)));

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

  miopenConvolutionDescriptor_t conv1_desc;
  MI_CALL(miopenCreateConvolutionDescriptor(&conv1_desc));
  MI_CALL(miopenInitConvolutionDescriptor(
        conv1_desc, miopenConvolution,
        pad1_h, pad1_w, str1_h, str1_w, dil1_h, dil1_w));

  // output
  int out1_n;
  int out1_c;
  int out1_h;
  int out1_w;
  
  MI_CALL(miopenGetConvolutionForwardOutputDim(
        conv1_desc, in1_desc, filt1_desc,
        &out1_n, &out1_c, &out1_h, &out1_w));

  std::cout << "out1_n: " << out1_n << std::endl;
  std::cout << "out1_c: " << out1_c << std::endl;
  std::cout << "out1_h: " << out1_h << std::endl;
  std::cout << "out1_w: " << out1_w << std::endl;
  std::cout << std::endl;

  miopenTensorDescriptor_t out1_desc;
  MI_CALL(miopenCreateTensorDescriptor(&out1_desc));
  MI_CALL(miopenSet4dTensorDescriptor(
        out1_desc, miopenHalf,
        out1_n, out1_c, out1_h, out1_w));

  float *out1_data;
  HIP_CALL(hipMalloc(
        &out1_data, out1_n * out1_c * out1_h * out1_w * sizeof(float)));

  // workspace
  size_t workspaceSize1;
  MI_CALL(miopenConvolutionForwardGetWorkSpaceSize(
        hipdnn1, filt1_desc, in1_desc, conv1_desc, out1_desc, &workspaceSize1));
  float *workspace1;
  HIP_CALL(hipMalloc(&workspace1, workspaceSize1));

  std::cout<<"Workspace size = "<<workspaceSize1<<std::endl;
  // algorithm
  int requestAlgoCount1 = 6;
  int returnedAlgoCount1;
  bool exhaustiveSearch1 = 0;
  miopenConvAlgoPerf_t perfResults1[3];
  MI_CALL(miopenFindConvolutionForwardAlgorithm(
        hipdnn1,
        in1_desc, in1_data,
	filt1_desc, filt1_data,
	conv1_desc, out1_desc, out1_data,
	requestAlgoCount1, &returnedAlgoCount1,
	perfResults1, workspace1, workspaceSize1,
	exhaustiveSearch1));
  std::cout<<"Returned Algo Count: "<<returnedAlgoCount1<<std::endl;        
  for(int i=0;i<returnedAlgoCount1;i++)
  {
  	std::cout<<"Algorithm: "<<perfResults1[i].fwd_algo<<"\t";
  	std::cout<<"Memory: "<<perfResults1[i].memory<<"\t";
  	std::cout<<"Time: "<<perfResults1[i].time<<std::endl;
  }
	//2nd Convolution
  const int in2_n = 32;
  const int in2_c = 5;
  const int in2_h = 64;
  const int in2_w = 64;
  std::cout << "in2_n: " << in2_n << std::endl;
  std::cout << "in2_c: " << in2_c << std::endl;
  std::cout << "in2_h: " << in2_h << std::endl;
  std::cout << "in2_w: " << in2_w << std::endl;
  std::cout << std::endl;

	//create a generic tensor descriptor object by allocating memory needed to hold the data structure
  miopenTensorDescriptor_t in2_desc;
  MI_CALL(miopenCreateTensorDescriptor(&in2_desc));

	//set tensor 4d descriptor initializes the previously created generic tensor descriptor into a 4D tensor. arguments - (tensorDesc, format type, datatype, n,c,h,w)
  MI_CALL(miopenSet4dTensorDescriptor(
        in2_desc, miopenHalf,
        in2_n, in2_c, in2_h, in2_w));

	//allocate space for in1 on gpu
  float *in2_data;
  HIP_CALL(hipMalloc(
        &in2_data, in2_n * in2_c * in2_h * in2_w * sizeof(float)));

  // filter
	//k= number of feature maps/filters = number of output channels
  const int filt2_k = 10;
  const int filt2_c = in2_c;
  const int filt2_h = 3;
  const int filt2_w = 3;
  std::cout << "filt2_k: " << filt2_k << std::endl;
  std::cout << "filt2_c: " << filt2_c << std::endl;
  std::cout << "filt2_h: " << filt2_h << std::endl;
  std::cout << "filt2_w: " << filt2_w << std::endl;
  std::cout << std::endl;

  miopenTensorDescriptor_t filt2_desc;
  MI_CALL(miopenCreateTensorDescriptor(&filt2_desc));
  //int filterDim1[] = {filt1_k, filt1_c, filt1_h, filt1_w};
  MI_CALL(miopenSet4dTensorDescriptor(
        filt2_desc, miopenHalf, 
	filt2_k, filt2_c, filt2_h, filt2_w));

  float *filt2_data;
  HIP_CALL(hipMalloc(
      &filt2_data, filt2_k * filt2_c * filt2_h * filt2_w * sizeof(float)));

  // convolution
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

  miopenConvolutionDescriptor_t conv2_desc;
  MI_CALL(miopenCreateConvolutionDescriptor(&conv2_desc));
  MI_CALL(miopenInitConvolutionDescriptor(
        conv2_desc, miopenConvolution,
        pad2_h, pad2_w, str2_h, str2_w, dil2_h, dil2_w));

  // output
  int out2_n;
  int out2_c;
  int out2_h;
  int out2_w;
  
  MI_CALL(miopenGetConvolutionForwardOutputDim(
        conv2_desc, in2_desc, filt2_desc,
        &out2_n, &out2_c, &out2_h, &out2_w));

  std::cout << "out2_n: " << out2_n << std::endl;
  std::cout << "out2_c: " << out2_c << std::endl;
  std::cout << "out2_h: " << out2_h << std::endl;
  std::cout << "out2_w: " << out2_w << std::endl;
  std::cout << std::endl;

  miopenTensorDescriptor_t out2_desc;
  MI_CALL(miopenCreateTensorDescriptor(&out2_desc));
  MI_CALL(miopenSet4dTensorDescriptor(
        out2_desc, miopenHalf,
        out2_n, out2_c, out2_h, out2_w));

  float *out2_data;
  HIP_CALL(hipMalloc(
        &out2_data, out2_n * out2_c * out2_h * out2_w * sizeof(float)));

  // workspace
  size_t workspaceSize2;
  MI_CALL(miopenConvolutionForwardGetWorkSpaceSize(
        hipdnn2, filt2_desc, in2_desc, conv2_desc, out2_desc, &workspaceSize2));
  float *workspace2;
  HIP_CALL(hipMalloc(&workspace2, workspaceSize2));

  std::cout<<"Workspace size = "<<workspaceSize2<<std::endl;
  // algorithm
  int requestAlgoCount2 = 6;
  int returnedAlgoCount2;
  bool exhaustiveSearch2 = true;
  miopenConvAlgoPerf_t perfResults2[5];
  MI_CALL(miopenFindConvolutionForwardAlgorithm(
        hipdnn2,
        in2_desc, in2_data,
	filt2_desc, filt2_data,
	conv2_desc, out2_desc, out2_data,
	requestAlgoCount2, &returnedAlgoCount2,
	perfResults2, workspace2, workspaceSize2,
	exhaustiveSearch2));
  std::cout<<"Returned Algo Count: "<<returnedAlgoCount2<<std::endl;        
  for(int i=0;i<returnedAlgoCount2;i++)
  {
  	std::cout<<"Algorithm: "<<perfResults2[i].fwd_algo<<"\t";
  	std::cout<<"Memory: "<<perfResults2[i].memory<<"\t";
  	std::cout<<"Time: "<<perfResults2[i].time<<std::endl;
  }

  // perform
  float alpha1 = 1.f;
  float alpha2 = 1.f;
  float beta1 = 0.f;
  float beta2 = 0.f;


	//generate input and kernel data
  //conv1
  dev_iota<<<in1_w * in1_h, in1_n * in1_c, 0, stream1>>>(in1_data);
  dev_const<<<filt1_w * filt1_h, filt1_k * filt1_c, 0, stream1>>>(filt1_data, 1.f);
  //conv 2
  dev_iota<<<in2_w * in2_h, in2_n * in2_c, 0, stream2>>>(in2_data);
  dev_const<<<filt2_w * filt2_h, filt2_k * filt2_c, 0, stream2>>>(filt1_data, 1.f);
  //extra
  //dev_iota<<<filt1_w * filt1_h, filt1_k * filt1_c, 0, stream1>>>(filt1_data);
  //dev_iota<<<filt2_w * filt2_h, filt2_k * filt2_c, 0, stream2>>>(filt1_data);
  HIP_CALL(hipDeviceSynchronize());
  
  MI_CALL(miopenEnableProfiling(hipdnn1, 1));
  MI_CALL(miopenEnableProfiling(hipdnn2, 1));
  float time1, time2;
  float t1=0;
  float t2=0;

  std::cout<<"Chosen algorithm 1 = "<<perfResults1[a1].fwd_algo<<std::endl;
  std::cout<<"Chosen algorithm 2 = "<<perfResults2[a2].fwd_algo<<std::endl;

  int conv_repeat = 20;

	std::chrono::duration<double> t11,t22;
  typedef std::chrono::high_resolution_clock Clock;
  Clock::time_point t_start = Clock::now();

  for (int i = 0; i < conv_repeat; i++)
  {
    Clock::time_point t_start1 = Clock::now();

    MI_CALL(miopenConvolutionForward(
        hipdnn1,
        &alpha1, in1_desc, in1_data, filt1_desc, filt1_data,
        conv1_desc, perfResults1[a1].fwd_algo, &beta1, out1_desc, out1_data,
				workspace1, workspaceSize1));

    Clock::time_point t_final1 = Clock::now();
    Clock::time_point t_start2 = Clock::now();

    MI_CALL(miopenConvolutionForward(
        hipdnn2,
        &alpha2, in2_desc, in2_data, filt2_desc, filt2_data,
        conv2_desc, perfResults2[a2].fwd_algo, &beta2, out2_desc, out2_data,
				workspace2, workspaceSize2));


    Clock::time_point t_final2 = Clock::now();

    MI_CALL(miopenGetKernelTime(hipdnn1, &time1));
    MI_CALL(miopenGetKernelTime(hipdnn2, &time2));
    t1 += time1;
    t2 += time2;

    std::chrono::duration<double> time_span1 = std::chrono::duration_cast<std::chrono::duration<double>>(t_final1 - t_start1)*1000;
    std::chrono::duration<double> time_span2 = std::chrono::duration_cast<std::chrono::duration<double>>(t_final2 - t_start2)*1000;
    t11 += time_span1;
    t22 += time_span2;

    HIP_CALL(hipDeviceSynchronize());
  }

  HIP_CALL(hipDeviceSynchronize());
  Clock::time_point t_final = Clock::now();
  std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t_final - t_start)*1000;

  // results
  std::cout << "total conv duration:" << time_span.count() << " ms" << std::endl;
  std::cout << "total conv1 duration:" << t11.count() << " ms" << std::endl;
  std::cout << "total conv2 duration:" << t22.count() << " ms" << std::endl;
  std::cout<< "t1 = "<<t1<<" ms"<<std::endl;
  std::cout<< "t2 = "<<t2<<" ms"<<std::endl;
  std::cout<< "total = "<<(t1+t2)<<" ms"<<std::endl;

//  std::cout << "in_data:" << std::endl;
//  //print(in1_data, in1_n, in1_c, in1_h, in1_w);
//  
//  std::cout << "filt_data:" << std::endl;
//  //print(filt1_data, filt1_k, filt1_c, filt1_h, filt1_w);
//  
//  std::cout << "out_data:" << std::endl;
//  //print(out1_data, out1_n, out1_c, out1_h, out1_w);
//  //printFirstKElements(out1_data, out1_n, out1_c, out1_h, out1_w, 20);
//
//  // finalizing
  HIP_CALL(hipFree(workspace1));
  HIP_CALL(hipFree(out1_data));
  MI_CALL(miopenDestroyTensorDescriptor(out1_desc));
  MI_CALL(miopenDestroyConvolutionDescriptor(conv1_desc));
  HIP_CALL(hipFree(filt1_data));
  MI_CALL(miopenDestroyTensorDescriptor(filt1_desc));
  HIP_CALL(hipFree(in1_data));
  MI_CALL(miopenDestroyTensorDescriptor(in1_desc));
  MI_CALL(miopenDestroy(hipdnn1));

  HIP_CALL(hipFree(workspace2));
  HIP_CALL(hipFree(out2_data));
  MI_CALL(miopenDestroyTensorDescriptor(out2_desc));
  MI_CALL(miopenDestroyConvolutionDescriptor(conv2_desc));
  HIP_CALL(hipFree(filt2_data));
  MI_CALL(miopenDestroyTensorDescriptor(filt2_desc));
  HIP_CALL(hipFree(in2_data));
  MI_CALL(miopenDestroyTensorDescriptor(in2_desc));
  MI_CALL(miopenDestroy(hipdnn2));
  return 0;
}
