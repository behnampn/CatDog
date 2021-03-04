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
#include <boost/program_options.hpp>
#include <hip/hip_runtime.h>
#include <hip_fp16.h>
#include <rocblas.h>


namespace po = boost::program_options;

#ifndef TYPENUM
#define TYPENUM 1 //0:__half 1:float 2:double 3:int 4:long long int
#endif

#if TYPENUM == 0
  #define DTYPE __half
//  #define HIPDNN_DATA_TYPE HIPDNN_DATA_HALF
#elif TYPENUM == 1
  #define DTYPE float
//  #define HIPDNN_DATA_TYPE HIPDNN_DATA_FLOAT
#endif


#define HIP_CALL(f) { \
  hipError_t err = (f); \
  if (err != hipSuccess) { \
    std::cout \
        << "    Error occurred: " << err << std::endl; \
    std::exit(1); \
  } \
}

#define ROCBLAS_CALL(f) { \
  rocblas_status err = (f); \
  if (err != rocblas_status_success) { \
    std::cout \
        << "    Error occurred: " << err << std::endl; \
    std::exit(1); \
  } \
}

__global__ void dev_const(DTYPE *px, DTYPE k) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  //#if TYPENUM == 0 //__half
  //  px[tid] = __float2half(k);
  //#else
    px[tid] = k;
  //#endif
}

__global__ void dev_iota(DTYPE *px, int max_elements) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid < max_elements){
    #if TYPENUM == 0 //__half
      px[tid] = __float2half(tid % 10);
    #else
      px[tid] = tid % 10;
    #endif
  }
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

void print(const DTYPE *data, int r, int c) {
  std::vector<DTYPE> buffer(1 << 20);
  HIP_CALL(hipMemcpy(
        buffer.data(), data,
        r * c * sizeof(DTYPE),
        hipMemcpyDeviceToHost));
  int a = 0;
  for (int i = 0; i < r; ++i) {
    for (int j = 0; j < c; ++j) {
          #if TYPENUM == 0 //__half
            std::cout << std::setw(4) << std::right << __half2float(buffer[a]);
          #else
            std::cout << std::setw(4) << std::right << buffer[a];
          #endif
          ++a;
        }
        std::cout << std::endl;
  }
  std::cout << std::endl;
}

void printFirstKElements(const DTYPE *data, int r, int c, int K) {
  std::vector<DTYPE> buffer(1 << 20);
  HIP_CALL(hipMemcpy(
        buffer.data(), data,
        r * c * sizeof(DTYPE),
        hipMemcpyDeviceToHost));
  for (int i = 0; i < K; ++i){ 
    #if DTYPE == 0 //__half
      std::cout << std::setw(4) << " "  << __half2float(buffer[i]);
    #else
      std::cout << std::setw(4) << " "  << buffer[i];
    #endif
  }
  std::cout << std::endl;
}


void parseArgs(po::variables_map& vm, po::options_description& desc, int argc, char* argv[]){
  desc.add_options()
    ("help,h", "produce help message")
    ("p1", po::value<int>(), "partition size of 1st convolution (SM count)")
    ("p2", po::value<int>(), "partition size of 2nd convolution (SM count)")
    ("Ar1", po::value<int>() -> default_value(32), "batch size of 1st convolution")
    ("Ar2", po::value<int>() -> default_value(32), "batch size of 2nd convolution")
    ("Ac1", po::value<int>() -> default_value(32), "filter dimension of 1st convolution (fw=fh)")
    ("Ac2", po::value<int>() -> default_value(32), "filter dimension of 2nd convolution (fw=fh)")
    ("Bc1", po::value<int>() -> default_value(32), "input dimension of 1st convolution (w=h)")
    ("Bc2", po::value<int>() -> default_value(32), "input dimension of 2nd convolution (w=h)")
    ("serial", "to run convolutions sequentially")
    ("single1", "run only 1st convolution")
    ("single2", "run only 2nd convolution")
    ("silent", "not print anything")
    ("repeat", po::value<int>() -> default_value(10), "number of iterations")
    ("print-device", "print device information")
    ("print-results", "print output of convolutions");
    //("include-path,I", po::value< vector<string> >(), "include path")
    //("input-file", po::value< vector<string> >(), "input file");
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);  
}

int main(int argc, char* argv[]) {

  po::variables_map vm;
  po::options_description desc("Allowed Options");
  parseArgs(vm, desc, argc, argv);  
  if (vm.count("help")){
    std::cout<<desc<<std::endl;
    return 0;
  }
  if (vm.count("print-device"))
    printCurrDeviceInfo();

  int id;
  HIP_CALL(hipGetDevice(&id));
  hipDeviceProp_t prop;
  HIP_CALL(hipGetDeviceProperties(&prop, id));
  int multiProcessorCount = prop.multiProcessorCount;
  
  hipStream_t stream1, stream2;
  int SM1_num, SM2_num;
  if (vm.count("p1") == 0){
    hipStreamCreateWithFlags( &stream1, hipStreamNonBlocking);
    SM1_num = multiProcessorCount;
  }else{
    uint64_t cuMaskSize = 2;
    uint64_t cuMaskL1 = (0x1L << vm["p1"].as<int>())-1;
    uint32_t cuMask1[2];
    cuMask1[0] = cuMaskL1 & 0xFFFFFFFF;
    cuMask1[1] = cuMaskL1 >> 32;
    if(!vm.count("silent")){
    	std::cout<<cuMask1[0]<<" "<<cuMask1[1]<<std::endl;
    }
    hipExtStreamCreateWithCUMask(& stream1, cuMaskSize, cuMask1);
    SM1_num = vm["p1"].as<int>();
  }

  if (vm.count("p2") == 0){
    hipStreamCreateWithFlags( &stream2, hipStreamNonBlocking);
    SM2_num = multiProcessorCount;
  }else{
    uint64_t cuMaskSize = 2; 
    uint64_t cuMaskL2 = ((0x1L << vm["p2"].as<int>()) - 1);
    if (vm.count("p1")){
      uint64_t cuMaskL1 = (0x1L << vm["p1"].as<int>())-1;
      std::bitset<64> nbit1(cuMaskL1);
      if (nbit1.count() + vm["p2"].as<int>() <= multiProcessorCount )
        cuMaskL2 = cuMaskL2 << nbit1.count();
    }
    uint32_t cuMask2[2];
    cuMask2[0] = cuMaskL2 & 0xFFFFFFFF;
    cuMask2[1] = cuMaskL2 >> 32;
    if(!vm.count("silent")){
      std::cout<<cuMask2[0]<<" "<<cuMask2[1]<<std::endl;
    }
    hipExtStreamCreateWithCUMask(& stream2, cuMaskSize, cuMask2);
    SM2_num = vm["p2"].as<int>();
  }
  if(!vm.count("silent")){  
    std::cout<<"number of multi-processor for op1"<<SM1_num<<std::endl;
    std::cout<<"number of multi-processor for op2"<<SM2_num<<std::endl;
  }


  rocblas_handle rocblas1, rocblas2;
  ROCBLAS_CALL(rocblas_create_handle(&rocblas1));
  ROCBLAS_CALL(rocblas_create_handle(&rocblas2));

  ROCBLAS_CALL(rocblas_set_stream(
    rocblas1,
    stream1));
  ROCBLAS_CALL(rocblas_set_stream(
    rocblas2,
    stream2));

  // input
  const int in1_Ar = vm["Ar1"].as<int>();
  const int in1_Ac = vm["Ac1"].as<int>();
  const int in1_Bc = vm["Bc1"].as<int>();

  if(!vm.count("silent")){
    std::cout << "in1_Ar: " << in1_Ar << std::endl;
    std::cout << "in1_Ac: " << in1_Ac << std::endl;
    std::cout << "in1_Bc: " << in1_Bc << std::endl;
    std::cout << std::endl;
  }

  DTYPE *in1_Adata, *in1_Bdata, *in1_Cdata;
  HIP_CALL(hipMalloc(
        &in1_Adata, in1_Ar * in1_Ac * sizeof(DTYPE)));
  HIP_CALL(hipMalloc(
        &in1_Bdata, in1_Ac * in1_Bc * sizeof(DTYPE)));
  HIP_CALL(hipMalloc(
        &in1_Cdata, in1_Ar * in1_Bc * sizeof(DTYPE)));

  const int in2_Ar = vm["Ar2"].as<int>();
  const int in2_Ac = vm["Ac2"].as<int>();
  const int in2_Bc = vm["Bc2"].as<int>();

  if(!vm.count("silent")){
    std::cout << "in2_Ar: " << in2_Ar << std::endl;
    std::cout << "in2_Ac: " << in2_Ac << std::endl;
    std::cout << "in2_Bc: " << in2_Bc << std::endl;
    std::cout << std::endl;
  }

  DTYPE *in2_Adata, *in2_Bdata, *in2_Cdata;;
  HIP_CALL(hipMalloc(
        &in2_Adata, in2_Ar * in2_Ac * sizeof(DTYPE)));
  HIP_CALL(hipMalloc(
        &in2_Bdata, in2_Ac * in2_Bc * sizeof(DTYPE)));
  HIP_CALL(hipMalloc(
        &in2_Cdata, in2_Ar * in2_Bc * sizeof(DTYPE)));


  // perform
  #if TYPENUM == 2 // double
    DTYPE alpha1 = 1.0;
    DTYPE alpha2 = 1.0;
    DTYPE beta1 = 0.0;
    DTYPE beta2 = 0.0;
  #else
    float alpha1 = 1.f;
    float alpha2 = 1.f;
    float beta1 = 0.f;
    float beta2 = 0.f;
  #endif
  dev_iota<<<(in1_Ar * in1_Ac + 63)/64, 64, 0, stream1>>>(in1_Adata, in1_Ar * in1_Ac);
  dev_iota<<<(in1_Ac * in1_Bc + 63)/64, 64, 0, stream1>>>(in1_Bdata, in1_Ac * in1_Bc);
  dev_iota<<<(in1_Ar * in1_Ac + 63)/64, 64, 0, stream1>>>(in1_Cdata, in1_Ar * in1_Bc);
  
  dev_iota<<<(in2_Ar * in2_Ac + 63)/64, 64, 0, stream2>>>(in2_Adata, in2_Ar * in2_Ac);
  dev_iota<<<(in2_Ac * in2_Bc + 63)/64, 64, 0, stream2>>>(in2_Bdata, in2_Ac * in2_Bc);
  dev_iota<<<(in2_Ar * in2_Ac + 63)/64, 64, 0, stream2>>>(in2_Cdata, in2_Ar * in2_Bc);
  
  HIP_CALL(hipDeviceSynchronize());

  int gemm_repeat = vm["repeat"].as<int>();
  bool serial = false;

  if (vm.count("serial"))
    serial = true;

  typedef std::chrono::high_resolution_clock Clock;
  Clock::time_point t_start = Clock::now();

  for (int i = 0; i < gemm_repeat; i++){
    if(!vm.count("single2"))	  
      ROCBLAS_CALL(rocblas_sgemm(
        rocblas1,
        rocblas_operation_none, rocblas_operation_none, in1_Ar, in1_Bc, in1_Ac,
        &alpha1, in1_Adata, in1_Ar, in1_Bdata, in1_Ac, &beta1, in1_Cdata, in1_Bc));

    if (serial)
      HIP_CALL(hipDeviceSynchronize());

    if(!vm.count("single1"))
      ROCBLAS_CALL(rocblas_sgemm(
        rocblas2,
        rocblas_operation_none, rocblas_operation_none, in2_Ar, in2_Bc, in2_Ac,
        &alpha2, in2_Adata, in2_Ar, in2_Bdata, in2_Ac, &beta2, in2_Cdata, in2_Bc));

    HIP_CALL(hipDeviceSynchronize()); //it can be uncomment if you want back to back execution
  }
  //HIP_CALL(hipDeviceSynchronize());

  Clock::time_point t_final = Clock::now();
  std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t_final - t_start); 
  if(!vm.count("silent")){   
    std::cout << "total conv duration:" << time_span.count() << " seconds" << std::endl;
  }
  
  std::cout << SM2_num <<","<<1.0/time_span.count()<<std::endl; 
  //results
  if (vm.count("print-results") && gemm_repeat && !vm.count("silent")){
    if(!vm.count("single2")){
      std::cout << "1st matrix" << std::endl;
      std::cout << "in_Adata:" << std::endl;
      print(in1_Adata, in1_Ar, in1_Ac);
  
      std::cout << "in_Bdata:" << std::endl;
      print(in1_Bdata, in1_Ac, in1_Bc);

      std::cout << "in_Cdata:" << std::endl;
      print(in1_Cdata, in1_Ar, in1_Bc);
      //printFirstKElements(out1_data, out1_n, out1_c, out1_h, out1_w, 20);
    }
    if(!vm.count("single1")){
      std::cout << "2nd matrix" << std::endl;
      std::cout << "in_Adata:" << std::endl;
      print(in2_Adata, in2_Ar, in2_Ac);
  
      std::cout << "in_Bdata:" << std::endl;
      print(in2_Bdata, in2_Ac, in2_Bc);

      std::cout << "in_Cdata:" << std::endl;
      print(in2_Cdata, in2_Ar, in2_Bc);
      //printFirstKElements(out1_data, out1_n, out1_c, out1_h, out1_w, 20);
    }
  }
  // finalizing

  HIP_CALL(hipFree(in1_Adata));
  HIP_CALL(hipFree(in2_Adata));
  HIP_CALL(hipFree(in1_Bdata));
  HIP_CALL(hipFree(in2_Bdata));
  HIP_CALL(hipFree(in1_Cdata));
  HIP_CALL(hipFree(in2_Cdata));
  ROCBLAS_CALL(rocblas_destroy_handle(rocblas1));
  ROCBLAS_CALL(rocblas_destroy_handle(rocblas2));
  return 0;
}
