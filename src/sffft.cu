// sfFFT: fused causal FFT convolution on NVIDIA tensor cores (tuned on GB10 / DGX Spark).
// SPDX-License-Identifier: MIT
// y[b,h,0:L] = causal_conv(u[b,h,0:L], k[h,0:L]) via length N=2L transform, N = R0*R1(*R2), radices 16/32/64.
// Every stage is a batch of small complex DFT matmuls on fp16 tensor cores; data stays in shared memory.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstring>
#include <chrono>
#include "selection.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cufft.h>
using namespace nvcuda;

#define CK(x) do{cudaError_t e_=(x); if(e_!=cudaSuccess){printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e_));exit(1);}}while(0)
#define CF(x) do{cufftResult r_=(x); if(r_!=CUFFT_SUCCESS){printf("cuFFT %s @%d: %d\n",#x,__LINE__,(int)r_);exit(1);}}while(0)

enum class Allocation { Device, Managed, Mapped, Hybrid };
static const char* allocation_name(Allocation mode) {
  return mode == Allocation::Device ? "device" : mode == Allocation::Managed ? "managed" : mode == Allocation::Mapped ? "mapped" : "hybrid";
}
using Clock = std::chrono::steady_clock;
static double elapsed_ms(Clock::time_point start) { return std::chrono::duration<double,std::milli>(Clock::now()-start).count(); }

struct Memory {
  struct Block { void* device; void* host; size_t bytes; Allocation kind; };
  Allocation mode; bool can_prefetch;
  std::vector<Block> blocks;
  size_t owned_bytes = 0, upload_bytes = 0, download_bytes = 0;
  double allocation_ms = 0, copy_ms = 0;
  Memory(Allocation mode_, bool can_prefetch_) : mode(mode_), can_prefetch(can_prefetch_) {}
  Memory(const Memory&) = delete;
  Memory& operator=(const Memory&) = delete;
  template<typename T> void allocate(T** pointer, size_t bytes, bool shared_io = true) {
    Allocation kind = mode == Allocation::Hybrid ? (shared_io ? Allocation::Mapped : Allocation::Device) : mode;
    auto start = Clock::now(); void* device = nullptr; void* host = nullptr;
    if (kind == Allocation::Mapped) {
      CK(cudaHostAlloc(&host, bytes, cudaHostAllocMapped)); CK(cudaHostGetDevicePointer(&device, host, 0));
    } else if (kind == Allocation::Managed) {
      CK(cudaMallocManaged(&device, bytes)); host = device;
    } else { CK(cudaMalloc(&device, bytes)); }
    *pointer = static_cast<T*>(device); blocks.push_back({device, host, bytes, kind}); owned_bytes += bytes;
    allocation_ms += elapsed_ms(start);
  }
  template<typename T> T* host_pointer(const T* device) {
    for (const auto& block : blocks) if (block.device == device) return static_cast<T*>(block.host);
    return nullptr;
  }
  void prefetch(const void* pointer, size_t bytes, bool to_gpu) {
    if (mode != Allocation::Managed || !can_prefetch) return;
#if CUDART_VERSION >= 13000
    cudaMemLocation location{}; location.type = to_gpu ? cudaMemLocationTypeDevice : cudaMemLocationTypeHost; location.id = 0;
    CK(cudaMemPrefetchAsync(pointer, bytes, location, 0));
#else
    CK(cudaMemPrefetchAsync(pointer, bytes, to_gpu ? 0 : cudaCpuDeviceId));
#endif
  }
  void prepare_gpu() { for (const auto& block : blocks) prefetch(block.device, block.bytes, true); }
  template<typename T> void upload(T* device, const T* source, size_t count) {
    T* host = host_pointer(device); if (host == source) return;
    auto start = Clock::now();
    if (host) std::memcpy(host, source, count*sizeof(T));
    else { CK(cudaMemcpy(device, source, count*sizeof(T), cudaMemcpyHostToDevice)); upload_bytes += count*sizeof(T); }
    copy_ms += elapsed_ms(start);
  }
  template<typename T> const T* read(const T* device, size_t count, std::vector<T>& shadow) {
    if (T* host = host_pointer(device)) {
      prefetch(device, count*sizeof(T), false); CK(cudaDeviceSynchronize()); return host;
    }
    shadow.resize(count); auto start = Clock::now();
    CK(cudaMemcpy(shadow.data(), device, count*sizeof(T), cudaMemcpyDeviceToHost));
    download_bytes += count*sizeof(T); copy_ms += elapsed_ms(start); return shadow.data();
  }
  ~Memory() {
    for (const auto& block : blocks) {
      if (block.kind == Allocation::Mapped) CK(cudaFreeHost(block.host)); else CK(cudaFree(block.device));
    }
  }
};

template<typename TAcc> using Acc = wmma::fragment<wmma::accumulator,16,16,16,TAcc>;
typedef wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> FA;
typedef wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> FB;

template<int R0_, int R1_, int R2_ = 1> struct Plan {
  static constexpr int S = R2_ > 1 ? 3 : 2;
  static constexpr int N = R0_*R1_*R2_, L = N/2;
  __host__ __device__ static constexpr int R(int j) { return j==0 ? R0_ : (j==1 ? R1_ : R2_); }
  __host__ __device__ static constexpr int Nj(int j) { return j==0 ? N : (j==1 ? N/R0_ : (j==2 ? N/(R0_*R1_) : 1)); }
  __host__ __device__ static constexpr int Foff(int j) {
    return j==0 ? 0 : j==1 ? (R1_==R0_ ? 0 : 2*R0_*R0_) :
           R2_==R0_ ? 0 : R2_==R1_ ? Foff(1) : 2*R0_*R0_ + (R1_==R0_ ? 0 : 2*R1_*R1_);
  }
  static constexpr int Fsize = 2*R0_*R0_ + (R1_==R0_ ? 0 : 2*R1_*R1_) +
                               (R2_>1 && R2_!=R0_ && R2_!=R1_ ? 2*R2_*R2_ : 0);
  template<int W, typename TAcc = float> static constexpr int smem() { return 4*N + 2*Fsize + W*256*sizeof(TAcc); }
};

template<class P, int J>
__device__ __forceinline__ float2 twid(int pos, const float2* __restrict__ tw, bool cj) {
  constexpr int Nm = P::Nj(J), Nm1 = P::Nj(J+1), RJ = P::R(J);
  const int nrest = pos % Nm1, km = (pos / Nm1) % RJ;
  int e = (P::N/Nm)*nrest*km; if (e > P::N/2) e -= P::N;
  float sv, cv; __sincosf(-6.283185307179586f*(float)e/(float)P::N, &sv, &cv);
  return make_float2(cv, cj ? -sv : sv);
}
__device__ __forceinline__ float tof(float x) { return x; }
__device__ __forceinline__ float tof(__nv_bfloat16 x) { return __bfloat162float(x); }
template<typename T> __device__ __forceinline__ T fromf(float x);
template<> __device__ __forceinline__ float fromf<float>(float x) { return x; }
template<> __device__ __forceinline__ __nv_bfloat16 fromf<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

__device__ __forceinline__ void st8h(half* d, const float* v) {
  __half2 h0 = __floats2half2_rn(v[0], v[1]), h1 = __floats2half2_rn(v[2], v[3]), h2 = __floats2half2_rn(v[4], v[5]), h3 = __floats2half2_rn(v[6], v[7]);
  uint4 q; q.x = *(unsigned*)&h0; q.y = *(unsigned*)&h1; q.z = *(unsigned*)&h2; q.w = *(unsigned*)&h3; *reinterpret_cast<uint4*>(d) = q; }
__device__ __forceinline__ void ld8f(const float* s, float* v) {
  float4 a = *reinterpret_cast<const float4*>(s), b = *reinterpret_cast<const float4*>(s+4);
  v[0]=a.x; v[1]=a.y; v[2]=a.z; v[3]=a.w; v[4]=b.x; v[5]=b.y; v[6]=b.z; v[7]=b.w; }
__device__ __forceinline__ void ld8f(const half* s, float* v) {
  uint4 q = *reinterpret_cast<const uint4*>(s); const half* h = reinterpret_cast<const half*>(&q);
  #pragma unroll
  for (int e = 0; e < 8; e++) v[e] = __half2float(h[e]); }
__device__ __forceinline__ void st8out(float* d, const float* v, float sc) {
  reinterpret_cast<float4*>(d)[0] = make_float4(v[0]*sc, v[1]*sc, v[2]*sc, v[3]*sc);
  reinterpret_cast<float4*>(d)[1] = make_float4(v[4]*sc, v[5]*sc, v[6]*sc, v[7]*sc); }
__device__ __forceinline__ void st8out(__nv_bfloat16* d, const float* v, float sc) {
  __nv_bfloat162 h[4]; for (int q = 0; q < 4; q++) h[q] = __floats2bfloat162_rn(v[2*q]*sc, v[2*q+1]*sc);
  *reinterpret_cast<uint4*>(d) = *reinterpret_cast<uint4*>(h); }

// DFT along axis J (stride s = Nj(J+1) >= 16): per (outer block, 16-column strip) left-multiply by F_R (or conj).
template<class P, int J, bool INV, int W, typename TIO, typename TAcc>
__device__ __forceinline__ void left_stage(half* Xr, half* Xi, const half* F, TAcc* scr,
                                           const float2* __restrict__ tw, const float2* __restrict__ K, TIO* yp) {
  constexpr int R = P::R(J), s = P::Nj(J+1), RT = R/16, ST = s/16, UNITS = (P::N/(R*s))*ST;
  constexpr bool RIN = (!INV && J == 0), ROUT = (INV && J == 0);
  constexpr int KT = RIN ? (R >= 32 ? R/32 : 1) : RT;
  constexpr int IT = ROUT ? (R >= 32 ? R/32 : 1) : RT;
  const half* Fr = F + P::Foff(J); const half* Fi = Fr + R*R;
  const float sc = rsqrtf((float)R);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  TAcc* sr = scr + warp*256;
  const int rr = lane >> 1, cc0 = (lane & 1)*8;
  for (int u = warp; u < UNITS; u += W) {
    const int outer = u / ST, js = u % ST, base = outer*R*s + js*16;
    FB br[KT], bi[KT];
    #pragma unroll
    for (int kk = 0; kk < KT; kk++) {
      wmma::load_matrix_sync(br[kk], Xr + base + kk*16*s, s);
      if constexpr (!RIN) wmma::load_matrix_sync(bi[kk], Xi + base + kk*16*s, s);
    }
    #pragma unroll
    for (int i = 0; i < IT; i++) {
      Acc<TAcc> cr, ci, t; wmma::fill_fragment(cr, 0.f); wmma::fill_fragment(ci, 0.f); wmma::fill_fragment(t, 0.f);
      #pragma unroll
      for (int kk = 0; kk < KT; kk++) {
        FA ar, ai; wmma::load_matrix_sync(ar, Fr + i*16*R + kk*16, R); wmma::load_matrix_sync(ai, Fi + i*16*R + kk*16, R);
        if constexpr (RIN) { wmma::mma_sync(cr, ar, br[kk], cr); wmma::mma_sync(ci, ai, br[kk], ci); }
        else if constexpr (ROUT) { wmma::mma_sync(cr, ar, br[kk], cr); wmma::mma_sync(cr, ai, bi[kk], cr); }
        else if constexpr (!INV) { wmma::mma_sync(cr, ar, br[kk], cr); wmma::mma_sync(t, ai, bi[kk], t);
                                   wmma::mma_sync(ci, ar, bi[kk], ci); wmma::mma_sync(ci, ai, br[kk], ci); }
        else { wmma::mma_sync(cr, ar, br[kk], cr); wmma::mma_sync(cr, ai, bi[kk], cr);
               wmma::mma_sync(ci, ar, bi[kk], ci); wmma::mma_sync(t, ai, br[kk], t); }
      }
      if constexpr (!RIN && !ROUT) {
        if constexpr (!INV) { for (int q = 0; q < cr.num_elements; q++) cr.x[q] -= t.x[q]; }
        else                { for (int q = 0; q < ci.num_elements; q++) ci.x[q] -= t.x[q]; }
      }
      float va[8], vb[8];
      wmma::store_matrix_sync(sr, cr, 16, wmma::mem_row_major); __syncwarp();
      ld8f(sr + rr*16 + cc0, va); __syncwarp();
      if constexpr (!ROUT) { wmma::store_matrix_sync(sr, ci, 16, wmma::mem_row_major); __syncwarp(); ld8f(sr + rr*16 + cc0, vb); __syncwarp(); }
      const int row = i*16 + rr;
      if constexpr (ROUT) {
        if (row < R/2) st8out(yp + row*s + js*16 + cc0, va, sc);
      } else {
        const int pos0 = outer*R*s + row*s + js*16 + cc0; float orr[8], oi[8];
        #pragma unroll
        for (int e = 0; e < 8; e++) {
          const float a = va[e]*sc, b = vb[e]*sc; float2 w;
          if constexpr (!INV) w = twid<P,J>(pos0+e, tw, false);
          else w = twid<P,(J>0?J-1:0)>(pos0+e, tw, true);
          orr[e] = a*w.x - b*w.y; oi[e] = a*w.y + b*w.x;
        }
        st8h(Xr + pos0, orr); st8h(Xi + pos0, oi);
      }
    }
  }
}

// DFT along the last axis (stride 1): per 16-row strip right-multiply by F_R (or conj). Forward ends with the kernel spectrum.
template<class P, bool INV, int W, typename TAcc>
__device__ __forceinline__ void right_stage(half* Xr, half* Xi, const half* F, TAcc* scr,
                                            const float2* __restrict__ tw, const float2* __restrict__ K) {
  constexpr int J = P::S - 1, R = P::R(J), RT = R/16, UNITS = P::N/(R*16);
  const half* Fr = F + P::Foff(J); const half* Fi = Fr + R*R;
  const float sc = rsqrtf((float)R);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  TAcc* sr = scr + warp*256;
  const int rr = lane >> 1, cc0 = (lane & 1)*8;
  for (int u = warp; u < UNITS; u += W) {
    const int base = u*16*R;
    FA ar[RT], ai[RT];
    #pragma unroll
    for (int kk = 0; kk < RT; kk++) { wmma::load_matrix_sync(ar[kk], Xr + base + kk*16, R); wmma::load_matrix_sync(ai[kk], Xi + base + kk*16, R); }
    #pragma unroll
    for (int i = 0; i < RT; i++) {
      Acc<TAcc> cr, ci, t; wmma::fill_fragment(cr, 0.f); wmma::fill_fragment(ci, 0.f); wmma::fill_fragment(t, 0.f);
      #pragma unroll
      for (int kk = 0; kk < RT; kk++) {
        FB fr, fi; wmma::load_matrix_sync(fr, Fr + kk*16*R + i*16, R); wmma::load_matrix_sync(fi, Fi + kk*16*R + i*16, R);
        if constexpr (!INV) { wmma::mma_sync(cr, ar[kk], fr, cr); wmma::mma_sync(t, ai[kk], fi, t);
                              wmma::mma_sync(ci, ar[kk], fi, ci); wmma::mma_sync(ci, ai[kk], fr, ci); }
        else { wmma::mma_sync(cr, ar[kk], fr, cr); wmma::mma_sync(cr, ai[kk], fi, cr);
               wmma::mma_sync(ci, ai[kk], fr, ci); wmma::mma_sync(t, ar[kk], fi, t); }
      }
      if constexpr (!INV) { for (int q = 0; q < cr.num_elements; q++) cr.x[q] -= t.x[q]; }
      else                { for (int q = 0; q < ci.num_elements; q++) ci.x[q] -= t.x[q]; }
      float va[8], vb[8];
      wmma::store_matrix_sync(sr, cr, 16, wmma::mem_row_major); __syncwarp(); ld8f(sr + rr*16 + cc0, va); __syncwarp();
      wmma::store_matrix_sync(sr, ci, 16, wmma::mem_row_major); __syncwarp(); ld8f(sr + rr*16 + cc0, vb); __syncwarp();
      const int pos0 = base + rr*R + i*16 + cc0; float orr[8], oi[8]; float2 kw[8];
      if constexpr (!INV) { const float4* kp = reinterpret_cast<const float4*>(K + pos0);
        #pragma unroll
        for (int q = 0; q < 4; q++) { float4 t4 = __ldg(kp + q); kw[2*q] = make_float2(t4.x, t4.y); kw[2*q+1] = make_float2(t4.z, t4.w); } }
      #pragma unroll
      for (int e = 0; e < 8; e++) {
        const float a = va[e]*sc, b = vb[e]*sc;
        float2 w; if constexpr (!INV) w = kw[e]; else w = twid<P,(J>0?J-1:0)>(pos0+e, tw, true);
        orr[e] = a*w.x - b*w.y; oi[e] = a*w.y + b*w.x;
      }
      st8h(Xr + pos0, orr); st8h(Xi + pos0, oi);
    }
  }
}

template<class P, int W, typename TIO, typename TAcc, bool CACHED>
__global__ void __launch_bounds__(W*32)
fftconv(const TIO* __restrict__ u, const float2* __restrict__ Kf, const float2* __restrict__ tw, const half* __restrict__ dft, TIO* __restrict__ y, int B, int H) {
  constexpr int N = P::N, L = P::L;
  extern __shared__ __align__(128) unsigned char smem[];
  half* Xr = (half*)smem; half* Xi = Xr + N; half* F = Xi + N; TAcc* scr = (TAcc*)(F + P::Fsize);
  const int seq = blockIdx.x, h = seq / B, b = seq % B;       // batch fastest: Kf[h] stays hot in L2
  const TIO* up = u + ((size_t)b*H + h)*L; TIO* yp = y + ((size_t)b*H + h)*L;
  const float2* K = Kf + (size_t)h*N;
  const int tid = threadIdx.x, NT = W*32;
  #pragma unroll
  for (int j = 0; j < P::S; j++) {
    const int R = P::R(j);
    if ((j > 0 && R == P::R(0)) || (j > 1 && R == P::R(1))) continue;
    half* Fr = F + P::Foff(j); half* Fi = Fr + R*R;
    if constexpr (CACHED) {
      const half* source = dft + (R == 16 ? 0 : R == 32 ? 512 : 2560);
      for (int idx = tid*8; idx < 2*R*R; idx += NT*8) *reinterpret_cast<uint4*>(Fr+idx) = *reinterpret_cast<const uint4*>(source+idx);
    } else {
      for (int idx = tid; idx < R*R; idx += NT) { float sv, cv; sincospif(2.0f*(float)(((idx/R)*(idx%R))%R)/(float)R, &sv, &cv);
        Fr[idx] = __float2half(cv); Fi[idx] = __float2half(-sv); }
    }
  }
  for (int idx = tid*8; idx < L; idx += NT*8) { float v[8];
    if constexpr (sizeof(TIO) == 4) { float4 a = *reinterpret_cast<const float4*>(up+idx), b = *reinterpret_cast<const float4*>(up+idx+4);
      v[0]=a.x; v[1]=a.y; v[2]=a.z; v[3]=a.w; v[4]=b.x; v[5]=b.y; v[6]=b.z; v[7]=b.w; }
    else { uint4 q = *reinterpret_cast<const uint4*>(up+idx); const __nv_bfloat16* hb = reinterpret_cast<const __nv_bfloat16*>(&q);
      for (int e = 0; e < 8; e++) v[e] = tof(hb[e]); }
    st8h(Xr + idx, v); }
  for (int idx = L + tid*8; idx < N; idx += NT*8) *reinterpret_cast<uint4*>(Xr + idx) = make_uint4(0,0,0,0);
  __syncthreads();
  left_stage<P,0,false,W,TIO,TAcc>(Xr, Xi, F, scr, tw, K, yp); __syncthreads();
  if constexpr (P::S == 3) { left_stage<P,1,false,W,TIO,TAcc>(Xr, Xi, F, scr, tw, K, yp); __syncthreads(); }
  right_stage<P,false,W,TAcc>(Xr, Xi, F, scr, tw, K); __syncwarp();   // same warp owns the same strips: no block barrier
  right_stage<P,true,W,TAcc>(Xr, Xi, F, scr, tw, K); __syncthreads();
  if constexpr (P::S == 3) { left_stage<P,1,true,W,TIO,TAcc>(Xr, Xi, F, scr, tw, K, yp); __syncthreads(); }
  left_stage<P,0,true,W,TIO,TAcc>(Xr, Xi, F, scr, tw, K, yp);
}

// ---- spectrum layout + cuFFT baseline ----
__global__ void permuteK(const cufftComplex* Kc, float2* Kf, int H, int N, int R0, int R1, int R2) {
  size_t idx = blockIdx.x*(size_t)blockDim.x + threadIdx.x; if (idx >= (size_t)H*N) return;
  int h = idx / N, p = idx % N, d0, d1, d2;
  if (R2 > 1) { d2 = p % R2; d1 = (p / R2) % R1; d0 = p / (R1*R2); } else { d2 = 0; d1 = p % R1; d0 = p / R1; }
  int k = d0 + R0*(d1 + R1*d2); cufftComplex v = Kc[(size_t)h*N + k]; Kf[idx] = make_float2(v.x, v.y);
}
__global__ void halfK(const cufftComplex* Kc, float2* Kh, int H, int N) {
  int M = N/2+1; size_t idx = blockIdx.x*(size_t)blockDim.x + threadIdx.x; if (idx >= (size_t)H*M) return;
  int h = idx / M, k = idx % M; cufftComplex q = Kc[(size_t)h*N + k]; Kh[idx] = make_float2(q.x/N, q.y/N);
}
__global__ void padK(const float* u, float* up, size_t BH, int L) {
  size_t idx = blockIdx.x*(size_t)blockDim.x + threadIdx.x; if (idx >= BH*2*L) return;
  size_t s = idx / (2*L); int n = idx % (2*L); up[idx] = n < L ? u[s*L+n] : 0.f;
}
__global__ void mulK(cufftComplex* U, const float2* Kh, size_t BH, int H, int M) {
  size_t idx = blockIdx.x*(size_t)blockDim.x + threadIdx.x; if (idx >= BH*M) return;
  size_t s = idx / M; int k = idx % M, h = s % H; float2 q = Kh[(size_t)h*M+k]; cufftComplex a = U[idx];
  U[idx] = make_cuComplex(a.x*q.x - a.y*q.y, a.x*q.y + a.y*q.x);
}
__global__ void sliceK(const float* yp, float* y, size_t BH, int L) {
  size_t idx = blockIdx.x*(size_t)blockDim.x + threadIdx.x; if (idx >= BH*L) return;
  size_t s = idx / L; int n = idx % L; y[idx] = yp[s*2*L+n];
}
__global__ void tobf16(const float* a, __nv_bfloat16* b, size_t n) { size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x; if (i < n) b[i] = __float2bfloat16(a[i]); }

static unsigned long long rng = 88172645463325252ULL;
static double urand() { rng ^= rng<<13; rng ^= rng>>7; rng ^= rng<<17; return ((rng>>11)+0.5)*(1.0/9007199254740992.0); }
static double nrand() { return sqrt(-2*log(urand()))*cos(2*M_PI*urand()); }

struct Bench {
  int L, N, H, B; size_t BH; int iters;
  Memory memory;
  std::vector<float> hu_shadow, hk, yref_shadow;
  float* hu; const float* yref = nullptr; double setup_ms = 0;
  std::vector<Candidate> candidates;
  float *du, *dy, *dup, *dyp, *dyref; __nv_bfloat16 *dub, *dyb; half* dft; int kernel_mode;
  float2 *dKh, *dtw, *dKf; cufftComplex *dKc, *dU;
  cufftHandle pf, pi; cudaEvent_t e0, e1;
  template<class F> float timeit(F f) {
    for (int i = 0; i < 3; i++) f(); CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(e0)); for (int i = 0; i < iters; i++) f(); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); CK(cudaGetLastError()); return ms/iters; }
  Bench(int L_, int B_, int H_, double input_scale, Allocation mode, bool can_prefetch, int kernel_mode_) : L(L_), N(2*L_), H(H_), B(B_), BH((size_t)B_*H_), memory(mode, can_prefetch), kernel_mode(kernel_mode_) {
    auto start = Clock::now();
    iters = std::max(10, (int)(20000/L)); if (iters > 200) iters = 200;
    int M = N/2+1;
    memory.allocate(&du, BH*L*4); memory.allocate(&dy, BH*L*4); memory.allocate(&dyref, BH*L*4);
    memory.allocate(&dub, BH*L*2, false); memory.allocate(&dyb, BH*L*2);
    memory.allocate(&dup, BH*N*4, false); memory.allocate(&dyp, BH*N*4, false); memory.allocate(&dU, BH*M*8, false);
    memory.allocate(&dKc, (size_t)H*N*8); memory.allocate(&dKh, (size_t)H*M*8, false); memory.allocate(&dtw, N*8, false);
    memory.allocate(&dKf, (size_t)H*N*8, false);
    dft = nullptr;
    if (kernel_mode != 0) memory.allocate(&dft, 10752*sizeof(half), false);
    hu = memory.host_pointer(du);
    if (!hu) { hu_shadow.resize(BH*L); hu = hu_shadow.data(); }
    hk.resize((size_t)H*L);
    for (size_t i = 0; i < BH*L; i++) hu[i] = (float)(nrand()*input_scale);
    for (int h = 0; h < H; h++) { double a = 1.0/(4.0 + (double)L*h/H);
      for (int n = 0; n < L; n++) hk[(size_t)h*L+n] = (float)(nrand()*exp(-a*n)*sqrt(2*a)); }
    std::vector<cufftComplex> hkc_shadow; cufftComplex* hkc = memory.host_pointer(dKc);
    if (!hkc) { hkc_shadow.resize((size_t)H*N); hkc = hkc_shadow.data(); }
    for (int h = 0; h < H; h++) for (int n = 0; n < N; n++) hkc[(size_t)h*N+n] = make_cuComplex(n < L ? hk[(size_t)h*L+n] : 0.f, 0.f);
    std::vector<float2> htw_shadow; float2* htw = memory.host_pointer(dtw);
    if (!htw) { htw_shadow.resize(N); htw = htw_shadow.data(); }
    for (int j = 0; j < N; j++) { double t = 2*M_PI*j/N; htw[j] = make_float2((float)cos(t), (float)-sin(t)); }
    memory.upload(du, hu, BH*L); memory.upload(dKc, hkc, (size_t)H*N); memory.upload(dtw, htw, N);
    if (dft) {
      std::vector<half> coefficients_shadow; half* coefficients = memory.host_pointer(dft);
      if (!coefficients) { coefficients_shadow.resize(10752); coefficients = coefficients_shadow.data(); }
      int offset = 0;
      for (int radix : {16,32,64}) {
        for (int i = 0; i < radix*radix; i++) {
          double angle = 2*M_PI*((i/radix)*(i%radix)%radix)/radix;
          coefficients[offset+i] = __float2half((float)cos(angle));
          coefficients[offset+radix*radix+i] = __float2half((float)-sin(angle));
        }
        offset += 2*radix*radix;
      }
      memory.upload(dft, coefficients, 10752);
    }
    memory.prepare_gpu();
    tobf16<<<(unsigned)((BH*L+255)/256),256>>>(du, dub, BH*L);
    cufftHandle pk; CF(cufftPlan1d(&pk, N, CUFFT_C2C, H)); CF(cufftExecC2C(pk, dKc, dKc, CUFFT_FORWARD)); cufftDestroy(pk);
    halfK<<<(unsigned)(((size_t)H*M+255)/256),256>>>(dKc, dKh, H, N);
    CF(cufftPlan1d(&pf, N, CUFFT_R2C, (int)BH)); CF(cufftPlan1d(&pi, N, CUFFT_C2R, (int)BH));
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1)); CK(cudaDeviceSynchronize()); setup_ms = elapsed_ms(start);
  }
  void core() { int M = N/2+1; CF(cufftExecR2C(pf, dup, dU)); mulK<<<(unsigned)((BH*M+255)/256),256>>>(dU, dKh, BH, H, M); CF(cufftExecC2R(pi, dU, dyp)); }
  void full() { padK<<<(unsigned)((BH*N+255)/256),256>>>(du, dup, BH, L); core(); sliceK<<<(unsigned)((BH*L+255)/256),256>>>(dyp, dyref, BH, L); }
  void baseline() {
    float t_full = timeit([&]{ full(); }), t_core = timeit([&]{ core(); });
    full(); CK(cudaDeviceSynchronize()); yref = memory.read(dyref, BH*L, yref_shadow);
    double rn = 0, rd = 0;
    for (size_t s : {(size_t)0, BH/2, BH-1}) { int h = s % H;
      for (int n = 0; n < L; n++) { double acc = 0; for (int m = 0; m <= n; m++) acc += (double)hu[s*L+m]*hk[(size_t)h*L+n-m];
        double d = yref[s*L+n]-acc; rn += d*d; rd += acc*acc; } }
    printf("RESULT,%d,cufft_pipeline_fp32,-,%.5f,%.3e\n", L, t_full, sqrt(rn/rd));
    printf("RESULT,%d,cufft_core_fp32,-,%.5f,%.3e\n", L, t_core, sqrt(rn/rd));
  }
  template<class P, int W, typename TAcc, bool CACHED> void fused(const char* name) {
    if (P::L != L) { printf("bad plan\n"); exit(1); }
    permuteK<<<(unsigned)(((size_t)H*N+255)/256),256>>>(dKc, dKf, H, N, P::R(0), P::R(1), P::R(2));
    constexpr int SM = P::template smem<W,TAcc>();
    CK(cudaFuncSetAttribute(fftconv<P,W,float,TAcc,CACHED>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM));
    CK(cudaFuncSetAttribute(fftconv<P,W,__nv_bfloat16,TAcc,CACHED>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM));
    int occ = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fftconv<P,W,float,TAcc,CACHED>, W*32, SM));
    auto f32 = [&]{ fftconv<P,W,float,TAcc,CACHED><<<(unsigned)BH, W*32, SM>>>(du, dKf, dtw, dft, dy, B, H); };
    auto b16 = [&]{ fftconv<P,W,__nv_bfloat16,TAcc,CACHED><<<(unsigned)BH, W*32, SM>>>(dub, dKf, dtw, dft, dyb, B, H); };
    float t32 = timeit(f32), t16 = timeit(b16);
    f32(); CK(cudaDeviceSynchronize()); std::vector<float> y_shadow; const float* y = memory.read(dy, BH*L, y_shadow);
    std::vector<__nv_bfloat16> yb_shadow; b16(); CK(cudaDeviceSynchronize()); const __nv_bfloat16* yb = memory.read(dyb, BH*L, yb_shadow);
    double n1 = 0, n2 = 0, d = 0;
    for (size_t i = 0; i < BH*L; i++) { double r = yref[i]; double a = y[i]-r, b = (double)__bfloat162float(yb[i])-r; n1 += a*a; n2 += b*b; d += r*r; }
    char config[96]; snprintf(config, sizeof(config), "%s/w%d/occ%d/smem%d/kernel_%s", name, W, occ, SM, CACHED ? "cached" : "legacy");
    const char* m32 = sizeof(TAcc) == 4 ? "fused_fp32io" : "fused_fp16acc_fp32io";
    const char* m16 = sizeof(TAcc) == 4 ? "fused_bf16io" : "fused_fp16acc_bf16io";
    const double err32 = relative_l2(n1, d), err16 = relative_l2(n2, d);
    candidates.push_back({m32, "fp32", config, t32, err32});
    candidates.push_back({m16, "bf16", config, t16, err16});
    printf("RESULT,%d,%s,%s,%.5f,%.3e\n", L, m32, config, t32, err32);
    printf("RESULT,%d,%s,%s,%.5f,%.3e\n", L, m16, config, t16, err16);
    fflush(stdout);
  }
  template<class P, int W> void variants(const char* name) {
    if (kernel_mode != 1) { fused<P,W,float,false>(name); fused<P,W,half,false>(name); }
    if (kernel_mode != 0) { fused<P,W,float,true>(name); fused<P,W,half,true>(name); }
  }
  bool select(double max_error, const char* io) {
    const Candidate* best = select_candidate(candidates, max_error, io);
    if (!best) { printf("NO_MATCH,%d,%s,%.9e\n", L, io, max_error); return false; }
    printf("SELECT,%d,%s,%s,%.5f,%.9e,%.9e,%s\n", L, best->method.c_str(), best->config.c_str(), best->ms, best->rel_err, max_error, best->io.c_str());
    return true;
  }
  void report_memory() {
    printf("MEMORY,%d,%s,%zu,%zu,%.3f,%zu,%zu,%.3f,%.3f\n", L, allocation_name(memory.mode), memory.owned_bytes,
           (hu_shadow.size()+yref_shadow.size())*sizeof(float), memory.allocation_ms, memory.upload_bytes, memory.download_bytes, memory.copy_ms, setup_ms);
  }
  ~Bench() { cufftDestroy(pf); cufftDestroy(pi); cudaEventDestroy(e0); cudaEventDestroy(e1); }
};

static void usage() {
  printf("Usage: sffft [B H [L]] [--max-error REL_L2] [--io fp32|bf16|any] [--seed N] [--input-scale SCALE] [--allocator device|managed|mapped|hybrid] [--kernel legacy|cached|auto] [--memory-info]\n"
         "L: 0 (all) or a power of two from 128 to 8192. Defaults: B=8 H=768, io=fp32.\n"
         "--max-error selects the fastest passing fused plan on these calibration inputs.\n"
         "Error is measured against cuFFT fp32; no match exits 2. Invalid arguments exit 1.\n");
}

int main(int argc, char** argv) {
  int B = 8, H = 768, only = 0, positional = 0, failed = 0;
  double max_error = -1, input_scale = 1; const char* io = "fp32";
  Allocation allocation = Allocation::Device; bool memory_info = false; int kernel_mode = 0;
  for (int i = 1; i < argc; i++) {
    const char* arg = argv[i]; char* end = nullptr; errno = 0;
    if (!strcmp(arg, "--help") || !strcmp(arg, "-h")) { usage(); return 0; }
    if (!strcmp(arg, "--memory-info")) { memory_info = true; continue; }
    if (!strcmp(arg, "--allocator")) {
      if (++i == argc) { usage(); return 1; }
      if (!strcmp(argv[i], "device")) allocation = Allocation::Device;
      else if (!strcmp(argv[i], "managed")) allocation = Allocation::Managed;
      else if (!strcmp(argv[i], "mapped")) allocation = Allocation::Mapped;
      else if (!strcmp(argv[i], "hybrid")) allocation = Allocation::Hybrid;
      else { fprintf(stderr, "--allocator must be device, managed, mapped, or hybrid\n"); return 1; }
    } else if (!strcmp(arg, "--kernel")) {
      if (++i == argc) { usage(); return 1; }
      if (!strcmp(argv[i], "legacy")) kernel_mode = 0;
      else if (!strcmp(argv[i], "cached")) kernel_mode = 1;
      else if (!strcmp(argv[i], "auto")) kernel_mode = 2;
      else { fprintf(stderr, "--kernel must be legacy, cached, or auto\n"); return 1; }
    } else if (!strcmp(arg, "--max-error")) {
      if (++i == argc) { usage(); return 1; }
      max_error = strtod(argv[i], &end);
      if (errno || end == argv[i] || *end || !std::isfinite(max_error) || max_error <= 0) {
        fprintf(stderr, "--max-error must be positive and finite\n"); return 1;
      }
    } else if (!strcmp(arg, "--input-scale")) {
      if (++i == argc) { usage(); return 1; }
      input_scale = strtod(argv[i], &end);
      if (errno || end == argv[i] || *end || !std::isfinite(input_scale) || input_scale <= 0 || input_scale > std::numeric_limits<float>::max()) {
        fprintf(stderr, "--input-scale must be positive, finite, and representable in fp32\n"); return 1;
      }
    } else if (!strcmp(arg, "--io")) {
      if (++i == argc) { usage(); return 1; }
      io = argv[i];
      if (strcmp(io, "fp32") && strcmp(io, "bf16") && strcmp(io, "any")) {
        fprintf(stderr, "--io must be fp32, bf16, or any\n"); return 1;
      }
    } else if (!strcmp(arg, "--seed")) {
      if (++i == argc) { usage(); return 1; }
      rng = strtoull(argv[i], &end, 10);
      if (errno || end == argv[i] || *end || argv[i][0] == '-' || !rng) {
        fprintf(stderr, "--seed must be a nonzero unsigned integer\n"); return 1;
      }
    } else {
      long value = strtol(arg, &end, 10);
      if (errno || end == arg || *end || value < 0 || value > INT_MAX || positional == 3) {
        fprintf(stderr, "Invalid argument: %s\n", arg); usage(); return 1;
      }
      if (positional == 0) B = (int)value;
      else if (positional == 1) H = (int)value;
      else only = (int)value;
      positional++;
    }
  }
  if (B < 1 || H < 1 || (long long)B*H > INT_MAX || (only && (only < 128 || only > 8192 || (only & (only-1))))) {
    fprintf(stderr, "B and H must be positive with B*H <= INT_MAX; L must be 0 or 128,256,512,1024,2048,4096,8192\n"); return 1;
  }
  if (allocation == Allocation::Mapped || allocation == Allocation::Hybrid) CK(cudaSetDeviceFlags(cudaDeviceMapHost));
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  int managed = 0, concurrent = 0, pageable = 0, host_tables = 0;
  CK(cudaDeviceGetAttribute(&managed, cudaDevAttrManagedMemory, 0));
  CK(cudaDeviceGetAttribute(&concurrent, cudaDevAttrConcurrentManagedAccess, 0));
  CK(cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, 0));
  CK(cudaDeviceGetAttribute(&host_tables, cudaDevAttrPageableMemoryAccessUsesHostPageTables, 0));
  printf("# MEMORY_CAPS,mapped=%d,managed=%d,concurrent_managed=%d,pageable=%d,host_page_tables=%d,integrated=%d\n", p.canMapHostMemory, managed, concurrent, pageable, host_tables, p.integrated);
  if (memory_info) return 0;
  if (((allocation == Allocation::Mapped || allocation == Allocation::Hybrid) && !p.canMapHostMemory) || (allocation == Allocation::Managed && !managed)) {
    fprintf(stderr, "Requested allocator is unsupported on this device\n"); return 1;
  }
  printf("# %s sm_%d%d %d SMs, smem/block %zu, L2 %d MB, B=%d H=%d\n", p.name, p.major, p.minor, p.multiProcessorCount, p.sharedMemPerBlockOptin, p.l2CacheSize>>20, B, H);
  printf("# calibration seed=%llu, input_scale=%.9e, max_error=%.9e, io=%s,allocator=%s,kernel=%s\n", rng, input_scale, max_error, io, allocation_name(allocation), kernel_mode == 0 ? "legacy" : kernel_mode == 1 ? "cached" : "auto");
  printf("# RESULT,L,method,config,ms,rel_err\n");
  printf("# MEMORY,L,allocator,owned_bytes,persistent_host_shadow_bytes,allocation_ms,upload_bytes,download_bytes,copy_ms,setup_ms\n");
  if (max_error > 0) printf("# SELECT,L,method,config,ms,rel_err,max_error,io\n");
#define RUN(LL, ...) if (!only || only == LL) { Bench b(LL, B, H, input_scale, allocation, concurrent, kernel_mode); b.baseline(); __VA_ARGS__ if (max_error > 0 && !b.select(max_error, io)) failed++; b.report_memory(); }
  RUN(128,  b.variants<Plan<16,16>,4>("16x16"); b.variants<Plan<16,16>,8>("16x16"); )
  RUN(256,  b.variants<Plan<32,16>,4>("32x16"); b.variants<Plan<16,32>,4>("16x32"); b.variants<Plan<32,16>,8>("32x16"); )
  RUN(512,  b.variants<Plan<32,32>,4>("32x32"); b.variants<Plan<32,32>,8>("32x32"); b.variants<Plan<64,16>,4>("64x16"); b.variants<Plan<16,64>,4>("16x64"); )
  RUN(1024, b.variants<Plan<64,32>,4>("64x32"); b.variants<Plan<32,64>,4>("32x64"); b.variants<Plan<64,32>,8>("64x32"); b.variants<Plan<32,64>,8>("32x64"); )
  RUN(2048, b.variants<Plan<16,16,16>,4>("16x16x16"); b.variants<Plan<16,16,16>,8>("16x16x16"); b.variants<Plan<64,64>,8>("64x64"); b.variants<Plan<64,64>,4>("64x64"); )
  RUN(4096, b.variants<Plan<32,16,16>,4>("32x16x16"); b.variants<Plan<16,32,16>,4>("16x32x16"); b.variants<Plan<16,16,32>,4>("16x16x32"); b.variants<Plan<32,16,16>,8>("32x16x16"); b.variants<Plan<16,16,32>,8>("16x16x32"); )
  RUN(8192, b.variants<Plan<32,32,16>,4>("32x32x16"); b.variants<Plan<16,32,32>,4>("16x32x32"); b.variants<Plan<64,16,16>,4>("64x16x16"); b.variants<Plan<32,32,16>,8>("32x32x16"); b.variants<Plan<16,32,32>,8>("16x32x32"); b.variants<Plan<16,16,64>,8>("16x16x64"); )
  return failed ? 2 : 0;
}
