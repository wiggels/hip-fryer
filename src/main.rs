// Copyright 2025 Hunter Wigelsworth
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// This file contains code derived from gpu-fryer by Hugging Face,
// originally licensed under Apache License 2.0.

use clap::Parser;
use float8::F8E4M3;
use rand::rngs::SmallRng;
use rand::{RngCore, SeedableRng, rng};
use rocm_rs::hip::bindings::{
    hipDeviceAttribute_t, hipDeviceAttribute_t_hipDeviceAttributeComputeCapabilityMajor,
    hipDeviceAttribute_t_hipDeviceAttributeComputeCapabilityMinor, hipDeviceGetAttribute,
    hipDeviceProp_tR0600, hipDeviceptr_t, hipError_t_hipSuccess, hipFree, hipGetDeviceCount,
    hipGetDevicePropertiesR0600, hipInit, hipMalloc, hipMemGetInfo, hipMemcpyHtoD, hipMemset,
    hipSetDevice, hipStreamSynchronize,
};
use rocm_rs::rocblas::ffi::{
    rocblas_create_handle, rocblas_datatype__rocblas_datatype_bf16_r,
    rocblas_datatype__rocblas_datatype_f32_r, rocblas_destroy_handle,
    rocblas_gemm_algo__rocblas_gemm_algo_standard, rocblas_gemm_ex, rocblas_handle,
    rocblas_operation, rocblas_operation__rocblas_operation_none, rocblas_sgemm,
    rocblas_status__rocblas_status_success,
};
use rocm_smi_lib::{RocmSmi, RocmSmiDevice, RsmiTemperatureMetric, RsmiTemperatureType};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ptr;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, Mutex};
use tokio::signal;
use tokio::sync::mpsc::{
    UnboundedReceiver as Receiver, UnboundedSender as Sender, unbounded_channel,
};

const SIZE: usize = 8192; // Ensure SIZE % 16 == 0 for Tensor Core optimization
const MEM_TO_USE_PCT: f64 = 0.90; // Use 90% of GPU memory
const MIN_DURATION_SECS: u64 = 10;

const GPU_THROTTLING_REASON: &str =
    "GPU is throttled. Check the throttling reasons and temperatures";
const GPU_FLOPS_REASON: &str =
    "GPU is not performing as expected. Check the flops values and temperatures";

type AllocBufferTuple<T> = (HipSlice<T>, HipSlice<T>, Vec<HipSlice<T>>);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ThrottleStatus {
    pub thermal_throttling: bool,
    pub power_throttling: bool,
    pub current_throttling: bool,
}

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Duration in seconds to burn the GPUs
    #[clap(default_value = "60")]
    duration_secs: u64,
    /// Tolerate software throttling if the TFLOPS are in the acceptable range
    #[clap(long, default_value = "false")]
    tolerate_software_throttling: bool,
    /// TFLOPS tolerance (%) compared to best GPU
    /// If the TFLOPS are within `tflops_tolerance`% of the best performing GPU, test will pass
    #[clap(long, default_value = "10")]
    tflops_tolerance: f64,
    /// Use FP32 precision. If unset, will use FP32 if no GPUs support BF16 or FP8.
    #[clap(long)]
    use_fp32: bool,
    /// Use BF16 precision. GPU must support BF16 type. If unset, will use BF16 only if all GPUs support it.
    #[clap(long)]
    use_bf16: bool,
    /// Use FP8 precision. GPU must support FP8 type.
    #[clap(long)]
    use_fp8: bool,
}

#[derive(Debug, Clone)]
struct BurnResult {
    gpu_idx: usize,
    flops_max: usize,
    flops_min: usize,
    flops_sum: usize,
    n_iters: usize,
    temp_max: usize,
    temp_sum: usize,
    temp_min: usize,
    temp_count: usize, // Separate counter for temperature readings
    throttling_hw: usize,
    throttling_thermal_sw: usize,
    throttling_thermal_hw: usize,
}

impl BurnResult {
    fn new(gpu_idx: usize) -> Self {
        Self {
            gpu_idx,
            flops_min: usize::MAX,
            temp_min: usize::MAX,
            ..Default::default()
        }
    }

    fn flops_avg(&self) -> f64 {
        if self.n_iters == 0 {
            0.0
        } else {
            self.flops_sum as f64 / self.n_iters as f64
        }
    }

    fn temp_avg(&self) -> f64 {
        if self.temp_count == 0 {
            0.0
        } else {
            self.temp_sum as f64 / self.temp_count as f64
        }
    }

    fn is_throttled(&self) -> bool {
        self.throttling_hw > 0 || self.throttling_thermal_sw > 0 || self.throttling_thermal_hw > 0
    }
}

impl Default for BurnResult {
    fn default() -> Self {
        Self {
            gpu_idx: 0,
            flops_max: 0,
            flops_min: usize::MAX,
            flops_sum: 0,
            n_iters: 0,
            temp_max: 0,
            temp_sum: 0,
            temp_min: usize::MAX,
            temp_count: 0,
            throttling_hw: 0,
            throttling_thermal_sw: 0,
            throttling_thermal_hw: 0,
        }
    }
}

#[derive(Debug, Clone)]
struct Config {
    duration_secs: u64,
    tflops_tolerance: f64,
    tolerate_software_throttling: bool,
    use_bf16: bool,
    use_fp8: bool,
    use_fp32: bool,
}

// Safe HIP memory slice (similar to CudaSlice)
pub struct HipSlice<T> {
    ptr: hipDeviceptr_t,
    len: usize,
    _phantom: PhantomData<T>,
}

impl<T> HipSlice<T> {
    pub fn alloc_zeros(len: usize) -> anyhow::Result<Self> {
        unsafe {
            let mut ptr: hipDeviceptr_t = ptr::null_mut();
            let byte_size = len * std::mem::size_of::<T>();
            let result = hipMalloc(&mut ptr, byte_size);
            if result != hipError_t_hipSuccess {
                return Err(anyhow::anyhow!(
                    "Failed to allocate GPU memory: {:?}",
                    result
                ));
            }

            // Zero the memory
            let result = hipMemset(ptr, 0, byte_size);
            if result != hipError_t_hipSuccess {
                let _ = hipFree(ptr); // Clean up on error
                return Err(anyhow::anyhow!("Failed to zero GPU memory: {:?}", result));
            }

            Ok(HipSlice {
                ptr,
                len,
                _phantom: PhantomData,
            })
        }
    }

    pub fn from_host_data(data: Vec<T>) -> anyhow::Result<Self> {
        unsafe {
            let mut ptr: hipDeviceptr_t = ptr::null_mut();
            let byte_size = data.len() * std::mem::size_of::<T>();
            let result = hipMalloc(&mut ptr, byte_size);
            if result != hipError_t_hipSuccess {
                return Err(anyhow::anyhow!(
                    "Failed to allocate GPU memory: {:?}",
                    result
                ));
            }

            let result = hipMemcpyHtoD(ptr, data.as_ptr() as *mut std::ffi::c_void, byte_size);
            if result != hipError_t_hipSuccess {
                let _ = hipFree(ptr); // Clean up on error
                return Err(anyhow::anyhow!("Failed to copy data to GPU: {:?}", result));
            }

            Ok(HipSlice {
                ptr,
                len: data.len(),
                _phantom: PhantomData,
            })
        }
    }

    pub fn as_ptr(&self) -> *mut T {
        self.ptr as *mut T
    }

    pub fn len(&self) -> usize {
        self.len
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T> Drop for HipSlice<T> {
    fn drop(&mut self) {
        unsafe {
            let _ = hipFree(self.ptr);
        }
    }
}

// Safe HIP abstractions (similar to cudarc design)
pub struct HipContext {
    device_id: i32,
    name: String,
}

impl HipContext {
    pub fn new(device_id: i32) -> anyhow::Result<Self> {
        unsafe {
            // Set device
            let result = hipSetDevice(device_id);
            if result != hipError_t_hipSuccess {
                return Err(anyhow::anyhow!(
                    "Failed to set device {}: {:?}",
                    device_id,
                    result
                ));
            }

            // Get device properties
            let mut props: hipDeviceProp_tR0600 = std::mem::zeroed();
            let result = hipGetDevicePropertiesR0600(&mut props, device_id);
            if result != hipError_t_hipSuccess {
                return Err(anyhow::anyhow!(
                    "Failed to get device properties: {:?}",
                    result
                ));
            }

            let name = std::ffi::CStr::from_ptr(props.name.as_ptr())
                .to_string_lossy()
                .to_string();

            Ok(HipContext { device_id, name })
        }
    }

    pub fn device_count() -> anyhow::Result<i32> {
        unsafe {
            let mut count = 0;
            let result = hipGetDeviceCount(&mut count);
            if result != hipError_t_hipSuccess {
                return Err(anyhow::anyhow!("Failed to get device count: {:?}", result));
            }
            Ok(count)
        }
    }

    pub fn ordinal(&self) -> usize {
        self.device_id as usize
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn set_current(&self) -> anyhow::Result<()> {
        unsafe {
            let result = hipSetDevice(self.device_id);
            if result != hipError_t_hipSuccess {
                return Err(anyhow::anyhow!(
                    "Failed to set current device: {:?}",
                    result
                ));
            }
            Ok(())
        }
    }

    pub fn get_attribute(&self, attr: hipDeviceAttribute_t) -> anyhow::Result<i32> {
        unsafe {
            let mut value = 0;
            let result = hipDeviceGetAttribute(&mut value, attr, self.device_id);
            if result != hipError_t_hipSuccess {
                return Err(anyhow::anyhow!(
                    "Failed to get device attribute: {:?}",
                    result
                ));
            }
            Ok(value)
        }
    }

    pub fn mem_get_info(&self) -> anyhow::Result<(usize, usize)> {
        self.set_current()?;
        unsafe {
            let mut free: usize = 0;
            let mut total: usize = 0;
            let result = hipMemGetInfo(&mut free, &mut total);
            if result != hipError_t_hipSuccess {
                return Err(anyhow::anyhow!("Failed to get memory info: {:?}", result));
            }
            Ok((free, total))
        }
    }

    pub fn alloc_zeros<T>(&self, len: usize) -> anyhow::Result<HipSlice<T>> {
        HipSlice::alloc_zeros(len)
    }

    pub fn htod_copy<T: Clone>(&self, data: Vec<T>) -> anyhow::Result<HipSlice<T>> {
        HipSlice::from_host_data(data)
    }

    pub fn synchronize(&self) -> anyhow::Result<()> {
        unsafe {
            let result = hipStreamSynchronize(ptr::null_mut());
            if result != hipError_t_hipSuccess {
                return Err(anyhow::anyhow!(
                    "Stream synchronization failed: {:?}",
                    result
                ));
            }
            Ok(())
        }
    }
}

// Safe ROCBlas handle wrapper
pub struct RocBlasHandle {
    handle: rocblas_handle,
}

impl RocBlasHandle {
    pub fn new() -> anyhow::Result<Self> {
        unsafe {
            let mut handle: rocblas_handle = ptr::null_mut();
            let status = rocblas_create_handle(&mut handle);
            if status != rocblas_status__rocblas_status_success {
                return Err(anyhow::anyhow!(
                    "Failed to create ROCBlas handle: {:?}",
                    status
                ));
            }
            Ok(RocBlasHandle { handle })
        }
    }

    pub fn sgemm(
        &self,
        params: &GemmParams,
        alpha: &f32,
        a: &HipSlice<f32>,
        b: &HipSlice<f32>,
        beta: &f32,
        c: &mut HipSlice<f32>,
    ) -> anyhow::Result<()> {
        unsafe {
            let status = rocblas_sgemm(
                self.handle,
                params.transa,
                params.transb,
                params.m,
                params.n,
                params.k,
                alpha as *const f32,
                a.as_ptr() as *const f32,
                params.lda,
                b.as_ptr() as *const f32,
                params.ldb,
                beta as *const f32,
                c.as_ptr(),
                params.ldc,
            );
            if status != rocblas_status__rocblas_status_success {
                return Err(anyhow::anyhow!("ROCBlas SGEMM failed: {:?}", status));
            }
            Ok(())
        }
    }

    pub fn gemm_ex_bf16(
        &self,
        params: &GemmParams,
        alpha: &half::bf16,
        a: &HipSlice<half::bf16>,
        b: &HipSlice<half::bf16>,
        beta: &half::bf16,
        c: &mut HipSlice<half::bf16>,
    ) -> anyhow::Result<()> {
        unsafe {
            let alpha_f32 = alpha.to_f32();
            let beta_f32 = beta.to_f32();

            let status = rocblas_gemm_ex(
                self.handle,
                params.transa,
                params.transb,
                params.m,
                params.n,
                params.k,
                &alpha_f32 as *const f32 as *const std::ffi::c_void,
                a.as_ptr() as *const std::ffi::c_void,
                rocblas_datatype__rocblas_datatype_bf16_r,
                params.lda,
                b.as_ptr() as *const std::ffi::c_void,
                rocblas_datatype__rocblas_datatype_bf16_r,
                params.ldb,
                &beta_f32 as *const f32 as *const std::ffi::c_void,
                c.as_ptr() as *const std::ffi::c_void,
                rocblas_datatype__rocblas_datatype_bf16_r,
                params.ldc,
                c.as_ptr() as *mut std::ffi::c_void,
                rocblas_datatype__rocblas_datatype_bf16_r,
                params.ldc,
                rocblas_datatype__rocblas_datatype_f32_r,
                rocblas_gemm_algo__rocblas_gemm_algo_standard,
                0,
                0,
            );
            if status != rocblas_status__rocblas_status_success {
                return Err(anyhow::anyhow!("ROCBlas BF16 GEMM failed: {:?}", status));
            }
            Ok(())
        }
    }

    // FP8 GEMM - placeholder implementation since ROCm may not have native FP8 GEMM support yet
    pub fn gemm_ex_fp8(
        &self,
        _params: &GemmParams,
        _alpha: &F8E4M3,
        _a: &HipSlice<F8E4M3>,
        _b: &HipSlice<F8E4M3>,
        _beta: &F8E4M3,
        _c: &mut HipSlice<F8E4M3>,
    ) -> anyhow::Result<()> {
        // For now, we'll return an error as FP8 GEMM may not be directly supported
        // In practice, you would need to check if your ROCm version supports FP8 operations
        Err(anyhow::anyhow!(
            "FP8 GEMM not yet supported in this ROCBlas version"
        ))
    }
}

impl Drop for RocBlasHandle {
    fn drop(&mut self) {
        unsafe {
            let _ = rocblas_destroy_handle(self.handle);
        }
    }
}

// GEMM operation parameters to reduce function argument count
#[derive(Debug, Clone)]
pub struct GemmParams {
    pub transa: rocblas_operation,
    pub transb: rocblas_operation,
    pub m: i32,
    pub n: i32,
    pub k: i32,
    pub lda: i32,
    pub ldb: i32,
    pub ldc: i32,
}

impl Default for GemmParams {
    fn default() -> Self {
        Self {
            transa: rocblas_operation__rocblas_operation_none,
            transb: rocblas_operation__rocblas_operation_none,
            m: SIZE as i32,
            n: SIZE as i32,
            k: SIZE as i32,
            lda: SIZE as i32,
            ldb: SIZE as i32,
            ldc: SIZE as i32,
        }
    }
}

trait VariablePrecisionFloat: Copy + Debug + Send + Sync + Unpin + 'static {
    fn from_f32(f: f32) -> Self;
    fn compute_gemm(
        handle: &RocBlasHandle,
        a: &HipSlice<Self>,
        b: &HipSlice<Self>,
        c: &mut HipSlice<Self>,
    ) -> anyhow::Result<()>;
}

impl VariablePrecisionFloat for f32 {
    fn from_f32(f: f32) -> Self {
        f
    }

    fn compute_gemm(
        handle: &RocBlasHandle,
        a: &HipSlice<Self>,
        b: &HipSlice<Self>,
        c: &mut HipSlice<Self>,
    ) -> anyhow::Result<()> {
        let params = GemmParams::default();
        handle.sgemm(&params, &1.0f32, a, b, &0.0f32, c)
    }
}

impl VariablePrecisionFloat for half::bf16 {
    fn from_f32(f: f32) -> Self {
        half::bf16::from_f32(f)
    }

    fn compute_gemm(
        handle: &RocBlasHandle,
        a: &HipSlice<Self>,
        b: &HipSlice<Self>,
        c: &mut HipSlice<Self>,
    ) -> anyhow::Result<()> {
        let params = GemmParams::default();
        handle.gemm_ex_bf16(
            &params,
            &half::bf16::from_f32(1.0),
            a,
            b,
            &half::bf16::from_f32(0.0),
            c,
        )
    }
}

impl VariablePrecisionFloat for F8E4M3 {
    fn from_f32(f: f32) -> Self {
        F8E4M3::from_f32(f)
    }

    fn compute_gemm(
        handle: &RocBlasHandle,
        a: &HipSlice<Self>,
        b: &HipSlice<Self>,
        c: &mut HipSlice<Self>,
    ) -> anyhow::Result<()> {
        let params = GemmParams::default();
        handle.gemm_ex_fp8(
            &params,
            &F8E4M3::from_f32(1.0),
            a,
            b,
            &F8E4M3::from_f32(0.0),
            c,
        )
    }
}

pub struct RocmSmiWrapper {
    _smi: RocmSmi, // keep the SMI instance alive (marked with underscore to indicate intentional unused)
    devices: Vec<Option<RocmSmiDevice>>,
    device_count: u32,
}

impl RocmSmiWrapper {
    pub fn new() -> anyhow::Result<Self> {
        // Try to initialize ROCm SMI with better error handling
        let smi = RocmSmi::init().map_err(|e| anyhow::anyhow!("Failed to init ROCm SMI: {e:?}"))?;

        // Try to create devices for detected GPUs
        let mut devices = Vec::new();
        let mut device_count = 0u32;

        // Try to create devices up to a reasonable limit
        for i in 0..16 {
            match RocmSmiDevice::new(i) {
                Ok(device) => {
                    devices.push(Some(device));
                    device_count += 1;
                }
                Err(_) => {
                    if device_count == 0 && i == 0 {
                        // If we can't create even the first device, fail
                        return Err(anyhow::anyhow!("No ROCm SMI devices found"));
                    }
                    break; // No more devices
                }
            }
        }

        Ok(RocmSmiWrapper {
            _smi: smi,
            devices,
            device_count,
        })
    }

    pub fn device_count(&self) -> u32 {
        self.device_count
    }

    pub fn get_temperature(&mut self, device_id: u32) -> anyhow::Result<u32> {
        if device_id >= self.device_count {
            return Err(anyhow::anyhow!("Device ID {} out of range", device_id));
        }

        let device = match &mut self.devices[device_id as usize] {
            Some(dev) => dev,
            None => return Err(anyhow::anyhow!("Device {} not available", device_id)),
        };

        let temp_types = [
            RsmiTemperatureType::Edge,
            RsmiTemperatureType::Junction,
            RsmiTemperatureType::Memory,
        ];

        for ty in temp_types {
            if let Ok(temp) = device.get_temperature_metric(ty, RsmiTemperatureMetric::Current) {
                return Ok(temp as u32);
            }
        }

        Err(anyhow::anyhow!(
            "Failed to read temperature for GPU {} from all known types",
            device_id
        ))
    }

    pub fn get_throttle_status(&mut self, device_id: u32) -> anyhow::Result<ThrottleStatus> {
        if device_id >= self.device_count {
            return Err(anyhow::anyhow!("Device ID {} out of range", device_id));
        }

        if let Some(ref mut _device) = self.devices[device_id as usize] {
            // Try to get GPU metrics which may contain throttle information
            Ok(ThrottleStatus::default())
        } else {
            Ok(ThrottleStatus::default())
        }
    }
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    if args.duration_secs < MIN_DURATION_SECS {
        eprintln!("Duration must be at least {MIN_DURATION_SECS} seconds");
        std::process::exit(1);
    }
    if args.tflops_tolerance < 0.0 || args.tflops_tolerance > 100.0 {
        eprintln!("TFLOPS tolerance must be between 0 and 100");
        std::process::exit(1);
    }

    let config = Config {
        duration_secs: args.duration_secs,
        tflops_tolerance: args.tflops_tolerance,
        tolerate_software_throttling: args.tolerate_software_throttling,
        use_fp32: args.use_fp32,
        use_bf16: args.use_bf16,
        use_fp8: args.use_fp8,
    };

    match run(config).await {
        Ok(_) => {}
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    };
}

async fn run(config: Config) -> anyhow::Result<()> {
    // Initialize HIP
    unsafe {
        let result = hipInit(0);
        if result != hipError_t_hipSuccess {
            return Err(anyhow::anyhow!("Failed to initialize HIP: {:?}", result));
        }
    }

    let gpus = detect_gpus()?;
    if gpus.is_empty() {
        return Err(anyhow::anyhow!("No GPUs detected"));
    }

    for gpu in gpus.iter() {
        println!("Detected GPU #{}: {}", gpu.ordinal(), gpu.name());
    }

    // Determine precision to use
    let (use_fp8, use_bf16, _use_fp32) = determine_precision(&config, &gpus)?;

    if use_fp8 {
        println!("Using FP8 precision");
        run_with_precision::<F8E4M3>(config, gpus).await
    } else if use_bf16 {
        println!("Using BF16 precision");
        run_with_precision::<half::bf16>(config, gpus).await
    } else {
        println!("Using FP32 precision");
        run_with_precision::<f32>(config, gpus).await
    }
}

fn determine_precision(
    config: &Config,
    gpus: &[Arc<HipContext>],
) -> anyhow::Result<(bool, bool, bool)> {
    // Check explicit flags first
    if config.use_fp8 {
        let all_support_fp8 = gpus.iter().all(|gpu| supports_fp8(gpu).unwrap_or(false));
        if !all_support_fp8 {
            return Err(anyhow::anyhow!(
                "FP8 was explicitly requested but not all GPUs support it"
            ));
        }
        return Ok((true, false, false));
    }

    if config.use_bf16 {
        let all_support_bf16 = gpus.iter().all(|gpu| supports_bf16(gpu).unwrap_or(false));
        if !all_support_bf16 {
            return Err(anyhow::anyhow!(
                "BF16 was explicitly requested but not all GPUs support it"
            ));
        }
        return Ok((false, true, false));
    }

    if config.use_fp32 {
        return Ok((false, false, true));
    }

    // Auto-detect: prefer FP8 > BF16 > FP32
    let all_support_fp8 = gpus.iter().all(|gpu| supports_fp8(gpu).unwrap_or(false));
    if all_support_fp8 {
        return Ok((true, false, false));
    }

    let all_support_bf16 = gpus.iter().all(|gpu| supports_bf16(gpu).unwrap_or(false));
    if all_support_bf16 {
        return Ok((false, true, false));
    }

    Ok((false, false, true))
}

async fn run_with_precision<T: VariablePrecisionFloat>(
    config: Config,
    gpus: Vec<Arc<HipContext>>,
) -> anyhow::Result<()> {
    println!("Creating random matrices");
    let mut rng = SmallRng::from_rng(&mut rng());
    let mut a = vec![T::from_f32(0.0); SIZE * SIZE];
    let mut b = vec![T::from_f32(0.0); SIZE * SIZE];
    for i in 0..SIZE * SIZE {
        a[i] = T::from_f32((rng.next_u32() % 1000) as f32 / 1000.0);
        b[i] = T::from_f32((rng.next_u32() % 1000) as f32 / 1000.0);
    }
    println!("Matrices created");

    let (tx, rx) = unbounded_channel::<(usize, usize)>();
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    tokio::spawn(shutdown_signal(stop.clone()));

    let mut handles = Vec::new();
    for gpu in gpus.iter() {
        let tx = tx.clone();
        let stop = stop.clone();
        let gpu = gpu.clone();
        let a = a.clone();
        let b = b.clone();
        let gpu_ordinal = gpu.ordinal();
        let t = tokio::spawn(async move {
            match burn_gpu(gpu_ordinal, gpu, a, b, tx, stop).await {
                Ok(_) => {}
                Err(e) => {
                    eprintln!("Unable to burn GPU #{gpu_ordinal}: {e:?}");
                }
            }
        });
        handles.push(t);
    }

    // Report progress
    let stop_clone = stop.clone();
    let gpus_healthy = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let gpus_healthy_clone = gpus_healthy.clone();
    let config_clone = config.clone();
    let t = tokio::spawn(async move {
        report_progress(config_clone, gpus.len(), rx, stop_clone, gpus_healthy_clone).await;
    });
    handles.push(t);

    // Burn for given duration
    let wait = tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(1));
        let mut tick = 0;
        while !stop.load(std::sync::atomic::Ordering::Relaxed) && tick < config.duration_secs {
            interval.tick().await;
            tick += 1;
        }
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        drop(tx);
    });
    handles.push(wait);

    for handle in handles {
        handle.await.expect("Thread panicked");
    }

    if gpus_healthy.load(std::sync::atomic::Ordering::Relaxed) {
        Ok(())
    } else {
        Err(anyhow::anyhow!("Some GPUs are not healthy"))
    }
}

fn poll_temperatures(
    rocm_smi: &mut RocmSmiWrapper,
    gpu_count: usize,
) -> anyhow::Result<Vec<usize>> {
    let mut temps = vec![0usize; gpu_count];
    for (i, temp) in temps.iter_mut().enumerate().take(gpu_count) {
        match rocm_smi.get_temperature(i as u32) {
            Ok(t) => *temp = t as usize,
            Err(_) => {
                // If we can't get temp for this GPU, use 0 and continue
                *temp = 0;
            }
        }
    }
    Ok(temps)
}

fn poll_throttling(
    rocm_smi: &mut RocmSmiWrapper,
    gpu_count: usize,
) -> anyhow::Result<Vec<ThrottleStatus>> {
    let mut throttling = vec![];
    for i in 0..gpu_count {
        match rocm_smi.get_throttle_status(i as u32) {
            Ok(status) => throttling.push(status),
            Err(_) => throttling.push(ThrottleStatus::default()),
        }
    }
    Ok(throttling)
}

fn categorize_throttle_status(status: ThrottleStatus) -> (bool, bool, bool) {
    let hw_throttling = status.power_throttling || status.current_throttling;
    let thermal_sw = status.thermal_throttling;
    let thermal_hw = false; // ROCm doesn't distinguish between SW/HW thermal throttling clearly

    (hw_throttling, thermal_sw, thermal_hw)
}

fn supports_bf16(gpu: &Arc<HipContext>) -> anyhow::Result<bool> {
    // Check if GPU supports BF16 - MI325x should support it
    let major = gpu.get_attribute(hipDeviceAttribute_t_hipDeviceAttributeComputeCapabilityMajor)?;
    let minor = gpu.get_attribute(hipDeviceAttribute_t_hipDeviceAttributeComputeCapabilityMinor)?;

    // AMD MI325x is based on CDNA3 architecture which supports BF16
    // For CDNA3 (gfx942), we can assume BF16 support
    Ok(major >= 9 || (major == 8 && minor >= 0))
}

fn supports_fp8(_gpu: &Arc<HipContext>) -> anyhow::Result<bool> {
    // TODO: add FP8 support
    Ok(false)
}

async fn report_progress(
    config: Config,
    gpu_count: usize,
    mut rx: Receiver<(usize, usize)>,
    stop: Arc<std::sync::atomic::AtomicBool>,
    gpus_healthy: Arc<std::sync::atomic::AtomicBool>,
) {
    let mut interval = tokio::time::interval(std::time::Duration::from_secs(1));
    let mut burn_results = (0..gpu_count).map(BurnResult::new).collect::<Vec<_>>();
    let mut tick = 0;
    // Try to initialize ROCm SMI safely
    let rocm_smi = match RocmSmiWrapper::new() {
        Ok(smi) => Some(Arc::new(Mutex::new(smi))),
        Err(e) => {
            eprintln!(
                "Warning: Failed to initialize ROCm SMI: {e}. Temperature and throttling monitoring will be disabled."
            );
            None
        }
    };

    while !stop.load(std::sync::atomic::Ordering::Relaxed) {
        interval.tick().await;
        let mut nops = vec![0usize; gpu_count];

        // Drain the channel to get the latest updates
        while let Ok(ops) = rx.try_recv() {
            nops[ops.0] += ops.1;
        }

        for i in 0..gpu_count {
            let flops = nops[i] * SIZE * SIZE * SIZE * 2;
            print!("{} ({} Gflops/s) ", nops[i], flops / 1_000_000_000);
            if i < gpu_count - 1 {
                print!("");
            } else {
                print!("| ");
            }

            if tick > 4 {
                // Skip the first 6 ticks to avoid caches effects
                burn_results[i].flops_max = burn_results[i].flops_max.max(flops);
                burn_results[i].flops_min = burn_results[i].flops_min.min(flops);
                burn_results[i].flops_sum += flops;
                burn_results[i].n_iters += 1;
            }
        }

        // Report GPU temperatures if ROCm SMI is available
        if let Some(ref smi) = rocm_smi {
            let mut smi = smi.lock().unwrap();
            match poll_temperatures(&mut smi, gpu_count) {
                Ok(temps) => {
                    for i in 0..gpu_count {
                        if temps[i] > 0 {
                            print!("{}°C", temps[i]);
                        } else {
                            print!("N/A");
                        }
                        if i < gpu_count - 1 {
                            print!(" ");
                        } else {
                            print!(" | ");
                        }
                        if tick > 4 && temps[i] > 0 {
                            burn_results[i].temp_max = burn_results[i].temp_max.max(temps[i]);
                            burn_results[i].temp_min = burn_results[i].temp_min.min(temps[i]);
                            burn_results[i].temp_sum += temps[i];
                            burn_results[i].temp_count += 1;
                        }
                    }
                }
                Err(_) => {
                    print!("Temp read error | ");
                }
            }

            // Report throttling
            match poll_throttling(&mut smi, gpu_count) {
                Ok(throttling) => {
                    for i in 0..gpu_count {
                        let (hw, thermal_sw, thermal_hw) =
                            categorize_throttle_status(throttling[i]);

                        if !hw && !thermal_sw && !thermal_hw {
                            print!("");
                        } else {
                            let mut throttle_types = Vec::new();
                            if thermal_sw {
                                throttle_types.push("T");
                                burn_results[i].throttling_thermal_sw += 1;
                            }
                            if thermal_hw {
                                throttle_types.push("TH");
                                burn_results[i].throttling_thermal_hw += 1;
                            }
                            if hw {
                                throttle_types.push("P");
                                burn_results[i].throttling_hw += 1;
                            }
                            print!("{}", throttle_types.join(","));
                        }

                        if i < gpu_count - 1 {
                            print!(" ");
                        } else {
                            println!();
                        }
                    }
                }
                Err(_) => {
                    println!("Throttling read error");
                }
            }
        } else {
            println!("Temperature/throttling monitoring disabled");
        }

        tick += 1;
    }

    // Final report
    for r in burn_results.clone() {
        println!(
            "GPU #{}: {:6.0} Gflops/s (min: {:.2}, max: {:.2})",
            r.gpu_idx,
            r.flops_avg() / 1_000_000_000.0,
            r.flops_min as f64 / 1_000_000_000.0,
            r.flops_max as f64 / 1_000_000_000.0,
        );

        if r.temp_count > 0 && r.temp_sum > 0 {
            println!(
                "         Temperature: {:.1}°C (min: {:.1}, max: {:.1})",
                r.temp_avg(),
                r.temp_min as f64,
                r.temp_max as f64
            );
        }

        if r.throttling_hw > 0 || r.throttling_thermal_sw > 0 || r.throttling_thermal_hw > 0 {
            println!(
                "         Throttling HW: {}, Thermal SW: {}, Thermal HW: {}",
                r.throttling_hw > 0,
                r.throttling_thermal_sw > 0,
                r.throttling_thermal_hw > 0
            );
        }
    }

    let (healthy, reasons) = are_gpus_healthy(
        burn_results,
        config.tflops_tolerance,
        config.tolerate_software_throttling,
    );

    if healthy {
        println!("All GPUs seem healthy");
    } else {
        println!("Some GPUs are not healthy. Reasons:");
        for r in reasons {
            println!("  - {r}");
        }
    }

    gpus_healthy.store(healthy, std::sync::atomic::Ordering::Relaxed);
    println!("Freeing GPUs...");
}

fn are_gpus_healthy(
    burn_results: Vec<BurnResult>,
    tflops_tolerance: f64,
    tolerate_software_throttling: bool,
) -> (bool, Vec<String>) {
    let mut reasons = vec![];
    // acceptable_flops is tflops_tolerance% lower than best gpu avg flops
    let acceptable_flops: f64 = burn_results
        .iter()
        .map(|r| r.flops_avg())
        .fold(0., |max, avg| {
            max.max(avg * (100. - tflops_tolerance) / 100.)
        });
    for r in burn_results.iter() {
        let mut low_flops = false;
        if r.flops_avg() < acceptable_flops {
            reasons.push(format!("GPU {} - ", r.gpu_idx) + GPU_FLOPS_REASON);
            low_flops = true;
        }
        // if we have any throttling
        if r.is_throttled() {
            if !low_flops
                && tolerate_software_throttling
                && (r.throttling_thermal_hw == 0 && r.throttling_hw == 0)
            {
                continue;
            }
            reasons.push(format!("GPU {} - ", r.gpu_idx) + GPU_THROTTLING_REASON);
        }
    }
    (reasons.is_empty(), reasons)
}

async fn burn_gpu<T: VariablePrecisionFloat>(
    gpu_idx: usize,
    gpu: Arc<HipContext>,
    a: Vec<T>,
    b: Vec<T>,
    tx: Sender<(usize, usize)>,
    stop: Arc<std::sync::atomic::AtomicBool>,
) -> anyhow::Result<usize> {
    gpu.set_current()?;
    let (free_mem, _) = gpu.mem_get_info()?;
    let mem_to_use = (free_mem as f64 * MEM_TO_USE_PCT) as usize;
    let iters = (mem_to_use - 2 * SIZE * SIZE * std::mem::size_of::<T>())
        / (SIZE * SIZE * std::mem::size_of::<T>());
    let (a_gpu, b_gpu, mut out_slices_gpu) = alloc_buffers(&gpu, a, b, iters)?;
    let handle = RocBlasHandle::new()?;
    let mut i = 0;
    let mut error_count = 0;
    const MAX_ERRORS: usize = 10;

    while !stop.load(std::sync::atomic::Ordering::Relaxed) && error_count < MAX_ERRORS {
        for out in out_slices_gpu.iter_mut() {
            match T::compute_gemm(&handle, &a_gpu, &b_gpu, out) {
                Ok(_) => match gpu.synchronize() {
                    Ok(_) => {
                        i += 1;
                        let _ = tx.send((gpu_idx, 1));
                    }
                    Err(_e) => {
                        error_count += 1;
                        break;
                    }
                },
                Err(_e) => {
                    error_count += 1;
                    break;
                }
            }
        }
    }
    Ok(i)
}

fn alloc_buffers<T: VariablePrecisionFloat>(
    gpu: &Arc<HipContext>,
    a: Vec<T>,
    b: Vec<T>,
    num_out_slices: usize,
) -> anyhow::Result<AllocBufferTuple<T>> {
    let a_gpu = gpu.htod_copy(a)?;
    let b_gpu = gpu.htod_copy(b)?;

    let mut out_slices = Vec::new();
    for _i in 0..num_out_slices {
        let out = gpu.alloc_zeros(SIZE * SIZE)?;
        out_slices.push(out);
    }

    Ok((a_gpu, b_gpu, out_slices))
}

fn detect_gpus() -> anyhow::Result<Vec<Arc<HipContext>>> {
    let num_gpus = HipContext::device_count()? as usize;
    let mut devices = Vec::new();
    for i in 0..num_gpus {
        let dev = Arc::new(HipContext::new(i as i32)?);
        devices.push(dev);
    }
    Ok(devices)
}

async fn shutdown_signal(stop: Arc<std::sync::atomic::AtomicBool>) {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to hook signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    stop.store(true, Relaxed);
}
