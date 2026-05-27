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

mod sys;

use clap::{Parser, ValueEnum};
use rocm_smi_lib::{RocmSmi, RocmSmiDevice, RsmiTemperatureMetric, RsmiTemperatureType};
use std::ffi::c_void;
use std::os::fd::RawFd;
use std::ptr;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, Barrier, Mutex, OnceLock};
use tokio::signal;
use tokio::sync::mpsc::{
    UnboundedReceiver as Receiver, UnboundedSender as Sender, unbounded_channel,
};

// Default square matrix size. 16384 is a good baseline on MI355X: large enough
// to keep every matrix core busy while small enough that per-kernel variance
// fits inside our 1s reporting tick. Override with --size to sweep toward peak;
// bigger N raises matrix-core occupancy and arithmetic intensity.
const DEFAULT_SIZE: usize = 16384;
const MEM_TO_USE_PCT: f64 = 0.85;
const MIN_DURATION_SECS: u64 = 10;
const WORKSPACE_BYTES: usize = 128 * 1024 * 1024; // 128 MiB workspace per hipBLASLt handle
// Number of independent streams (and matching output buffers) per GPU.
// Each stream keeps one matmul in flight while we sync the next. On
// MI355X / hipBLASLt 1.2.2, depth 4 measured ~8% higher throughput than
// single-stream submit-no-sync — host launch overhead is non-trivial here.
const PIPELINE_DEPTH: usize = 4;
// Number of algos to ask hipBLASLt for during tuning; we time each in the
// prewarm as a sanity check. The heuristic returns algos in order of
// estimated throughput so the first one is almost always the winner.
//
// NOTE: we deliberately burn with algos[0] (the heuristic's first pick), NOT a
// per-GPU "fastest by timing" choice. Two reasons, both learned the hard way:
// (1) single-shot matmul timing is overhead-dominated and mis-ranks vs. the
// pipelined steady state, and (2) actually executing arbitrary FP8 candidates
// can hang the GPU (rocRoller JIT/exec hangs). algos[0] is the stable kernel.
const TUNE_ALGOS: usize = 4;

const GPU_FLOPS_REASON: &str =
    "GPU is not performing as expected. Check the flops values and temperatures";

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum Precision {
    Fp32,
    Bf16,
    /// OCP FP8 (E4M3)
    Fp8,
}

impl Precision {
    fn name(&self) -> &'static str {
        match self {
            Precision::Fp32 => "FP32",
            Precision::Bf16 => "BF16",
            Precision::Fp8 => "FP8 (E4M3)",
        }
    }

    fn bytes_per_element(&self) -> usize {
        match self {
            Precision::Fp32 => 4,
            Precision::Bf16 => 2,
            Precision::Fp8 => 1,
        }
    }

    fn matrix_bytes(&self, n: usize) -> usize {
        n * n * self.bytes_per_element()
    }

    fn data_type(&self) -> sys::hipDataType {
        match self {
            Precision::Fp32 => sys::HIP_R_32F,
            Precision::Bf16 => sys::HIP_R_16BF,
            Precision::Fp8 => sys::HIP_R_8F_E4M3,
        }
    }

    /// Output ("D") tensor type. We always accumulate in f32, then write the
    /// result out as the smallest precision that doesn't cost us matrix-core
    /// throughput. For FP8 inputs the convention (matches HuggingFace's CUDA
    /// gpu-fryer and AMD's hipBLASLt samples) is to write BF16 — that's 4x
    /// less writeback bandwidth than f32 and lets the next matmul start sooner.
    fn output_data_type(&self) -> sys::hipDataType {
        match self {
            Precision::Fp32 => sys::HIP_R_32F,
            Precision::Bf16 | Precision::Fp8 => sys::HIP_R_16BF,
        }
    }

    fn output_bytes_per_element(&self) -> usize {
        match self.output_data_type() {
            x if x == sys::HIP_R_32F => 4,
            x if x == sys::HIP_R_16BF => 2,
            _ => 4,
        }
    }

    fn output_matrix_bytes(&self, n: usize) -> usize {
        n * n * self.output_bytes_per_element()
    }

    /// FP8 matmul kernels on CDNA4 only ship a tuned path for the TN data
    /// layout (transA = T, transB = N); NN works but at lower throughput.
    fn prefers_tn_layout(&self) -> bool {
        matches!(self, Precision::Fp8)
    }
}

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Duration in seconds to burn the GPUs.
    #[clap(default_value = "60")]
    duration_secs: u64,
    /// Pass threshold: each GPU must be within this % of the best GPU's average.
    #[clap(long, default_value = "10")]
    tflops_tolerance: f64,
    /// Numeric format used for the burn GEMM. Defaults to the highest-throughput
    /// format every visible GPU supports.
    #[clap(long, value_enum)]
    precision: Option<Precision>,
    /// Square GEMM dimension N (the burn runs an N×N×N matmul). Larger N raises
    /// matrix-core occupancy and arithmetic intensity. Must be a multiple of 64.
    #[clap(long, default_value_t = DEFAULT_SIZE)]
    size: usize,
}

#[derive(Debug, Clone)]
struct BurnResult {
    gpu_idx: usize,
    flops_max: f64,
    flops_min: f64,
    flops_sum: f64,
    n_iters: usize,
    temp_max: u32,
    temp_sum: u64,
    temp_min: u32,
    temp_count: u32,
}

impl BurnResult {
    fn new(gpu_idx: usize) -> Self {
        Self {
            gpu_idx,
            flops_max: 0.0,
            flops_min: f64::INFINITY,
            flops_sum: 0.0,
            n_iters: 0,
            temp_max: 0,
            temp_sum: 0,
            temp_min: u32::MAX,
            temp_count: 0,
        }
    }

    fn flops_avg(&self) -> f64 {
        if self.n_iters == 0 {
            0.0
        } else {
            self.flops_sum / self.n_iters as f64
        }
    }

    fn temp_avg(&self) -> f64 {
        if self.temp_count == 0 {
            0.0
        } else {
            self.temp_sum as f64 / self.temp_count as f64
        }
    }
}

#[derive(Debug, Clone)]
struct Config {
    duration_secs: u64,
    tflops_tolerance: f64,
    precision: Precision,
    /// Square GEMM dimension N for this run (the burn does N×N×N).
    size: usize,
    /// Algorithm bytes selected by the prewarm tuner. Cloned into every burn
    /// thread so they share the same chosen kernel.
    tuned_algo: sys::hipblasLtMatmulAlgo_t,
}

// --- stderr suppression -------------------------------------------------------

/// RAII guard that redirects fd 2 (stderr) to /dev/null for its lifetime.
///
/// hipBLASLt's `origami` latency-prediction model unconditionally writes a
/// "Warning: Latency not found ... mi_input_type=BFloat8Float8_fnuz" line to
/// `std::cerr` for every FP8 candidate solution it scores inside
/// `hipblasLtMatmulAlgoGetHeuristic`. No env var or log level gates it, so the
/// only way to keep the console readable is to mute fd 2 across the heuristic
/// call. Our progress output is on stdout, so it is never affected; the burn
/// loop runs an explicit algo (no heuristic), so it never re-triggers the warning.
///
/// A process-global mutex serializes the dup2 save/restore so concurrent
/// per-GPU heuristic queries can't race on the shared fd. `new` returns `None`
/// if the redirect can't be set up, in which case we simply let the warnings through.
struct StderrSilencer {
    saved: RawFd,
    _lock: std::sync::MutexGuard<'static, ()>,
}

impl StderrSilencer {
    fn new() -> Option<Self> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let lock = LOCK.get_or_init(|| Mutex::new(())).lock().unwrap();
        use std::io::Write;
        let _ = std::io::stderr().flush();
        unsafe {
            let saved = libc::dup(libc::STDERR_FILENO);
            if saved < 0 {
                return None;
            }
            let devnull = libc::open(c"/dev/null".as_ptr(), libc::O_WRONLY);
            if devnull < 0 {
                libc::close(saved);
                return None;
            }
            let rc = libc::dup2(devnull, libc::STDERR_FILENO);
            libc::close(devnull);
            if rc < 0 {
                libc::close(saved);
                return None;
            }
            Some(Self { saved, _lock: lock })
        }
    }
}

impl Drop for StderrSilencer {
    fn drop(&mut self) {
        unsafe {
            libc::dup2(self.saved, libc::STDERR_FILENO);
            libc::close(self.saved);
        }
    }
}

// --- HIP wrappers -------------------------------------------------------------

fn hip_check(rc: sys::hipError_t, ctx: &str) -> anyhow::Result<()> {
    if rc == sys::hipSuccess {
        Ok(())
    } else {
        Err(anyhow::anyhow!("{ctx} failed with HIP error {rc}"))
    }
}

fn lt_check(rc: sys::hipblasStatus_t, ctx: &str) -> anyhow::Result<()> {
    if rc == sys::HIPBLAS_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(anyhow::anyhow!("{ctx} failed with hipBLAS status {rc}"))
    }
}

pub struct DeviceBuffer {
    ptr: sys::hipDeviceptr_t,
    bytes: usize,
}

impl DeviceBuffer {
    pub fn alloc_filled(bytes: usize, fill_byte: u8) -> anyhow::Result<Self> {
        unsafe {
            let mut ptr: sys::hipDeviceptr_t = ptr::null_mut();
            hip_check(sys::hipMalloc(&mut ptr, bytes), "hipMalloc")?;
            hip_check(
                sys::hipMemset(ptr, fill_byte as i32, bytes),
                "hipMemset (fill)",
            )?;
            Ok(Self { ptr, bytes })
        }
    }

    pub fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        unsafe {
            let _ = sys::hipFree(self.ptr);
        }
        self.bytes = 0;
    }
}

pub struct HipContext {
    device_id: i32,
    name: String,
    /// ISA arch (e.g. "gfx950"). Resolvable even when `name` is the generic
    /// "AMD Radeon Graphics" containers report; used as a peak-table fallback.
    arch: Option<String>,
}

impl HipContext {
    pub fn new(device_id: i32) -> anyhow::Result<Self> {
        unsafe {
            hip_check(sys::hipSetDevice(device_id), "hipSetDevice")?;
            let mut buf = [0i8; 256];
            hip_check(
                sys::hipDeviceGetName(buf.as_mut_ptr(), buf.len() as i32, device_id),
                "hipDeviceGetName",
            )?;
            let name = std::ffi::CStr::from_ptr(buf.as_ptr())
                .to_string_lossy()
                .into_owned();
            let arch = sys::device_arch(device_id);
            Ok(HipContext {
                device_id,
                name,
                arch,
            })
        }
    }

    pub fn device_count() -> anyhow::Result<i32> {
        unsafe {
            let mut count = 0;
            hip_check(sys::hipGetDeviceCount(&mut count), "hipGetDeviceCount")?;
            Ok(count)
        }
    }

    pub fn ordinal(&self) -> usize {
        self.device_id as usize
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn arch(&self) -> Option<&str> {
        self.arch.as_deref()
    }

    /// Lowercased string for peak-table matching: the marketing name plus the
    /// ISA arch, so a generic "AMD Radeon Graphics" still matches on "gfx950".
    pub fn match_key(&self) -> String {
        match &self.arch {
            Some(arch) => format!("{} {}", self.name.to_lowercase(), arch),
            None => self.name.to_lowercase(),
        }
    }

    pub fn set_current(&self) -> anyhow::Result<()> {
        unsafe { hip_check(sys::hipSetDevice(self.device_id), "hipSetDevice") }
    }

    pub fn mem_get_info(&self) -> anyhow::Result<(usize, usize)> {
        self.set_current()?;
        unsafe {
            let mut free: usize = 0;
            let mut total: usize = 0;
            hip_check(sys::hipMemGetInfo(&mut free, &mut total), "hipMemGetInfo")?;
            Ok((free, total))
        }
    }
}

// --- hipBLASLt wrappers --------------------------------------------------------

struct LtHandle(sys::hipblasLtHandle_t);

impl LtHandle {
    fn new() -> anyhow::Result<Self> {
        unsafe {
            let mut h: sys::hipblasLtHandle_t = ptr::null_mut();
            lt_check(sys::hipblasLtCreate(&mut h), "hipblasLtCreate")?;
            Ok(Self(h))
        }
    }
}

impl Drop for LtHandle {
    fn drop(&mut self) {
        unsafe {
            let _ = sys::hipblasLtDestroy(self.0);
        }
    }
}

struct MatLayout(sys::hipblasLtMatrixLayout_t);

impl MatLayout {
    fn new(dtype: sys::hipDataType, rows: u64, cols: u64, ld: i64) -> anyhow::Result<Self> {
        unsafe {
            let mut l: sys::hipblasLtMatrixLayout_t = ptr::null_mut();
            lt_check(
                sys::hipblasLtMatrixLayoutCreate(&mut l, dtype, rows, cols, ld),
                "hipblasLtMatrixLayoutCreate",
            )?;
            Ok(Self(l))
        }
    }
}

impl Drop for MatLayout {
    fn drop(&mut self) {
        unsafe {
            let _ = sys::hipblasLtMatrixLayoutDestroy(self.0);
        }
    }
}

struct MatmulDesc(sys::hipblasLtMatmulDesc_t);

impl MatmulDesc {
    fn new(compute: sys::hipblasComputeType_t, scale: sys::hipDataType) -> anyhow::Result<Self> {
        unsafe {
            let mut d: sys::hipblasLtMatmulDesc_t = ptr::null_mut();
            lt_check(
                sys::hipblasLtMatmulDescCreate(&mut d, compute, scale),
                "hipblasLtMatmulDescCreate",
            )?;
            Ok(Self(d))
        }
    }

    fn set_i32(
        &self,
        attr: sys::hipblasLtMatmulDescAttributes_t,
        value: i32,
    ) -> anyhow::Result<()> {
        unsafe {
            lt_check(
                sys::hipblasLtMatmulDescSetAttribute(
                    self.0,
                    attr,
                    &value as *const i32 as *const c_void,
                    std::mem::size_of::<i32>(),
                ),
                "hipblasLtMatmulDescSetAttribute(i32)",
            )
        }
    }
}

impl Drop for MatmulDesc {
    fn drop(&mut self) {
        unsafe {
            let _ = sys::hipblasLtMatmulDescDestroy(self.0);
        }
    }
}

struct MatmulPref(sys::hipblasLtMatmulPreference_t);

impl MatmulPref {
    fn new(max_workspace_bytes: u64) -> anyhow::Result<Self> {
        unsafe {
            let mut p: sys::hipblasLtMatmulPreference_t = ptr::null_mut();
            lt_check(
                sys::hipblasLtMatmulPreferenceCreate(&mut p),
                "hipblasLtMatmulPreferenceCreate",
            )?;
            lt_check(
                sys::hipblasLtMatmulPreferenceSetAttribute(
                    p,
                    sys::HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                    &max_workspace_bytes as *const u64 as *const c_void,
                    std::mem::size_of::<u64>(),
                ),
                "hipblasLtMatmulPreferenceSetAttribute(workspace)",
            )?;
            Ok(Self(p))
        }
    }
}

impl Drop for MatmulPref {
    fn drop(&mut self) {
        unsafe {
            let _ = sys::hipblasLtMatmulPreferenceDestroy(self.0);
        }
    }
}

struct Stream(sys::hipStream_t);

impl Stream {
    fn new() -> anyhow::Result<Self> {
        unsafe {
            let mut s: sys::hipStream_t = ptr::null_mut();
            hip_check(sys::hipStreamCreate(&mut s), "hipStreamCreate")?;
            Ok(Self(s))
        }
    }

    fn sync(&self) -> anyhow::Result<()> {
        unsafe { hip_check(sys::hipStreamSynchronize(self.0), "hipStreamSynchronize") }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe {
            let _ = sys::hipStreamDestroy(self.0);
        }
    }
}

// --- SMI wrapper --------------------------------------------------------------

pub struct RocmSmiWrapper {
    _smi: RocmSmi,
    devices: Vec<Option<RocmSmiDevice>>,
    device_count: u32,
}

impl RocmSmiWrapper {
    pub fn new() -> anyhow::Result<Self> {
        let smi = RocmSmi::init().map_err(|e| anyhow::anyhow!("Failed to init ROCm SMI: {e:?}"))?;
        let mut devices = Vec::new();
        let mut device_count = 0u32;
        for i in 0..16 {
            match RocmSmiDevice::new(i) {
                Ok(device) => {
                    devices.push(Some(device));
                    device_count += 1;
                }
                Err(_) => break,
            }
        }
        if device_count == 0 {
            return Err(anyhow::anyhow!("No ROCm SMI devices found"));
        }
        Ok(RocmSmiWrapper {
            _smi: smi,
            devices,
            device_count,
        })
    }

    pub fn get_temperature(&mut self, device_id: u32) -> anyhow::Result<u32> {
        if device_id >= self.device_count {
            return Err(anyhow::anyhow!("Device ID {device_id} out of range"));
        }
        let device = self.devices[device_id as usize]
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Device {device_id} not available"))?;
        for ty in [
            RsmiTemperatureType::Junction,
            RsmiTemperatureType::Edge,
            RsmiTemperatureType::Memory,
        ] {
            if let Ok(t) = device.get_temperature_metric(ty, RsmiTemperatureMetric::Current) {
                return Ok(t as u32);
            }
        }
        Err(anyhow::anyhow!(
            "No temperature sensor responded for GPU {device_id}"
        ))
    }
}

// --- Peak throughput table ----------------------------------------------------

/// Returns dense peak in FLOP/s for (match key, precision) or `None` if we don't
/// have a published spec for the part. `key` is `HipContext::match_key()` — the
/// lowercased marketing name plus the ISA arch (e.g. "gfx950").
fn peak_flops_per_sec(key: &str, precision: Precision) -> Option<f64> {
    let n = key.to_lowercase();
    // Each entry: marketing-name substring + ISA arch, and peak in PFLOP/s for
    // each precision. Sourced from AMD product briefs (dense, no structured
    // sparsity). MI355X numbers from the user-supplied datasheet.
    struct Spec {
        match_str: &'static str,
        arch: &'static str,
        fp32: f64,
        bf16: f64,
        fp8: f64,
    }
    const SPECS: &[Spec] = &[
        Spec {
            match_str: "mi355",
            arch: "gfx950",
            fp32: 0.1573,
            bf16: 2.5,
            fp8: 5.0,
        },
        Spec {
            match_str: "mi325",
            arch: "gfx942",
            fp32: 0.1633,
            bf16: 2.6,
            fp8: 5.22,
        },
        Spec {
            match_str: "mi300x",
            arch: "gfx942",
            fp32: 0.1633,
            bf16: 2.6,
            fp8: 5.22,
        },
        Spec {
            match_str: "mi250",
            arch: "gfx90a",
            fp32: 0.0479,
            bf16: 0.383,
            fp8: 0.0,
        },
        Spec {
            match_str: "mi210",
            arch: "gfx90a",
            fp32: 0.0226,
            bf16: 0.181,
            fp8: 0.0,
        },
    ];

    // Prefer the marketing-name match (exact per-SKU); fall back to the ISA arch
    // when the name is unresolved (containers report "AMD Radeon Graphics").
    // Arch is coarser — gfx942 maps MI300X≡MI325X (identical peaks) and gfx90a
    // resolves to the first listed match — but it's enough to pick precision.
    let spec = SPECS
        .iter()
        .find(|s| n.contains(s.match_str))
        .or_else(|| SPECS.iter().find(|s| n.contains(s.arch)))?;
    let v = match precision {
        Precision::Fp32 => spec.fp32,
        Precision::Bf16 => spec.bf16,
        Precision::Fp8 => spec.fp8,
    };
    if v > 0.0 { Some(v * 1e15) } else { None }
}

fn precision_supported(key: &str, precision: Precision) -> bool {
    peak_flops_per_sec(key, precision)
        .map(|p| p > 0.0)
        .unwrap_or(false)
}

fn auto_precision(gpus: &[Arc<HipContext>]) -> Precision {
    // Prefer the highest-throughput format every GPU supports.
    for p in [Precision::Fp8, Precision::Bf16] {
        if gpus.iter().all(|g| precision_supported(&g.match_key(), p)) {
            return p;
        }
    }
    Precision::Fp32
}

// --- Entry point --------------------------------------------------------------

#[tokio::main]
async fn main() {
    let args = Args::parse();
    if args.duration_secs < MIN_DURATION_SECS {
        eprintln!("Duration must be at least {MIN_DURATION_SECS} seconds");
        std::process::exit(1);
    }
    if !(0.0..=100.0).contains(&args.tflops_tolerance) {
        eprintln!("TFLOPS tolerance must be between 0 and 100");
        std::process::exit(1);
    }
    if args.size < 1024 || args.size % 64 != 0 {
        eprintln!("--size must be a multiple of 64 and at least 1024");
        std::process::exit(1);
    }

    if let Err(e) = run(args).await {
        eprintln!("Error: {e:#}");
        std::process::exit(1);
    }
}

async fn run(args: Args) -> anyhow::Result<()> {
    unsafe { hip_check(sys::hipInit(0), "hipInit")? };

    let gpus = detect_gpus()?;
    if gpus.is_empty() {
        return Err(anyhow::anyhow!("No GPUs detected"));
    }
    for gpu in &gpus {
        match gpu.arch() {
            Some(arch) => println!("Detected GPU #{}: {} ({arch})", gpu.ordinal(), gpu.name()),
            None => println!("Detected GPU #{}: {}", gpu.ordinal(), gpu.name()),
        }
    }

    let precision = match args.precision {
        Some(p) => {
            for gpu in &gpus {
                if !precision_supported(&gpu.match_key(), p) {
                    return Err(anyhow::anyhow!(
                        "Precision {} is not supported on {} ({})",
                        p.name(),
                        gpu.name(),
                        gpu.arch().unwrap_or("unknown arch"),
                    ));
                }
            }
            p
        }
        None => auto_precision(&gpus),
    };

    let size = args.size;
    println!("Using {} precision, {size}x{size} GEMM", precision.name());

    // hipBLASLt's first-call JIT holds a process-wide comgr lock. If every
    // GPU races into it simultaneously they serialize. Warming a single GPU
    // first populates the on-disk comgr cache so the parallel warmups that
    // follow hit it instantly. We also use this pass to time several candidate
    // algorithms and report the fastest one.
    let tuned_algo = prewarm_kernel_cache(&gpus[0], precision, size)?;

    let config = Config {
        duration_secs: args.duration_secs,
        tflops_tolerance: args.tflops_tolerance,
        precision,
        size,
        tuned_algo,
    };

    let (tx, rx) = unbounded_channel::<(usize, usize)>();
    let stop = Arc::new(AtomicBool::new(false));
    tokio::spawn(shutdown_signal(stop.clone()));

    // Gate all burn threads behind a barrier so the first call into hipBLASLt
    // (which JITs kernels) happens on every GPU before any of them start their
    // measured burn loop. Without this, slow JIT serialization on the
    // process-wide tuning cache makes some GPUs ramp up many seconds after
    // others, which both pollutes the measurement window and looks broken.
    let warmup_barrier = Arc::new(Barrier::new(gpus.len()));

    let mut handles = Vec::new();
    for gpu in gpus.iter() {
        let tx = tx.clone();
        let stop = stop.clone();
        let gpu = gpu.clone();
        let config = config.clone();
        let barrier = warmup_barrier.clone();
        let gpu_ordinal = gpu.ordinal();
        let t = std::thread::spawn(move || {
            if let Err(e) = burn_gpu(gpu_ordinal, gpu, config, tx, stop, barrier) {
                eprintln!("Unable to burn GPU #{gpu_ordinal}: {e:#}");
            }
        });
        handles.push(t);
    }
    drop(tx);

    let gpus_healthy = Arc::new(AtomicBool::new(true));
    let gpus_healthy_clone = gpus_healthy.clone();
    let stop_clone = stop.clone();
    // Match keys (name + arch), not display names: report_progress uses these
    // only for the peak-of-spec lookup, which must work on generic names too.
    let gpu_keys: Vec<String> = gpus.iter().map(|g| g.match_key()).collect();
    let config_for_report = config.clone();
    let progress = tokio::spawn(async move {
        report_progress(
            config_for_report,
            gpu_keys,
            rx,
            stop_clone,
            gpus_healthy_clone,
        )
        .await;
    });

    for handle in handles {
        if let Err(e) = handle.join() {
            eprintln!("burn thread panicked: {e:?}");
        }
    }
    progress.await.ok();

    if gpus_healthy.load(Relaxed) {
        Ok(())
    } else {
        Err(anyhow::anyhow!("Some GPUs are not healthy"))
    }
}

struct DescriptorSet {
    handle: LtHandle,
    a: DeviceBuffer,
    b: DeviceBuffer,
    workspace: DeviceBuffer,
    a_layout: MatLayout,
    b_layout: MatLayout,
    cd_layout: MatLayout,
    desc: MatmulDesc,
    pref: MatmulPref,
}

fn build_descriptors(precision: Precision, size: usize) -> anyhow::Result<DescriptorSet> {
    let matrix_bytes = precision.matrix_bytes(size);

    let a = DeviceBuffer::alloc_filled(matrix_bytes, 0x11)?;
    let b = DeviceBuffer::alloc_filled(matrix_bytes, 0x22)?;
    let workspace = DeviceBuffer::alloc_filled(WORKSPACE_BYTES, 0)?;

    let handle = LtHandle::new()?;
    let a_layout = MatLayout::new(precision.data_type(), size as u64, size as u64, size as i64)?;
    let b_layout = MatLayout::new(precision.data_type(), size as u64, size as u64, size as i64)?;
    let cd_layout = MatLayout::new(
        precision.output_data_type(),
        size as u64,
        size as u64,
        size as i64,
    )?;

    let desc = MatmulDesc::new(sys::HIPBLAS_COMPUTE_32F, sys::HIP_R_32F)?;
    let trans_a = if precision.prefers_tn_layout() {
        sys::HIPBLAS_OP_T
    } else {
        sys::HIPBLAS_OP_N
    };
    desc.set_i32(sys::HIPBLASLT_MATMUL_DESC_TRANSA, trans_a)?;
    desc.set_i32(sys::HIPBLASLT_MATMUL_DESC_TRANSB, sys::HIPBLAS_OP_N)?;
    let pref = MatmulPref::new(WORKSPACE_BYTES as u64)?;
    Ok(DescriptorSet {
        handle,
        a,
        b,
        workspace,
        a_layout,
        b_layout,
        cd_layout,
        desc,
        pref,
    })
}

/// Time a single matmul + sync on the given stream with the given algo.
fn time_matmul(
    set: &DescriptorSet,
    algo: &sys::hipblasLtMatmulAlgo_t,
    out: &DeviceBuffer,
    stream: &Stream,
) -> anyhow::Result<f64> {
    use std::time::Instant;
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let start = Instant::now();
    unsafe {
        lt_check(
            sys::hipblasLtMatmul(
                set.handle.0,
                set.desc.0,
                &alpha as *const f32 as *const c_void,
                set.a.as_ptr(),
                set.a_layout.0,
                set.b.as_ptr(),
                set.b_layout.0,
                &beta as *const f32 as *const c_void,
                out.as_ptr(),
                set.cd_layout.0,
                out.as_ptr(),
                set.cd_layout.0,
                algo,
                set.workspace.as_ptr(),
                WORKSPACE_BYTES,
                stream.0,
            ),
            "hipblasLtMatmul (tune)",
        )?;
    }
    stream.sync()?;
    Ok(start.elapsed().as_secs_f64())
}

/// Allocate the minimum buffers needed to run one matmul of the requested
/// precision on a single GPU, then ask hipBLASLt for up to `TUNE_ALGOS`
/// candidate algorithms and time each one. Returns the algorithm with the
/// highest measured throughput, and populates the on-disk comgr cache so the
/// parallel burn threads don't all stall on the process-wide compile lock.
fn prewarm_kernel_cache(
    gpu: &HipContext,
    precision: Precision,
    size: usize,
) -> anyhow::Result<sys::hipblasLtMatmulAlgo_t> {
    use std::time::Instant;
    println!(
        "Pre-warming hipBLASLt kernel cache on GPU #{} for {} (tuning up to {TUNE_ALGOS} algos)...",
        gpu.ordinal(),
        precision.name()
    );
    let start = Instant::now();
    gpu.set_current()?;

    let set = build_descriptors(precision, size)?;
    let output_bytes = precision.output_matrix_bytes(size);
    let out = DeviceBuffer::alloc_filled(output_bytes, 0)?;
    let stream = Stream::new()?;

    let mut algos = vec![sys::hipblasLtMatmulHeuristicResult_t::default(); TUNE_ALGOS];
    let mut returned: i32 = 0;
    {
        // FP8 solution scoring floods stderr with latency-table warnings; mute
        // fd 2 just for the heuristic call. See StderrSilencer.
        let _silencer = (precision == Precision::Fp8)
            .then(StderrSilencer::new)
            .flatten();
        unsafe {
            lt_check(
                sys::hipblasLtMatmulAlgoGetHeuristic(
                    set.handle.0,
                    set.desc.0,
                    set.a_layout.0,
                    set.b_layout.0,
                    set.cd_layout.0,
                    set.cd_layout.0,
                    set.pref.0,
                    algos.len() as i32,
                    algos.as_mut_ptr(),
                    &mut returned,
                ),
                "hipblasLtMatmulAlgoGetHeuristic (prewarm)",
            )?;
        }
    }
    if returned < 1 {
        return Err(anyhow::anyhow!(
            "hipBLASLt returned no algorithm for {}",
            precision.name()
        ));
    }

    let mut best_algo = algos[0].algo;
    let mut best_time = f64::INFINITY;
    let flops_per_matmul = 2.0 * (size as f64).powi(3);

    // Discard the very first matmul to absorb hipBLASLt's lazy initialization
    // and any one-shot JIT cost. After that, each algo gets one timed sample.
    let _ = time_matmul(&set, &algos[0].algo, &out, &stream)?;
    for (i, r) in algos.iter().take(returned as usize).enumerate() {
        let t = time_matmul(&set, &r.algo, &out, &stream)?;
        let tflops = flops_per_matmul / t / 1e12;
        println!("  algo {i:>2}: {t:>6.3}s  ({tflops:>7.1} TFLOP/s)");
        if t < best_time {
            best_time = t;
            best_algo = r.algo;
        }
    }
    let best_tflops = flops_per_matmul / best_time / 1e12;
    println!(
        "Best algo: {best_tflops:.1} TFLOP/s ({:.1}s spent tuning)",
        start.elapsed().as_secs_f64()
    );
    Ok(best_algo)
}

// --- Burn task (one OS thread per GPU) ----------------------------------------

fn burn_gpu(
    gpu_idx: usize,
    gpu: Arc<HipContext>,
    config: Config,
    tx: Sender<(usize, usize)>,
    stop: Arc<AtomicBool>,
    warmup_barrier: Arc<Barrier>,
) -> anyhow::Result<()> {
    gpu.set_current()?;
    let precision = config.precision;
    let size = config.size;
    let output_bytes = precision.output_matrix_bytes(size);

    let (free_mem, _) = gpu.mem_get_info()?;
    let budget = (free_mem as f64 * MEM_TO_USE_PCT) as usize;
    let consumed = precision.matrix_bytes(size) * 2 + WORKSPACE_BYTES;
    if consumed + output_bytes * PIPELINE_DEPTH > budget {
        return Err(anyhow::anyhow!(
            "GPU {gpu_idx}: only {} MiB free; need ~{} MiB for {} at {size}x{size} with pipeline depth {PIPELINE_DEPTH}",
            free_mem / (1024 * 1024),
            (consumed + output_bytes * PIPELINE_DEPTH) / (1024 * 1024),
            precision.name(),
        ));
    }

    let set = build_descriptors(precision, size)?;

    // PIPELINE_DEPTH independent streams + matching output buffers. We sync
    // the oldest in-flight matmul and resubmit on its stream while the
    // others continue to run, so the matrix cores stay fed across kernel
    // boundaries.
    let mut streams: Vec<Stream> = Vec::with_capacity(PIPELINE_DEPTH);
    let mut outputs: Vec<DeviceBuffer> = Vec::with_capacity(PIPELINE_DEPTH);
    for _ in 0..PIPELINE_DEPTH {
        streams.push(Stream::new()?);
        outputs.push(DeviceBuffer::alloc_filled(output_bytes, 0)?);
    }

    // Re-query the heuristic on *this* handle and burn with its first pick.
    // The algo bytes can reference per-handle internal state (a kernel module
    // loaded into the GPU context that owned the lookup), so the prewarm's algo
    // isn't always portable; the call is fast once the comgr cache is hot. We
    // intentionally do NOT time-and-pick among candidates here — see TUNE_ALGOS.
    let algo = {
        let mut algos = [sys::hipblasLtMatmulHeuristicResult_t::default(); 1];
        let mut returned: i32 = 0;
        {
            // Mute the FP8 latency-table warning flood for the heuristic call;
            // the global lock in StderrSilencer serializes this across GPUs.
            let _silencer = (precision == Precision::Fp8)
                .then(StderrSilencer::new)
                .flatten();
            unsafe {
                lt_check(
                    sys::hipblasLtMatmulAlgoGetHeuristic(
                        set.handle.0,
                        set.desc.0,
                        set.a_layout.0,
                        set.b_layout.0,
                        set.cd_layout.0,
                        set.cd_layout.0,
                        set.pref.0,
                        1,
                        algos.as_mut_ptr(),
                        &mut returned,
                    ),
                    "hipblasLtMatmulAlgoGetHeuristic (burn)",
                )?;
            }
        }
        if returned < 1 {
            return Err(anyhow::anyhow!(
                "GPU #{gpu_idx}: no algorithm for {}",
                precision.name()
            ));
        }
        let _ = config.tuned_algo; // suppress unused warning
        algos[0].algo
    };
    let alpha_f32: f32 = 1.0;
    let beta_f32: f32 = 0.0;

    let submit = |stream: &Stream, out: &DeviceBuffer| -> sys::hipblasStatus_t {
        unsafe {
            sys::hipblasLtMatmul(
                set.handle.0,
                set.desc.0,
                &alpha_f32 as *const f32 as *const c_void,
                set.a.as_ptr(),
                set.a_layout.0,
                set.b.as_ptr(),
                set.b_layout.0,
                &beta_f32 as *const f32 as *const c_void,
                out.as_ptr(),
                set.cd_layout.0,
                out.as_ptr(),
                set.cd_layout.0,
                &algo,
                set.workspace.as_ptr(),
                WORKSPACE_BYTES,
                stream.0,
            )
        }
    };

    // Quick per-thread warmup so per-stream lazy state is up before the
    // measurement window opens. comgr cache is hot from prewarm so this is
    // fast.
    eprintln!(
        "GPU #{gpu_idx}: warming up hipBLASLt ({})",
        precision.name()
    );
    for i in 0..PIPELINE_DEPTH {
        let rc = submit(&streams[i], &outputs[i]);
        if rc != sys::HIPBLAS_STATUS_SUCCESS {
            return Err(anyhow::anyhow!("GPU #{gpu_idx}: warmup matmul status {rc}"));
        }
    }
    for s in &streams {
        s.sync()?;
    }
    eprintln!("GPU #{gpu_idx}: warmup complete");

    warmup_barrier.wait();

    let mut error_count = 0;
    const MAX_ERRORS: usize = 16;

    // Prime the pipeline so all PIPELINE_DEPTH streams have an in-flight matmul.
    for i in 0..PIPELINE_DEPTH {
        if submit(&streams[i], &outputs[i]) != sys::HIPBLAS_STATUS_SUCCESS {
            error_count += 1;
        }
    }

    let mut next = 0usize;
    while !stop.load(Relaxed) && error_count < MAX_ERRORS {
        if let Err(e) = streams[next].sync() {
            error_count += 1;
            if error_count == 1 {
                eprintln!("GPU #{gpu_idx}: stream sync failed: {e}");
            }
            continue;
        }
        let _ = tx.send((gpu_idx, 1));
        let rc = submit(&streams[next], &outputs[next]);
        if rc != sys::HIPBLAS_STATUS_SUCCESS {
            error_count += 1;
            if error_count == 1 {
                eprintln!(
                    "GPU #{gpu_idx}: hipblasLtMatmul status {rc} (suppressing further errors)"
                );
            }
        }
        next = (next + 1) % PIPELINE_DEPTH;
    }

    for s in &streams {
        let _ = s.sync();
    }
    Ok(())
}

// --- Progress / reporting -----------------------------------------------------

// With the warmup barrier in place, every GPU enters the burn loop within a
// few milliseconds of every other. We still skip 1 reporting tick to discard
// the partial-second sample that straddles the barrier release.
const PER_GPU_WARMUP_TICKS: u32 = 1;

async fn report_progress(
    config: Config,
    gpu_keys: Vec<String>,
    mut rx: Receiver<(usize, usize)>,
    stop: Arc<AtomicBool>,
    gpus_healthy: Arc<AtomicBool>,
) {
    use tokio::sync::mpsc::error::TryRecvError;
    let gpu_count = gpu_keys.len();
    let mut interval = tokio::time::interval(std::time::Duration::from_secs(1));
    let mut burn_results = (0..gpu_count).map(BurnResult::new).collect::<Vec<_>>();
    let mut nonzero_ticks = vec![0u32; gpu_count];
    let mut burning = false;
    let mut burn_seconds_left = config.duration_secs;
    let flops_per_matmul = 2.0 * (config.size as f64).powi(3);
    let rocm_smi = match RocmSmiWrapper::new() {
        Ok(smi) => Some(Arc::new(Mutex::new(smi))),
        Err(e) => {
            eprintln!(
                "Warning: Failed to initialize ROCm SMI: {e}. Temperature monitoring disabled."
            );
            None
        }
    };

    let mut burn_channel_dead = false;
    loop {
        interval.tick().await;
        if stop.load(Relaxed) {
            break;
        }
        let mut nops = vec![0usize; gpu_count];
        loop {
            match rx.try_recv() {
                Ok((idx, n)) => nops[idx] += n,
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    burn_channel_dead = true;
                    break;
                }
            }
        }
        for i in 0..gpu_count {
            if nops[i] > 0 {
                nonzero_ticks[i] += 1;
            }
        }

        let phase = if burning {
            format!(
                "burn {:>3}/{}",
                config.duration_secs - burn_seconds_left + 1,
                config.duration_secs
            )
        } else {
            "warmup".to_string()
        };

        let mut line = format!("[{phase}] ");
        for i in 0..gpu_count {
            let flops = nops[i] as f64 * flops_per_matmul;
            line.push_str(&format_throughput(flops));
            if i < gpu_count - 1 {
                line.push_str(" | ");
            }
            if burning && nonzero_ticks[i] > PER_GPU_WARMUP_TICKS {
                burn_results[i].flops_max = burn_results[i].flops_max.max(flops);
                burn_results[i].flops_min = burn_results[i].flops_min.min(flops);
                burn_results[i].flops_sum += flops;
                burn_results[i].n_iters += 1;
            }
        }

        if let Some(ref smi) = rocm_smi {
            let temps: Vec<Option<u32>> = {
                let mut smi = smi.lock().unwrap();
                (0..gpu_count)
                    .map(|i| smi.get_temperature(i as u32).ok())
                    .collect()
            };
            line.push_str(" || ");
            for i in 0..gpu_count {
                match temps[i] {
                    Some(t) => {
                        line.push_str(&format!("{t}°C"));
                        if burning && nonzero_ticks[i] > PER_GPU_WARMUP_TICKS {
                            burn_results[i].temp_max = burn_results[i].temp_max.max(t);
                            burn_results[i].temp_min = burn_results[i].temp_min.min(t);
                            burn_results[i].temp_sum += t as u64;
                            burn_results[i].temp_count += 1;
                        }
                    }
                    None => line.push_str(" N/A"),
                }
                if i < gpu_count - 1 {
                    line.push(' ');
                }
            }
        }

        println!("{line}");

        if !burning {
            if nonzero_ticks.iter().all(|&n| n > PER_GPU_WARMUP_TICKS) {
                burning = true;
                println!(
                    "Warmup complete. Burning all {gpu_count} GPUs for {}s...",
                    config.duration_secs
                );
            } else if burn_channel_dead {
                eprintln!("All burn threads exited before reaching steady state");
                stop.store(true, Relaxed);
                break;
            }
        } else {
            burn_seconds_left = burn_seconds_left.saturating_sub(1);
            if burn_seconds_left == 0 || burn_channel_dead {
                stop.store(true, Relaxed);
                break;
            }
        }
    }

    println!();
    for (i, r) in burn_results.iter().enumerate() {
        if r.n_iters == 0 {
            println!(
                "GPU #{}: no samples collected (burn never reached steady state)",
                r.gpu_idx
            );
            continue;
        }
        let peak = peak_flops_per_sec(&gpu_keys[i], config.precision);
        let pct = peak.map(|p| 100.0 * r.flops_avg() / p);
        match pct {
            Some(pct) => println!(
                "GPU #{}: avg {} (min {}, max {})  -  {:.1}% of peak {} ({})",
                r.gpu_idx,
                format_throughput(r.flops_avg()),
                format_throughput(r.flops_min),
                format_throughput(r.flops_max),
                pct,
                config.precision.name(),
                format_throughput(peak.unwrap()),
            ),
            None => println!(
                "GPU #{}: avg {} (min {}, max {})",
                r.gpu_idx,
                format_throughput(r.flops_avg()),
                format_throughput(r.flops_min),
                format_throughput(r.flops_max),
            ),
        }
        if r.temp_count > 0 {
            println!(
                "         Temperature: avg {:.1}°C, min {}°C, max {}°C",
                r.temp_avg(),
                r.temp_min,
                r.temp_max
            );
        }
    }

    let (healthy, reasons) = are_gpus_healthy(&burn_results, config.tflops_tolerance);
    if healthy {
        println!("All GPUs seem healthy");
    } else {
        println!("Some GPUs are not healthy. Reasons:");
        for r in reasons {
            println!("  - {r}");
        }
    }
    gpus_healthy.store(healthy, Relaxed);
    println!("Freeing GPUs...");
}

fn format_throughput(flops_per_sec: f64) -> String {
    let v = flops_per_sec;
    if v >= 1e15 {
        format!("{:6.2} PFLOP/s", v / 1e15)
    } else if v >= 1e12 {
        format!("{:6.2} TFLOP/s", v / 1e12)
    } else if v >= 1e9 {
        format!("{:6.2} GFLOP/s", v / 1e9)
    } else if v >= 1e6 {
        format!("{:6.2} MFLOP/s", v / 1e6)
    } else {
        format!("{v:6.0} FLOP/s")
    }
}

fn are_gpus_healthy(burn_results: &[BurnResult], tflops_tolerance: f64) -> (bool, Vec<String>) {
    let best_avg = burn_results
        .iter()
        .map(|r| r.flops_avg())
        .fold(0f64, f64::max);
    let acceptable = best_avg * (100. - tflops_tolerance) / 100.;
    let mut reasons = vec![];
    for r in burn_results.iter() {
        if r.flops_avg() < acceptable {
            reasons.push(format!("GPU {} - {GPU_FLOPS_REASON}", r.gpu_idx));
        }
    }
    (reasons.is_empty(), reasons)
}

fn detect_gpus() -> anyhow::Result<Vec<Arc<HipContext>>> {
    let num_gpus = HipContext::device_count()? as usize;
    let mut devices = Vec::new();
    for i in 0..num_gpus {
        devices.push(Arc::new(HipContext::new(i as i32)?));
    }
    Ok(devices)
}

async fn shutdown_signal(stop: Arc<AtomicBool>) {
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

#[cfg(test)]
mod tests {
    use super::*;

    // Containers report a generic "AMD Radeon Graphics" name; match_key() appends
    // the ISA arch so the peak table still resolves. These guard that fallback.
    #[test]
    fn arch_fallback_resolves_when_name_is_generic() {
        let generic = "amd radeon graphics gfx950";
        assert!(precision_supported(generic, Precision::Fp8));
        assert_eq!(
            peak_flops_per_sec(generic, Precision::Fp8),
            Some(5.0 * 1e15)
        );
        // gfx942 (MI300X/MI325X) supports FP8 too.
        assert!(precision_supported(
            "amd radeon graphics gfx942",
            Precision::Fp8
        ));
    }

    #[test]
    fn marketing_name_takes_priority_over_arch() {
        // A resolved MI210 name must use MI210 numbers, not the first gfx90a spec.
        assert_eq!(
            peak_flops_per_sec("amd instinct mi210 gfx90a", Precision::Bf16),
            Some(0.181 * 1e15)
        );
    }

    #[test]
    fn unknown_device_has_no_peak() {
        assert_eq!(
            peak_flops_per_sec("some future gpu gfx9999", Precision::Fp8),
            None
        );
        assert!(!precision_supported(
            "some future gpu gfx9999",
            Precision::Fp8
        ));
    }
}
