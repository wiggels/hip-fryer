// Minimal FFI bindings to HIP and hipBLASLt for hip-fryer.
//
// We hand-roll this small wrapper instead of pulling in `rocm-rs` because that
// crate's published bindings reference rocBLAS symbols (the experimental f8
// data types and the `rocblas_computetype` enum) that ROCm 7 removed.
#![allow(non_camel_case_types, non_upper_case_globals)]

use std::ffi::c_void;
use std::os::raw::{c_char, c_int, c_uint};

// --- HIP runtime --------------------------------------------------------------

pub type hipError_t = c_int;
pub const hipSuccess: hipError_t = 0;

pub type hipDeviceptr_t = *mut c_void;
pub type hipStream_t = *mut c_void;

#[link(name = "amdhip64")]
unsafe extern "C" {
    pub fn hipInit(flags: c_uint) -> hipError_t;
    pub fn hipGetDeviceCount(count: *mut c_int) -> hipError_t;
    pub fn hipSetDevice(device_id: c_int) -> hipError_t;
    pub fn hipDeviceGetName(name: *mut c_char, len: c_int, device: c_int) -> hipError_t;
    pub fn hipMalloc(ptr: *mut hipDeviceptr_t, size: usize) -> hipError_t;
    pub fn hipFree(ptr: hipDeviceptr_t) -> hipError_t;
    pub fn hipMemset(dest: hipDeviceptr_t, value: c_int, size_bytes: usize) -> hipError_t;
    pub fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> hipError_t;
    pub fn hipStreamSynchronize(stream: hipStream_t) -> hipError_t;
    pub fn hipStreamCreate(stream: *mut hipStream_t) -> hipError_t;
    pub fn hipStreamDestroy(stream: hipStream_t) -> hipError_t;
}

// --- hipBLAS / hipBLASLt ------------------------------------------------------

pub type hipblasStatus_t = c_int;
pub const HIPBLAS_STATUS_SUCCESS: hipblasStatus_t = 0;

pub type hipblasOperation_t = c_int;
pub const HIPBLAS_OP_N: hipblasOperation_t = 111;
pub const HIPBLAS_OP_T: hipblasOperation_t = 112;

// From `hipDataType` in /opt/rocm/include/hip/library_types.h
pub type hipDataType = c_int;
pub const HIP_R_32F: hipDataType = 0;
pub const HIP_R_16BF: hipDataType = 14;
pub const HIP_R_8F_E4M3: hipDataType = 28;

// From `hipblasComputeType_t` in /opt/rocm/include/hipblas-common/hipblas-common.h
pub type hipblasComputeType_t = c_int;
pub const HIPBLAS_COMPUTE_32F: hipblasComputeType_t = 2;

// Opaque handle types
pub type hipblasLtHandle_t = *mut c_void;
pub type hipblasLtMatrixLayout_t = *mut c_void;
pub type hipblasLtMatmulDesc_t = *mut c_void;
pub type hipblasLtMatmulPreference_t = *mut c_void;

// hipblasLtMatmulDescAttributes_t enum
pub type hipblasLtMatmulDescAttributes_t = c_int;
pub const HIPBLASLT_MATMUL_DESC_TRANSA: hipblasLtMatmulDescAttributes_t = 0;
pub const HIPBLASLT_MATMUL_DESC_TRANSB: hipblasLtMatmulDescAttributes_t = 1;

// hipblasLtMatmulPreferenceAttributes_t enum
pub type hipblasLtMatmulPreferenceAttributes_t = c_int;
pub const HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES: hipblasLtMatmulPreferenceAttributes_t = 1;

// hipblasLtMatmulAlgo_t (opaque-ish; layout matches the C struct so the
// heuristic API can hand one back to us).
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct hipblasLtMatmulAlgo_t {
    pub data: [u8; 16],
    pub max_workspace_bytes: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct hipblasLtMatmulHeuristicResult_t {
    pub algo: hipblasLtMatmulAlgo_t,
    pub workspace_size: usize,
    pub state: hipblasStatus_t,
    pub waves_count: f32,
    pub reserved: [c_int; 4],
}

impl Default for hipblasLtMatmulHeuristicResult_t {
    fn default() -> Self {
        Self {
            algo: hipblasLtMatmulAlgo_t::default(),
            workspace_size: 0,
            state: HIPBLAS_STATUS_SUCCESS,
            waves_count: 0.0,
            reserved: [0; 4],
        }
    }
}

#[link(name = "hipblaslt")]
unsafe extern "C" {
    pub fn hipblasLtCreate(handle: *mut hipblasLtHandle_t) -> hipblasStatus_t;
    pub fn hipblasLtDestroy(handle: hipblasLtHandle_t) -> hipblasStatus_t;

    pub fn hipblasLtMatrixLayoutCreate(
        mat_layout: *mut hipblasLtMatrixLayout_t,
        data_type: hipDataType,
        rows: u64,
        cols: u64,
        ld: i64,
    ) -> hipblasStatus_t;
    pub fn hipblasLtMatrixLayoutDestroy(mat_layout: hipblasLtMatrixLayout_t) -> hipblasStatus_t;

    pub fn hipblasLtMatmulDescCreate(
        desc: *mut hipblasLtMatmulDesc_t,
        compute_type: hipblasComputeType_t,
        scale_type: hipDataType,
    ) -> hipblasStatus_t;
    pub fn hipblasLtMatmulDescDestroy(desc: hipblasLtMatmulDesc_t) -> hipblasStatus_t;
    pub fn hipblasLtMatmulDescSetAttribute(
        desc: hipblasLtMatmulDesc_t,
        attr: hipblasLtMatmulDescAttributes_t,
        buf: *const c_void,
        size_in_bytes: usize,
    ) -> hipblasStatus_t;

    pub fn hipblasLtMatmulPreferenceCreate(
        pref: *mut hipblasLtMatmulPreference_t,
    ) -> hipblasStatus_t;
    pub fn hipblasLtMatmulPreferenceDestroy(pref: hipblasLtMatmulPreference_t) -> hipblasStatus_t;
    pub fn hipblasLtMatmulPreferenceSetAttribute(
        pref: hipblasLtMatmulPreference_t,
        attr: hipblasLtMatmulPreferenceAttributes_t,
        buf: *const c_void,
        size_in_bytes: usize,
    ) -> hipblasStatus_t;

    pub fn hipblasLtMatmulAlgoGetHeuristic(
        handle: hipblasLtHandle_t,
        desc: hipblasLtMatmulDesc_t,
        a_desc: hipblasLtMatrixLayout_t,
        b_desc: hipblasLtMatrixLayout_t,
        c_desc: hipblasLtMatrixLayout_t,
        d_desc: hipblasLtMatrixLayout_t,
        pref: hipblasLtMatmulPreference_t,
        requested_algo_count: c_int,
        heuristic_results_array: *mut hipblasLtMatmulHeuristicResult_t,
        return_algo_count: *mut c_int,
    ) -> hipblasStatus_t;

    #[allow(clippy::too_many_arguments)]
    pub fn hipblasLtMatmul(
        handle: hipblasLtHandle_t,
        desc: hipblasLtMatmulDesc_t,
        alpha: *const c_void,
        a: *const c_void,
        a_desc: hipblasLtMatrixLayout_t,
        b: *const c_void,
        b_desc: hipblasLtMatrixLayout_t,
        beta: *const c_void,
        c: *const c_void,
        c_desc: hipblasLtMatrixLayout_t,
        d: *mut c_void,
        d_desc: hipblasLtMatrixLayout_t,
        algo: *const hipblasLtMatmulAlgo_t,
        workspace: *mut c_void,
        workspace_size_in_bytes: usize,
        stream: hipStream_t,
    ) -> hipblasStatus_t;
}
