#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

// Minimal CUDA/NCCL type aliases. The hook intentionally avoids CUDA headers so
// it can still be compiled on machines without a toolkit.
typedef int CUresult;
typedef void* CUfunction;
typedef void* CUmodule;
typedef void* CUstream;
typedef void* CUevent;
typedef void* CUlibrary;
typedef void* CUkernel;
typedef int cudaError_t;
typedef void* cudaStream_t;

typedef int ncclResult_t;
typedef int ncclDataType_t;
typedef int ncclRedOp_t;
typedef void* ncclComm_t;

#define CU_SUCCESS 0
#define CU_EVENT_DEFAULT 0
#define WARP_SIZE 32ULL

typedef struct CUlaunchConfig_st {
    unsigned int gridDimX;
    unsigned int gridDimY;
    unsigned int gridDimZ;
    unsigned int blockDimX;
    unsigned int blockDimY;
    unsigned int blockDimZ;
    unsigned int sharedMemBytes;
    CUstream hStream;
    void* attrs;
    unsigned int numAttrs;
} CUlaunchConfig;

typedef struct cuda_dim3_st {
    unsigned int x;
    unsigned int y;
    unsigned int z;
} cuda_dim3;

typedef struct cudaLaunchConfig_runtime_st {
    cuda_dim3 gridDim;
    cuda_dim3 blockDim;
    size_t dynamicSmemBytes;
    cudaStream_t stream;
    void* attrs;
    unsigned int numAttrs;
} cudaLaunchConfig_runtime;

typedef CUresult (*cuModuleLoadData_t)(CUmodule*, const void*);
typedef CUresult (*cuModuleLoadDataEx_t)(CUmodule*, const void*, unsigned int, void*, void**);
typedef CUresult (*cuModuleLoadFatBinary_t)(CUmodule*, const void*);
typedef CUresult (*cuModuleGetFunction_t)(CUfunction*, CUmodule, const char*);
typedef CUresult (*cuKernelGetFunction_t)(CUfunction*, CUkernel);
typedef CUresult (*cuLibraryLoadData_t)(CUlibrary*, const void*, void*, void**, unsigned int, void*, void**, unsigned int);
typedef CUresult (*cuLibraryGetKernel_t)(CUkernel*, CUlibrary, const char*);
typedef CUresult (*cuLibraryGetModule_t)(CUmodule*, CUlibrary);
typedef CUresult (*cuFuncGetName_t)(const char**, CUfunction);
typedef CUresult (*cuLaunchKernel_t)(CUfunction, unsigned int, unsigned int,
                                     unsigned int, unsigned int, unsigned int,
                                     unsigned int, unsigned int, CUstream,
                                     void**, void**);
typedef CUresult (*cuLaunchKernelEx_t)(const CUlaunchConfig*, CUfunction, void**, void**);
typedef CUresult (*cuEventCreate_t)(CUevent*, unsigned int);
typedef CUresult (*cuEventRecord_t)(CUevent, CUstream);
typedef CUresult (*cuEventQuery_t)(CUevent);
typedef CUresult (*cuEventSynchronize_t)(CUevent);
typedef CUresult (*cuEventElapsedTime_t)(float*, CUevent, CUevent);
typedef CUresult (*cuEventDestroy_t)(CUevent);
typedef CUresult (*cuGetProcAddress_t)(const char*, void**, int, uint64_t, void*);
typedef cudaError_t (*cudaLaunchKernel_runtime_t)(const void*, cuda_dim3, cuda_dim3,
                                                  void**, size_t, cudaStream_t);
typedef cudaError_t (*cudaLaunchKernelExC_t)(const cudaLaunchConfig_runtime*, const void*, void**);
typedef cudaError_t (*cudaLaunchCooperativeKernel_t)(const void*, cuda_dim3, cuda_dim3,
                                                     void**, size_t, cudaStream_t);

typedef ncclResult_t (*ncclAllReduce_t)(const void*, void*, size_t,
                                        ncclDataType_t, ncclRedOp_t,
                                        ncclComm_t, CUstream);
typedef ncclResult_t (*ncclAllGather_t)(const void*, void*, size_t,
                                        ncclDataType_t, ncclComm_t, CUstream);
typedef ncclResult_t (*ncclReduceScatter_t)(const void*, void*, size_t,
                                            ncclDataType_t, ncclRedOp_t,
                                            ncclComm_t, CUstream);
typedef ncclResult_t (*ncclBroadcast_t)(const void*, void*, size_t,
                                        ncclDataType_t, int, ncclComm_t,
                                        CUstream);
typedef void* (*dlsym_fn_t)(void*, const char*);

static cuModuleLoadData_t real_cuModuleLoadData = NULL;
static cuModuleLoadDataEx_t real_cuModuleLoadDataEx = NULL;
static cuModuleLoadFatBinary_t real_cuModuleLoadFatBinary = NULL;
static cuModuleGetFunction_t real_cuModuleGetFunction = NULL;
static cuKernelGetFunction_t real_cuKernelGetFunction = NULL;
static cuLibraryLoadData_t real_cuLibraryLoadData = NULL;
static cuLibraryGetKernel_t real_cuLibraryGetKernel = NULL;
static cuLibraryGetModule_t real_cuLibraryGetModule = NULL;
static cuFuncGetName_t real_cuFuncGetName = NULL;
static cuLaunchKernel_t real_cuLaunchKernel = NULL;
static cuLaunchKernel_t real_cuLaunchKernel_ptsz = NULL;
static cuLaunchKernelEx_t real_cuLaunchKernelEx = NULL;
static cuLaunchKernelEx_t real_cuLaunchKernelEx_ptsz = NULL;
static cuEventCreate_t real_cuEventCreate = NULL;
static cuEventRecord_t real_cuEventRecord = NULL;
static cuEventQuery_t real_cuEventQuery = NULL;
static cuEventSynchronize_t real_cuEventSynchronize = NULL;
static cuEventElapsedTime_t real_cuEventElapsedTime = NULL;
static cuEventDestroy_t real_cuEventDestroy = NULL;
static cuGetProcAddress_t real_cuGetProcAddress = NULL;
static cuGetProcAddress_t real_cuGetProcAddress_v2 = NULL;
static cudaLaunchKernel_runtime_t real_cudaLaunchKernel = NULL;
static cudaLaunchKernel_runtime_t real_cudaLaunchKernel_ptsz = NULL;
static cudaLaunchKernelExC_t real_cudaLaunchKernelExC = NULL;
static cudaLaunchKernelExC_t real_cudaLaunchKernelExC_ptsz = NULL;
static cudaLaunchCooperativeKernel_t real_cudaLaunchCooperativeKernel = NULL;
static cudaLaunchCooperativeKernel_t real_cudaLaunchCooperativeKernel_ptsz = NULL;

static ncclAllReduce_t real_ncclAllReduce = NULL;
static ncclAllGather_t real_ncclAllGather = NULL;
static ncclReduceScatter_t real_ncclReduceScatter = NULL;
static ncclBroadcast_t real_ncclBroadcast = NULL;

static int g_sock = -1;
static struct sockaddr_un g_addr;
static pthread_once_t g_init_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t g_map_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_queue_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_queue_cond = PTHREAD_COND_INITIALIZER;
static pthread_t g_poller;
static int g_poller_started = 0;
static int g_should_run = 1;
static int g_cuda_event_enabled = 1;
static int g_sync_stream_anchor = 1;
static int g_cuda_event_pool_enabled = 1;
static int g_trace_getproc_errors = 0;
static int g_patch_dlsym = 0;
static int g_patch_getproc = 1;
static int g_disasm_enabled = 0;
static int g_emit_code_events = 0;
static int g_async_launch_emit = 1;
static unsigned long long g_launch_seq = 0;

struct code_item {
    void* key;
    void* code;
    size_t size;
    char* name;
    int owns_code;
    int disassembly_started;
    struct code_item* next;
};

struct stream_anchor {
    CUstream stream;
    CUevent event;
    long long host_ns;
    int valid;
    struct stream_anchor* next;
};

struct launch_item {
    unsigned long long launch_id;
    int ret;
    char* kernel_name;
    CUfunction func;
    CUstream stream;
    CUevent start_event;
    CUevent end_event;
    struct event_pair* event_pair;
    struct stream_anchor* anchor;
    unsigned int grid[3];
    unsigned int block[3];
    unsigned int shared_mem;
    unsigned long long warps_per_block;
    unsigned long long total_warps;
    long long cpu_enqueue_start_ns;
    long long cpu_enqueue_end_ns;
    struct launch_item* next;
};

struct event_pair {
    CUevent start_event;
    CUevent end_event;
    struct event_pair* next;
};

struct launch_emit_record {
    long long ts_ns;
    int pid;
    unsigned long long tid;
    char kernel_name[256];
    char alignment[64];
    long long gpu_start_ns;
    long long gpu_end_ns;
    long long gpu_duration_ns;
    long long cpu_enqueue_start_ns;
    long long cpu_enqueue_end_ns;
    long long poll_ready_ns;
    void* func;
    void* stream;
    unsigned long long launch_id;
    int ret;
    unsigned int grid[3];
    unsigned int block[3];
    unsigned int shared_mem;
    unsigned long long warps_per_block;
    unsigned long long total_warps;
    int cuda_event_timing;
};

static struct code_item* g_code_map = NULL;
static struct stream_anchor* g_anchors = NULL;
static struct launch_item* g_pending = NULL;
static struct event_pair* g_event_pool = NULL;
static struct launch_emit_record* g_launch_emit_queue = NULL;
static size_t g_launch_emit_cap = 0;
static size_t g_launch_emit_head = 0;
static size_t g_launch_emit_tail = 0;
static size_t g_launch_emit_count = 0;
static pthread_mutex_t g_launch_emit_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_launch_emit_cond = PTHREAD_COND_INITIALIZER;
static pthread_t g_launch_emit_thread;
static int g_launch_emit_started = 0;
static int g_launch_emit_stop = 0;

static int cuda_events_ready(void);

static long long now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (long long)ts.tv_sec * 1000000000LL + (long long)ts.tv_nsec;
}

static int env_flag(const char* name, int default_value) {
    const char* value = getenv(name);
    if (!value || !value[0]) {
        return default_value;
    }
    return !(strcmp(value, "0") == 0 || strcasecmp(value, "false") == 0 ||
             strcasecmp(value, "off") == 0 || strcasecmp(value, "no") == 0);
}

static unsigned long env_ulong(const char* name, unsigned long default_value) {
    const char* value = getenv(name);
    if (!value || !value[0]) {
        return default_value;
    }
    char* end = NULL;
    errno = 0;
    unsigned long parsed = strtoul(value, &end, 10);
    if (errno != 0 || end == value || parsed == 0) {
        return default_value;
    }
    return parsed;
}

static dlsym_fn_t real_dlsym_func = NULL;
static void* g_libcuda_handle = NULL;
static void* g_libcudart_handle = NULL;
static void* g_libnccl_handle = NULL;
static int g_getproc_symbols_refreshed = 0;
static int g_driver_symbols_refreshed = 0;
static int g_runtime_symbols_refreshed = 0;
static int g_nccl_symbols_refreshed = 0;

static dlsym_fn_t get_real_dlsym(void) {
#ifdef __linux__
    if (!real_dlsym_func) {
        real_dlsym_func = (dlsym_fn_t)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
        if (!real_dlsym_func) {
            real_dlsym_func = (dlsym_fn_t)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.34");
        }
    }
#else
    if (!real_dlsym_func) {
        real_dlsym_func = dlsym;
    }
#endif
    return real_dlsym_func;
}

// Forward declarations for cuGetProcAddress wrappers used in symbol refresh.
CUresult cuGetProcAddress(const char* symbol, void** pfn, int cudaVersion, uint64_t flags, void* symbolStatus);
CUresult cuGetProcAddress_v2(const char* symbol, void** pfn, int cudaVersion, uint64_t flags, void* symbolStatus);

static void ensure_libcuda_loaded(void) {
    if (g_libcuda_handle) {
        return;
    }
    g_libcuda_handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!g_libcuda_handle) {
        g_libcuda_handle = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
    }
}

static void* dlsym_libcuda(const char* symbol) {
    if (!symbol) {
        return NULL;
    }
    ensure_libcuda_loaded();
    if (!g_libcuda_handle) {
        return NULL;
    }
    dlsym_fn_t real_dlsym = get_real_dlsym();
    return real_dlsym ? real_dlsym(g_libcuda_handle, symbol) : NULL;
}

static void ensure_libcudart_loaded(void) {
    if (g_libcudart_handle) {
        return;
    }
    g_libcudart_handle = dlopen("libcudart.so.12", RTLD_NOW | RTLD_GLOBAL);
    if (!g_libcudart_handle) {
        g_libcudart_handle = dlopen("libcudart.so.11.0", RTLD_NOW | RTLD_GLOBAL);
    }
    if (!g_libcudart_handle) {
        g_libcudart_handle = dlopen("libcudart.so", RTLD_NOW | RTLD_GLOBAL);
    }
}

static void* dlsym_libcudart(const char* symbol) {
    if (!symbol) {
        return NULL;
    }
    ensure_libcudart_loaded();
    if (!g_libcudart_handle) {
        return NULL;
    }
    dlsym_fn_t real_dlsym = get_real_dlsym();
    return real_dlsym ? real_dlsym(g_libcudart_handle, symbol) : NULL;
}

static void ensure_libnccl_loaded(void) {
    if (g_libnccl_handle) {
        return;
    }
    g_libnccl_handle = dlopen("libnccl.so.2", RTLD_NOW | RTLD_GLOBAL);
    if (!g_libnccl_handle) {
        g_libnccl_handle = dlopen("libnccl.so", RTLD_NOW | RTLD_GLOBAL);
    }
}

static void* dlsym_libnccl(const char* symbol) {
    if (!symbol) {
        return NULL;
    }
    ensure_libnccl_loaded();
    if (!g_libnccl_handle) {
        return NULL;
    }
    dlsym_fn_t real_dlsym = get_real_dlsym();
    return real_dlsym ? real_dlsym(g_libnccl_handle, symbol) : NULL;
}

static void* dlsym_next_any(const char* name, const char* alt_name) {
    dlsym_fn_t real_dlsym = get_real_dlsym();
    void* fn = real_dlsym ? real_dlsym(RTLD_NEXT, name) : NULL;
    if (!fn && alt_name) {
        fn = real_dlsym ? real_dlsym(RTLD_NEXT, alt_name) : NULL;
    }
    return fn;
}

static int contains_text(const char* text, const char* needle) {
    return text && needle && strstr(text, needle) != NULL;
}

static int __attribute__((unused)) is_interposable_cuda_symbol(const char* symbol) {
    if (!symbol) {
        return 0;
    }
    return strcmp(symbol, "cuModuleLoadData") == 0 ||
           strcmp(symbol, "cuModuleLoadDataEx") == 0 ||
           strcmp(symbol, "cuModuleLoadFatBinary") == 0 ||
           strcmp(symbol, "cuModuleGetFunction") == 0 ||
           strcmp(symbol, "cuKernelGetFunction") == 0 ||
           strcmp(symbol, "cuLibraryLoadData") == 0 ||
           strcmp(symbol, "cuLibraryGetKernel") == 0 ||
           strcmp(symbol, "cuLibraryGetModule") == 0 ||
           strcmp(symbol, "cuFuncGetName") == 0 ||
           strcmp(symbol, "cuLaunchKernel") == 0 ||
           strcmp(symbol, "cuLaunchKernel_ptsz") == 0 ||
           strcmp(symbol, "cuLaunchKernelEx") == 0 ||
           strcmp(symbol, "cuLaunchKernelEx_ptsz") == 0 ||
           strcmp(symbol, "cudaLaunchKernel") == 0 ||
           strcmp(symbol, "cudaLaunchKernel_ptsz") == 0 ||
           strcmp(symbol, "cudaLaunchCooperativeKernel") == 0 ||
           strcmp(symbol, "cudaLaunchCooperativeKernel_ptsz") == 0 ||
           strcmp(symbol, "cudaLaunchKernelExC") == 0 ||
           strcmp(symbol, "cudaLaunchKernelExC_ptsz") == 0 ||
           strcmp(symbol, "cuGetProcAddress") == 0 ||
           strcmp(symbol, "cuGetProcAddress_v2") == 0;
}

static int is_blocked_interposer_caller(void* caller) {
    Dl_info info;
    memset(&info, 0, sizeof(info));
    if (!caller || !dladdr(caller, &info) || !info.dli_fname) {
        return 0;
    }
    const char* path = info.dli_fname;
    return contains_text(path, "/ucx/") ||
           contains_text(path, "/hpcx/") ||
           contains_text(path, "libucs") ||
           contains_text(path, "libucp") ||
           contains_text(path, "libuct") ||
           contains_text(path, "libucm") ||
           contains_text(path, "libmpi") ||
           contains_text(path, "libopen-pal") ||
           contains_text(path, "libopen-rte") ||
           contains_text(path, "libhcoll") ||
           contains_text(path, "libnccl");
}

static void refresh_getproc_symbols(void) {
    if (g_getproc_symbols_refreshed) {
        return;
    }
    if (!real_cuGetProcAddress) {
        real_cuGetProcAddress = (cuGetProcAddress_t)dlsym_next_any("cuGetProcAddress", NULL);
        if (real_cuGetProcAddress == cuGetProcAddress) {
            real_cuGetProcAddress = (cuGetProcAddress_t)dlsym_libcuda("cuGetProcAddress");
        }
    }
    if (!real_cuGetProcAddress_v2) {
        real_cuGetProcAddress_v2 = (cuGetProcAddress_t)dlsym_next_any("cuGetProcAddress_v2", NULL);
        if (real_cuGetProcAddress_v2 == cuGetProcAddress_v2) {
            real_cuGetProcAddress_v2 = (cuGetProcAddress_t)dlsym_libcuda("cuGetProcAddress_v2");
        }
    }
    if (real_cuGetProcAddress || real_cuGetProcAddress_v2) {
        g_getproc_symbols_refreshed = 1;
    }
}

static void refresh_driver_symbols(void) {
    if (g_driver_symbols_refreshed) {
        return;
    }
    if (!real_cuModuleLoadData) {
        real_cuModuleLoadData = (cuModuleLoadData_t)dlsym_next_any("cuModuleLoadData", NULL);
        if (!real_cuModuleLoadData) {
            real_cuModuleLoadData = (cuModuleLoadData_t)dlsym_libcuda("cuModuleLoadData");
        }
    }
    if (!real_cuModuleLoadDataEx) {
        real_cuModuleLoadDataEx = (cuModuleLoadDataEx_t)dlsym_next_any("cuModuleLoadDataEx", NULL);
        if (!real_cuModuleLoadDataEx) {
            real_cuModuleLoadDataEx = (cuModuleLoadDataEx_t)dlsym_libcuda("cuModuleLoadDataEx");
        }
    }
    if (!real_cuModuleLoadFatBinary) {
        real_cuModuleLoadFatBinary = (cuModuleLoadFatBinary_t)dlsym_next_any("cuModuleLoadFatBinary", NULL);
        if (!real_cuModuleLoadFatBinary) {
            real_cuModuleLoadFatBinary = (cuModuleLoadFatBinary_t)dlsym_libcuda("cuModuleLoadFatBinary");
        }
    }
    if (!real_cuModuleGetFunction) {
        real_cuModuleGetFunction = (cuModuleGetFunction_t)dlsym_next_any("cuModuleGetFunction", NULL);
        if (!real_cuModuleGetFunction) {
            real_cuModuleGetFunction = (cuModuleGetFunction_t)dlsym_libcuda("cuModuleGetFunction");
        }
    }
    if (!real_cuKernelGetFunction) {
        real_cuKernelGetFunction = (cuKernelGetFunction_t)dlsym_next_any("cuKernelGetFunction", NULL);
        if (!real_cuKernelGetFunction) {
            real_cuKernelGetFunction = (cuKernelGetFunction_t)dlsym_libcuda("cuKernelGetFunction");
        }
    }
    if (!real_cuLibraryLoadData) {
        real_cuLibraryLoadData = (cuLibraryLoadData_t)dlsym_next_any("cuLibraryLoadData", NULL);
        if (!real_cuLibraryLoadData) {
            real_cuLibraryLoadData = (cuLibraryLoadData_t)dlsym_libcuda("cuLibraryLoadData");
        }
    }
    if (!real_cuLibraryGetKernel) {
        real_cuLibraryGetKernel = (cuLibraryGetKernel_t)dlsym_next_any("cuLibraryGetKernel", NULL);
        if (!real_cuLibraryGetKernel) {
            real_cuLibraryGetKernel = (cuLibraryGetKernel_t)dlsym_libcuda("cuLibraryGetKernel");
        }
    }
    if (!real_cuLibraryGetModule) {
        real_cuLibraryGetModule = (cuLibraryGetModule_t)dlsym_next_any("cuLibraryGetModule", NULL);
        if (!real_cuLibraryGetModule) {
            real_cuLibraryGetModule = (cuLibraryGetModule_t)dlsym_libcuda("cuLibraryGetModule");
        }
    }
    if (!real_cuFuncGetName) {
        real_cuFuncGetName = (cuFuncGetName_t)dlsym_next_any("cuFuncGetName", NULL);
        if (!real_cuFuncGetName) {
            real_cuFuncGetName = (cuFuncGetName_t)dlsym_libcuda("cuFuncGetName");
        }
    }
    if (!real_cuLaunchKernel) {
        real_cuLaunchKernel = (cuLaunchKernel_t)dlsym_next_any("cuLaunchKernel", NULL);
        if (!real_cuLaunchKernel) {
            real_cuLaunchKernel = (cuLaunchKernel_t)dlsym_libcuda("cuLaunchKernel");
        }
    }
    if (!real_cuLaunchKernel_ptsz) {
        real_cuLaunchKernel_ptsz = (cuLaunchKernel_t)dlsym_next_any("cuLaunchKernel_ptsz", NULL);
        if (!real_cuLaunchKernel_ptsz) {
            real_cuLaunchKernel_ptsz = (cuLaunchKernel_t)dlsym_libcuda("cuLaunchKernel_ptsz");
        }
    }
    if (!real_cuLaunchKernelEx) {
        real_cuLaunchKernelEx = (cuLaunchKernelEx_t)dlsym_next_any("cuLaunchKernelEx", NULL);
        if (!real_cuLaunchKernelEx) {
            real_cuLaunchKernelEx = (cuLaunchKernelEx_t)dlsym_libcuda("cuLaunchKernelEx");
        }
    }
    if (!real_cuLaunchKernelEx_ptsz) {
        real_cuLaunchKernelEx_ptsz = (cuLaunchKernelEx_t)dlsym_next_any("cuLaunchKernelEx_ptsz", NULL);
        if (!real_cuLaunchKernelEx_ptsz) {
            real_cuLaunchKernelEx_ptsz = (cuLaunchKernelEx_t)dlsym_libcuda("cuLaunchKernelEx_ptsz");
        }
    }
    if (!real_cuEventCreate) {
        real_cuEventCreate = (cuEventCreate_t)dlsym_next_any("cuEventCreate", NULL);
        if (!real_cuEventCreate) {
            real_cuEventCreate = (cuEventCreate_t)dlsym_libcuda("cuEventCreate");
        }
    }
    if (!real_cuEventRecord) {
        real_cuEventRecord = (cuEventRecord_t)dlsym_next_any("cuEventRecord", NULL);
        if (!real_cuEventRecord) {
            real_cuEventRecord = (cuEventRecord_t)dlsym_libcuda("cuEventRecord");
        }
    }
    if (!real_cuEventQuery) {
        real_cuEventQuery = (cuEventQuery_t)dlsym_next_any("cuEventQuery", NULL);
        if (!real_cuEventQuery) {
            real_cuEventQuery = (cuEventQuery_t)dlsym_libcuda("cuEventQuery");
        }
    }
    if (!real_cuEventSynchronize) {
        real_cuEventSynchronize = (cuEventSynchronize_t)dlsym_next_any("cuEventSynchronize", NULL);
        if (!real_cuEventSynchronize) {
            real_cuEventSynchronize = (cuEventSynchronize_t)dlsym_libcuda("cuEventSynchronize");
        }
    }
    if (!real_cuEventElapsedTime) {
        real_cuEventElapsedTime = (cuEventElapsedTime_t)dlsym_next_any("cuEventElapsedTime", NULL);
        if (!real_cuEventElapsedTime) {
            real_cuEventElapsedTime = (cuEventElapsedTime_t)dlsym_libcuda("cuEventElapsedTime");
        }
    }
    if (!real_cuEventDestroy) {
        real_cuEventDestroy = (cuEventDestroy_t)dlsym_next_any("cuEventDestroy_v2", "cuEventDestroy");
        if (!real_cuEventDestroy) {
            real_cuEventDestroy = (cuEventDestroy_t)dlsym_libcuda("cuEventDestroy_v2");
        }
        if (!real_cuEventDestroy) {
            real_cuEventDestroy = (cuEventDestroy_t)dlsym_libcuda("cuEventDestroy");
        }
    }
    refresh_getproc_symbols();
    if (real_cuLaunchKernel || real_cuLaunchKernel_ptsz ||
        real_cuLaunchKernelEx || real_cuLaunchKernelEx_ptsz) {
        g_driver_symbols_refreshed = 1;
    }
}

static void refresh_runtime_symbols(void) {
    if (g_runtime_symbols_refreshed) {
        return;
    }
    if (!real_cudaLaunchKernel) {
        real_cudaLaunchKernel = (cudaLaunchKernel_runtime_t)dlsym_next_any("cudaLaunchKernel", NULL);
        if (!real_cudaLaunchKernel) {
            real_cudaLaunchKernel = (cudaLaunchKernel_runtime_t)dlsym_libcudart("cudaLaunchKernel");
        }
    }
    if (!real_cudaLaunchKernel_ptsz) {
        real_cudaLaunchKernel_ptsz = (cudaLaunchKernel_runtime_t)dlsym_next_any("cudaLaunchKernel_ptsz", NULL);
        if (!real_cudaLaunchKernel_ptsz) {
            real_cudaLaunchKernel_ptsz = (cudaLaunchKernel_runtime_t)dlsym_libcudart("cudaLaunchKernel_ptsz");
        }
    }
    if (!real_cudaLaunchKernelExC) {
        real_cudaLaunchKernelExC = (cudaLaunchKernelExC_t)dlsym_next_any("cudaLaunchKernelExC", NULL);
        if (!real_cudaLaunchKernelExC) {
            real_cudaLaunchKernelExC = (cudaLaunchKernelExC_t)dlsym_libcudart("cudaLaunchKernelExC");
        }
    }
    if (!real_cudaLaunchKernelExC_ptsz) {
        real_cudaLaunchKernelExC_ptsz = (cudaLaunchKernelExC_t)dlsym_next_any("cudaLaunchKernelExC_ptsz", NULL);
        if (!real_cudaLaunchKernelExC_ptsz) {
            real_cudaLaunchKernelExC_ptsz = (cudaLaunchKernelExC_t)dlsym_libcudart("cudaLaunchKernelExC_ptsz");
        }
    }
    if (!real_cudaLaunchCooperativeKernel) {
        real_cudaLaunchCooperativeKernel =
            (cudaLaunchCooperativeKernel_t)dlsym_next_any("cudaLaunchCooperativeKernel", NULL);
        if (!real_cudaLaunchCooperativeKernel) {
            real_cudaLaunchCooperativeKernel =
                (cudaLaunchCooperativeKernel_t)dlsym_libcudart("cudaLaunchCooperativeKernel");
        }
    }
    if (!real_cudaLaunchCooperativeKernel_ptsz) {
        real_cudaLaunchCooperativeKernel_ptsz =
            (cudaLaunchCooperativeKernel_t)dlsym_next_any("cudaLaunchCooperativeKernel_ptsz", NULL);
        if (!real_cudaLaunchCooperativeKernel_ptsz) {
            real_cudaLaunchCooperativeKernel_ptsz =
                (cudaLaunchCooperativeKernel_t)dlsym_libcudart("cudaLaunchCooperativeKernel_ptsz");
        }
    }
    if (real_cudaLaunchKernel || real_cudaLaunchKernel_ptsz ||
        real_cudaLaunchKernelExC || real_cudaLaunchKernelExC_ptsz ||
        real_cudaLaunchCooperativeKernel || real_cudaLaunchCooperativeKernel_ptsz) {
        g_runtime_symbols_refreshed = 1;
    }
}

static void refresh_nccl_symbols(void) {
    if (g_nccl_symbols_refreshed) {
        return;
    }
    if (!real_ncclAllReduce) {
        real_ncclAllReduce = (ncclAllReduce_t)dlsym_next_any("ncclAllReduce", NULL);
        if (!real_ncclAllReduce) {
            real_ncclAllReduce = (ncclAllReduce_t)dlsym_libnccl("ncclAllReduce");
        }
    }
    if (!real_ncclAllGather) {
        real_ncclAllGather = (ncclAllGather_t)dlsym_next_any("ncclAllGather", NULL);
        if (!real_ncclAllGather) {
            real_ncclAllGather = (ncclAllGather_t)dlsym_libnccl("ncclAllGather");
        }
    }
    if (!real_ncclReduceScatter) {
        real_ncclReduceScatter = (ncclReduceScatter_t)dlsym_next_any("ncclReduceScatter", NULL);
        if (!real_ncclReduceScatter) {
            real_ncclReduceScatter = (ncclReduceScatter_t)dlsym_libnccl("ncclReduceScatter");
        }
    }
    if (!real_ncclBroadcast) {
        real_ncclBroadcast = (ncclBroadcast_t)dlsym_next_any("ncclBroadcast", NULL);
        if (!real_ncclBroadcast) {
            real_ncclBroadcast = (ncclBroadcast_t)dlsym_libnccl("ncclBroadcast");
        }
    }
    if (real_ncclAllReduce || real_ncclAllGather ||
        real_ncclReduceScatter || real_ncclBroadcast) {
        g_nccl_symbols_refreshed = 1;
    }
}

static void json_escape(const char* in, char* out, size_t out_size) {
    if (out_size == 0) {
        return;
    }
    size_t j = 0;
    if (!in) {
        out[0] = '\0';
        return;
    }
    for (size_t i = 0; in[i] != '\0' && j + 2 < out_size; i++) {
        unsigned char c = (unsigned char)in[i];
        if (c == '"' || c == '\\') {
            if (j + 2 >= out_size) {
                break;
            }
            out[j++] = '\\';
            out[j++] = (char)c;
        } else if (c >= 0x20 && c < 0x7f) {
            out[j++] = (char)c;
        } else {
            if (j + 6 >= out_size) {
                break;
            }
            snprintf(out + j, out_size - j, "\\u%04x", c);
            j += 6;
        }
    }
    out[j] = '\0';
}

static void send_raw(const char* raw, size_t len) {
    if (g_sock < 0 || !raw || len == 0) {
        return;
    }
    sendto(g_sock, raw, len, 0, (struct sockaddr*)&g_addr, sizeof(g_addr));
}

static void send_json(const char* fmt, ...) {
    if (g_sock < 0) {
        return;
    }
    char buf[4096];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (n <= 0) {
        return;
    }
    if (n >= (int)sizeof(buf)) {
        n = (int)sizeof(buf) - 1;
    }
    send_raw(buf, (size_t)n);
}

static void format_send_launch_record(const struct launch_emit_record* rec) {
    if (!rec) {
        return;
    }
    char escaped[512];
    json_escape(rec->kernel_name, escaped, sizeof(escaped));
    char buf[4096];
    unsigned long long threads_per_block =
        (unsigned long long)rec->block[0] * rec->block[1] * rec->block[2];
    unsigned long long blocks_total =
        (unsigned long long)rec->grid[0] * rec->grid[1] * rec->grid[2];
    int n = snprintf(
        buf, sizeof(buf),
        "{\"ts_ns\":%lld,\"pid\":%d,\"tid\":%llu,\"layer\":\"cuda\",\"kind\":\"kernel_launch\",\"source\":\"weaver_hook\",\"kernel_name\":\"%s\",\"gpu_start_ns\":%lld,\"gpu_end_ns\":%lld,\"dur_ns\":%lld,\"cpu_enqueue_start_ns\":%lld,\"cpu_enqueue_end_ns\":%lld,\"stream\":\"%p\",\"payload\":{\"launch_id\":%llu,\"ret\":%d,\"kernel\":\"%s\",\"kernel_name\":\"%s\",\"func\":\"%p\",\"stream\":\"%p\",\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],\"shared_mem\":%u,\"shared_memory\":%u,\"threads_per_block\":%llu,\"blocks_total\":%llu,\"warps_per_block\":%llu,\"total_warps\":%llu,\"warp_size\":32,\"warp_scope\":\"block_runtime\",\"gpu_duration_ns\":%lld,\"gpu_start_ns\":%lld,\"gpu_end_ns\":%lld,\"cuda_event_timing\":%s,\"time_alignment\":\"%s\",\"poll_ready_ns\":%lld,\"cpu_enqueue_start_ns\":%lld,\"cpu_enqueue_end_ns\":%lld}}",
        rec->gpu_start_ns, rec->pid, rec->tid, escaped,
        rec->gpu_start_ns, rec->gpu_end_ns, rec->gpu_duration_ns,
        rec->cpu_enqueue_start_ns, rec->cpu_enqueue_end_ns, rec->stream,
        rec->launch_id, rec->ret, escaped, escaped, rec->func, rec->stream,
        rec->grid[0], rec->grid[1], rec->grid[2],
        rec->block[0], rec->block[1], rec->block[2],
        rec->shared_mem, rec->shared_mem,
        threads_per_block, blocks_total,
        rec->warps_per_block, rec->total_warps,
        rec->gpu_duration_ns, rec->gpu_start_ns, rec->gpu_end_ns,
        rec->cuda_event_timing ? "true" : "false", rec->alignment,
        rec->poll_ready_ns, rec->cpu_enqueue_start_ns, rec->cpu_enqueue_end_ns);
    if (n <= 0) {
        return;
    }
    if (n >= (int)sizeof(buf)) {
        n = (int)sizeof(buf) - 1;
    }
    send_raw(buf, (size_t)n);
}

static void* launch_emit_loop(void* unused) {
    (void)unused;
    while (1) {
        pthread_mutex_lock(&g_launch_emit_lock);
        while (g_launch_emit_count == 0 && !g_launch_emit_stop) {
            pthread_cond_wait(&g_launch_emit_cond, &g_launch_emit_lock);
        }
        if (g_launch_emit_count == 0 && g_launch_emit_stop) {
            pthread_mutex_unlock(&g_launch_emit_lock);
            break;
        }
        struct launch_emit_record rec = g_launch_emit_queue[g_launch_emit_head];
        g_launch_emit_head = (g_launch_emit_head + 1U) % g_launch_emit_cap;
        g_launch_emit_count--;
        pthread_mutex_unlock(&g_launch_emit_lock);
        format_send_launch_record(&rec);
    }
    return NULL;
}

static void start_launch_emit_thread(void) {
    if (!g_async_launch_emit || g_launch_emit_started || g_sock < 0) {
        return;
    }
    unsigned long cap = env_ulong("WEAVER_LAUNCH_EMIT_QUEUE_SIZE", 8192UL);
    if (cap < 256UL) {
        cap = 256UL;
    }
    g_launch_emit_queue = (struct launch_emit_record*)calloc(cap, sizeof(*g_launch_emit_queue));
    if (!g_launch_emit_queue) {
        return;
    }
    g_launch_emit_cap = (size_t)cap;
    g_launch_emit_head = 0;
    g_launch_emit_tail = 0;
    g_launch_emit_count = 0;
    g_launch_emit_stop = 0;
    if (pthread_create(&g_launch_emit_thread, NULL, launch_emit_loop, NULL) == 0) {
        g_launch_emit_started = 1;
    }
}

static int enqueue_launch_record(const struct launch_emit_record* rec) {
    if (!rec || !g_async_launch_emit || !g_launch_emit_started || !g_launch_emit_queue) {
        return 0;
    }
    pthread_mutex_lock(&g_launch_emit_lock);
    if (g_launch_emit_count == g_launch_emit_cap) {
        pthread_mutex_unlock(&g_launch_emit_lock);
        return 0;
    }
    g_launch_emit_queue[g_launch_emit_tail] = *rec;
    g_launch_emit_tail = (g_launch_emit_tail + 1U) % g_launch_emit_cap;
    g_launch_emit_count++;
    pthread_cond_signal(&g_launch_emit_cond);
    pthread_mutex_unlock(&g_launch_emit_lock);
    return 1;
}

static void stop_launch_emit_thread(void) {
    if (!g_launch_emit_started) {
        free(g_launch_emit_queue);
        g_launch_emit_queue = NULL;
        g_launch_emit_cap = 0;
        return;
    }
    pthread_mutex_lock(&g_launch_emit_lock);
    g_launch_emit_stop = 1;
    pthread_cond_broadcast(&g_launch_emit_cond);
    pthread_mutex_unlock(&g_launch_emit_lock);
    pthread_join(g_launch_emit_thread, NULL);
    g_launch_emit_started = 0;
    free(g_launch_emit_queue);
    g_launch_emit_queue = NULL;
    g_launch_emit_cap = 0;
}

static size_t elf_size(const unsigned char* data, size_t max_scan) {
    if (max_scan < 64) {
        return 0;
    }
    uint64_t shoff = 0;
    uint16_t shentsize = 0;
    uint16_t shnum = 0;
    uint64_t phoff = 0;
    uint16_t phentsize = 0;
    uint16_t phnum = 0;
    memcpy(&phoff, data + 32, sizeof(phoff));
    memcpy(&shoff, data + 40, sizeof(shoff));
    memcpy(&phentsize, data + 54, sizeof(phentsize));
    memcpy(&phnum, data + 56, sizeof(phnum));
    memcpy(&shentsize, data + 58, sizeof(shentsize));
    memcpy(&shnum, data + 60, sizeof(shnum));
    uint64_t a = shoff + (uint64_t)shentsize * (uint64_t)shnum;
    uint64_t b = phoff + (uint64_t)phentsize * (uint64_t)phnum;
    uint64_t size = a > b ? a : b;
    if (size == 0 || size > (uint64_t)max_scan) {
        return 0;
    }
    return (size_t)size;
}

static int get_managed_code(const void* image, void** out_code, size_t* out_size) {
    if (!image || !out_code || !out_size) {
        return -1;
    }
    const unsigned char* bytes = (const unsigned char*)image;
    uint32_t magic = 0;
    memcpy(&magic, bytes, sizeof(magic));

    const void* code = image;
    size_t size = 0;

    if (magic == 0x464c457fU || magic == 0x7f454c46U) {
        size = elf_size(bytes, 256U * 1024U * 1024U);
    } else if (magic == 0xba55ed50U || magic == 0x50ed55baU) {
        uint64_t fat_size = 0;
        memcpy(&fat_size, bytes + 8, sizeof(fat_size));
        size = (size_t)fat_size + 16U;
    } else if (magic == 0x466243b1U || magic == 0xb1436246U) {
        const void* wrapped = NULL;
        memcpy(&wrapped, bytes + 16, sizeof(wrapped));
        if (!wrapped) {
            return -1;
        }
        const unsigned char* wrapped_bytes = (const unsigned char*)wrapped;
        uint64_t fat_size = 0;
        memcpy(&fat_size, wrapped_bytes + 8, sizeof(fat_size));
        code = wrapped;
        size = (size_t)fat_size + 16U;
    } else if (bytes[0] == '/' && bytes[1] == '/') {
        size = strlen((const char*)image) + 1U;
    }

    if (size == 0 || size > 512U * 1024U * 1024U) {
        return -1;
    }
    void* managed = malloc(size);
    if (!managed) {
        return -1;
    }
    memcpy(managed, code, size);
    *out_code = managed;
    *out_size = size;
    return 0;
}

static struct code_item* code_find_locked(void* key) {
    struct code_item* cur = g_code_map;
    while (cur) {
        if (cur->key == key) {
            return cur;
        }
        cur = cur->next;
    }
    return NULL;
}

static void code_set(void* key, void* code, size_t size, const char* name) {
    if (!key) {
        free(code);
        return;
    }
    pthread_mutex_lock(&g_map_lock);
    struct code_item* item = code_find_locked(key);
    if (!item) {
        item = (struct code_item*)calloc(1, sizeof(*item));
        item->key = key;
        item->next = g_code_map;
        g_code_map = item;
    } else if (item->code != code) {
        if (item->owns_code) {
            free(item->code);
        }
        free(item->name);
        item->name = NULL;
        item->disassembly_started = 0;
    }
    item->code = code;
    item->size = size;
    item->owns_code = 1;
    if (name) {
        item->name = strdup(name);
    }
    pthread_mutex_unlock(&g_map_lock);
}

static void code_copy_key(void* old_key, void* new_key, const char* name) {
    if (!old_key || !new_key) {
        return;
    }
    pthread_mutex_lock(&g_map_lock);
    struct code_item* old_item = code_find_locked(old_key);
    if (old_item) {
        struct code_item* item = code_find_locked(new_key);
        if (!item) {
            item = (struct code_item*)calloc(1, sizeof(*item));
            item->key = new_key;
            item->next = g_code_map;
            g_code_map = item;
        }
        if (item->owns_code && item->code != old_item->code) {
            free(item->code);
        }
        item->code = old_item->code;
        item->size = old_item->size;
        item->owns_code = 0;
        free(item->name);
        item->name = name ? strdup(name) : (old_item->name ? strdup(old_item->name) : NULL);
        item->disassembly_started = 0;
    }
    pthread_mutex_unlock(&g_map_lock);
}

static void code_set_name(void* key, const char* name) {
    if (!key || !name || !name[0]) {
        return;
    }
    pthread_mutex_lock(&g_map_lock);
    struct code_item* item = code_find_locked(key);
    if (!item) {
        item = (struct code_item*)calloc(1, sizeof(*item));
        item->key = key;
        item->next = g_code_map;
        g_code_map = item;
    }
    if (!item->name || strcmp(item->name, name) != 0) {
        free(item->name);
        item->name = strdup(name);
    }
    pthread_mutex_unlock(&g_map_lock);
}

static char* code_name_for(CUfunction func) {
    pthread_mutex_lock(&g_map_lock);
    struct code_item* item = code_find_locked(func);
    char* ret = item && item->name ? strdup(item->name) : NULL;
    pthread_mutex_unlock(&g_map_lock);
    if (ret) {
        return ret;
    }

    refresh_driver_symbols();
    if (func && real_cuFuncGetName) {
        const char* driver_name = NULL;
        if (real_cuFuncGetName(&driver_name, func) == CU_SUCCESS &&
            driver_name && driver_name[0]) {
            code_set_name(func, driver_name);
            return strdup(driver_name);
        }
    }
    if (func) {
        code_set_name(func, "<unknown>");
    }
    return strdup("<unknown>");
}

static char* runtime_name_for(const void* func) {
    char* name = code_name_for((CUfunction)func);
    if (name && strcmp(name, "<unknown>") != 0) {
        return name;
    }
    free(name);

    Dl_info info;
    memset(&info, 0, sizeof(info));
    if (func && dladdr(func, &info) && info.dli_sname && info.dli_sname[0]) {
        code_set_name((CUfunction)func, info.dli_sname);
        return strdup(info.dli_sname);
    }
    if (func) {
        code_set_name((CUfunction)func, "<runtime_kernel>");
    }
    return strdup("<runtime_kernel>");
}

static int write_kernel_binary_once(CUfunction func, const char* kernel_name,
                                    char* path_out, size_t path_size) {
    int should_write = 0;
    void* code = NULL;
    size_t size = 0;
    pthread_mutex_lock(&g_map_lock);
    struct code_item* item = code_find_locked(func);
    if (item && item->code && item->size > 0 && !item->disassembly_started) {
        item->disassembly_started = 1;
        code = item->code;
        size = item->size;
        should_write = 1;
    }
    pthread_mutex_unlock(&g_map_lock);

    if (!should_write) {
        return -1;
    }

    const char* dir = getenv("WEAVER_TRACE_DIR");
    if (!dir || !dir[0]) {
        dir = "/tmp";
    }
    mkdir(dir, 0755);
    unsigned long long seq = __sync_add_and_fetch(&g_launch_seq, 1);
    snprintf(path_out, path_size, "%s/weaver_kernel_%d_%llu.bin", dir, getpid(), seq);
    FILE* fp = fopen(path_out, "wb");
    if (!fp) {
        return -1;
    }
    fwrite(code, 1, size, fp);
    fclose(fp);

    char escaped[1024];
    json_escape(kernel_name, escaped, sizeof(escaped));
    send_json(
        "{\"ts_ns\":%lld,\"pid\":%d,\"tid\":%llu,\"layer\":\"neutrino\",\"kind\":\"binary_captured\",\"kernel_name\":\"%s\",\"payload\":{\"path\":\"%s\",\"bytes\":%zu,\"func\":\"%p\"}}",
        now_ns(), getpid(), (unsigned long long)pthread_self(), escaped, path_out, size, func);
    return 0;
}

static void launch_disassembler(CUfunction func, const char* kernel_name) {
    if (!g_disasm_enabled) {
        return;
    }

    char path[1024];
    if (write_kernel_binary_once(func, kernel_name, path, sizeof(path)) != 0) {
        return;
    }

    pid_t pid = fork();
    if (pid < 0) {
        return;
    }
    if (pid == 0) {
        const char* py = getenv("WEAVER_PYTHON");
        const char* helper = getenv("WEAVER_DISASM_HELPER");
        const char* sock = getenv("WEAVER_SOCK");
        if (!py || !py[0]) {
            py = "python3";
        }
        if (!helper || !helper[0]) {
            helper = "weaver.collector.disassemble";
        }
        if (!sock || !sock[0]) {
            sock = "/tmp/weaver.sock";
        }
        unsetenv("LD_PRELOAD");
        unsetenv("DYLD_INSERT_LIBRARIES");
        execlp(py, py, "-m", helper, "--binary", path, "--kernel", kernel_name,
               "--sock", sock, (char*)NULL);
        _exit(127);
    }
}

static struct stream_anchor* get_stream_anchor(CUstream stream) {
    if (!g_sync_stream_anchor || !real_cuEventCreate || !real_cuEventRecord ||
        !real_cuEventSynchronize) {
        return NULL;
    }

    pthread_mutex_lock(&g_map_lock);
    struct stream_anchor* cur = g_anchors;
    while (cur) {
        if (cur->stream == stream) {
            pthread_mutex_unlock(&g_map_lock);
            return cur->valid ? cur : NULL;
        }
        cur = cur->next;
    }
    struct stream_anchor* anchor = (struct stream_anchor*)calloc(1, sizeof(*anchor));
    anchor->stream = stream;
    anchor->next = g_anchors;
    g_anchors = anchor;
    pthread_mutex_unlock(&g_map_lock);

    if (real_cuEventCreate(&anchor->event, CU_EVENT_DEFAULT) != CU_SUCCESS) {
        return NULL;
    }
    if (real_cuEventRecord(anchor->event, stream) != CU_SUCCESS) {
        return NULL;
    }
    if (real_cuEventSynchronize(anchor->event) != CU_SUCCESS) {
        return NULL;
    }
    anchor->host_ns = now_ns();
    anchor->valid = 1;
    send_json(
        "{\"ts_ns\":%lld,\"pid\":%d,\"tid\":%llu,\"layer\":\"cuda\",\"kind\":\"stream_anchor\",\"payload\":{\"stream\":\"%p\",\"host_ns\":%lld}}",
        anchor->host_ns, getpid(), (unsigned long long)pthread_self(), stream, anchor->host_ns);
    return anchor;
}

static struct event_pair* acquire_event_pair(CUevent* start_event, CUevent* end_event) {
    *start_event = NULL;
    *end_event = NULL;
    if (!cuda_events_ready()) {
        return NULL;
    }

    struct event_pair* pair = NULL;
    if (g_cuda_event_pool_enabled) {
        pthread_mutex_lock(&g_map_lock);
        pair = g_event_pool;
        if (pair) {
            g_event_pool = pair->next;
        }
        pthread_mutex_unlock(&g_map_lock);
        if (pair) {
            pair->next = NULL;
            *start_event = pair->start_event;
            *end_event = pair->end_event;
            return pair;
        }
    }

    pair = (struct event_pair*)calloc(1, sizeof(*pair));
    if (!pair) {
        return NULL;
    }
    if (real_cuEventCreate(&pair->start_event, CU_EVENT_DEFAULT) != CU_SUCCESS) {
        *start_event = NULL;
        free(pair);
        return NULL;
    }
    if (real_cuEventCreate(&pair->end_event, CU_EVENT_DEFAULT) != CU_SUCCESS) {
        if (real_cuEventDestroy && pair->start_event) {
            real_cuEventDestroy(pair->start_event);
        }
        *start_event = NULL;
        *end_event = NULL;
        free(pair);
        return NULL;
    }
    *start_event = pair->start_event;
    *end_event = pair->end_event;
    return pair;
}

static void release_event_pair(struct event_pair* pair, int destroy) {
    if (!pair) {
        return;
    }
    if (!destroy && g_cuda_event_pool_enabled && pair->start_event && pair->end_event) {
        pthread_mutex_lock(&g_map_lock);
        pair->next = g_event_pool;
        g_event_pool = pair;
        pthread_mutex_unlock(&g_map_lock);
        return;
    }
    if (real_cuEventDestroy) {
        if (pair->start_event) {
            real_cuEventDestroy(pair->start_event);
        }
        if (pair->end_event) {
            real_cuEventDestroy(pair->end_event);
        }
    }
    free(pair);
}

static struct event_pair* begin_event_timing(CUstream stream, CUevent* start_event, CUevent* end_event) {
    struct event_pair* pair = acquire_event_pair(start_event, end_event);
    if (!pair) {
        return NULL;
    }
    if (real_cuEventRecord(*start_event, stream) == CU_SUCCESS) {
        return pair;
    }
    release_event_pair(pair, 1);
    *start_event = NULL;
    *end_event = NULL;
    return NULL;
}

static void destroy_event_pool(void) {
    pthread_mutex_lock(&g_map_lock);
    struct event_pair* pair = g_event_pool;
    g_event_pool = NULL;
    pthread_mutex_unlock(&g_map_lock);
    while (pair) {
        struct event_pair* next = pair->next;
        release_event_pair(pair, 1);
        pair = next;
    }
}

static void emit_kernel_launch(struct launch_item* item, long long ready_ns,
                               long long gpu_duration_ns, long long gpu_start_ns,
                               long long gpu_end_ns, const char* alignment,
                               int cuda_event_timing) {
    struct launch_emit_record rec;
    memset(&rec, 0, sizeof(rec));
    rec.ts_ns = gpu_start_ns;
    rec.pid = getpid();
    rec.tid = (unsigned long long)pthread_self();
    snprintf(rec.kernel_name, sizeof(rec.kernel_name), "%s",
             item->kernel_name ? item->kernel_name : "<unknown>");
    snprintf(rec.alignment, sizeof(rec.alignment), "%s", alignment ? alignment : "");
    rec.gpu_start_ns = gpu_start_ns;
    rec.gpu_end_ns = gpu_end_ns;
    rec.gpu_duration_ns = gpu_duration_ns;
    rec.cpu_enqueue_start_ns = item->cpu_enqueue_start_ns;
    rec.cpu_enqueue_end_ns = item->cpu_enqueue_end_ns;
    rec.poll_ready_ns = ready_ns;
    rec.func = item->func;
    rec.stream = item->stream;
    rec.launch_id = item->launch_id;
    rec.ret = item->ret;
    rec.grid[0] = item->grid[0];
    rec.grid[1] = item->grid[1];
    rec.grid[2] = item->grid[2];
    rec.block[0] = item->block[0];
    rec.block[1] = item->block[1];
    rec.block[2] = item->block[2];
    rec.shared_mem = item->shared_mem;
    rec.warps_per_block = item->warps_per_block;
    rec.total_warps = item->total_warps;
    rec.cuda_event_timing = cuda_event_timing;
    if (!enqueue_launch_record(&rec)) {
        format_send_launch_record(&rec);
    }
}

static void emit_launch_resolve_error(const char* api_name) {
    char escaped[128];
    json_escape(api_name ? api_name : "", escaped, sizeof(escaped));
    send_json(
        "{\"ts_ns\":%lld,\"pid\":%d,\"tid\":%llu,\"layer\":\"hook\",\"kind\":\"launch_resolve_error\",\"payload\":{\"api\":\"%s\"}}",
        now_ns(), getpid(), (unsigned long long)pthread_self(), escaped);
}

static void destroy_launch_item(struct launch_item* item) {
    if (!item) {
        return;
    }
    release_event_pair(item->event_pair, 0);
    free(item->kernel_name);
    free(item);
}

static int emit_ready_launch_item(struct launch_item* item) {
    if (!item || !real_cuEventQuery || real_cuEventQuery(item->end_event) != CU_SUCCESS) {
        return 0;
    }

    float elapsed_ms = 0.0f;
    long long ready_ns = now_ns();
    long long dur_ns = 0;
    if (real_cuEventElapsedTime &&
        real_cuEventElapsedTime(&elapsed_ms, item->start_event, item->end_event) == CU_SUCCESS) {
        dur_ns = (long long)(elapsed_ms * 1000000.0f);
    }
    if (dur_ns <= 0) {
        dur_ns = item->cpu_enqueue_end_ns - item->cpu_enqueue_start_ns;
    }

    long long gpu_start_ns = ready_ns - dur_ns;
    long long gpu_end_ns = ready_ns;
    const char* alignment = "poll_ready_anchor";
    if (item->anchor && item->anchor->valid && real_cuEventElapsedTime) {
        float start_ms = 0.0f;
        float end_ms = 0.0f;
        if (real_cuEventElapsedTime(&start_ms, item->anchor->event, item->start_event) == CU_SUCCESS &&
            real_cuEventElapsedTime(&end_ms, item->anchor->event, item->end_event) == CU_SUCCESS) {
            gpu_start_ns = item->anchor->host_ns + (long long)(start_ms * 1000000.0f);
            gpu_end_ns = item->anchor->host_ns + (long long)(end_ms * 1000000.0f);
            alignment = "stream_anchor";
        }
    }

    emit_kernel_launch(item, ready_ns, dur_ns, gpu_start_ns, gpu_end_ns, alignment, 1);
    return 1;
}

static void queue_launch_item(struct launch_item* item) {
    pthread_mutex_lock(&g_queue_lock);
    item->next = g_pending;
    g_pending = item;
    pthread_cond_signal(&g_queue_cond);
    pthread_mutex_unlock(&g_queue_lock);
}

static void* poller_run(void* unused) {
    (void)unused;
    while (g_should_run) {
        pthread_mutex_lock(&g_queue_lock);
        if (!g_pending && g_should_run) {
            struct timespec deadline;
            clock_gettime(CLOCK_REALTIME, &deadline);
            deadline.tv_nsec += 1000000L;
            if (deadline.tv_nsec >= 1000000000L) {
                deadline.tv_sec += 1;
                deadline.tv_nsec -= 1000000000L;
            }
            pthread_cond_timedwait(&g_queue_cond, &g_queue_lock, &deadline);
        }

        struct launch_item** curp = &g_pending;
        while (*curp) {
            struct launch_item* item = *curp;
            CUresult q = real_cuEventQuery ? real_cuEventQuery(item->end_event) : 1;
            if (q != CU_SUCCESS) {
                curp = &((*curp)->next);
                continue;
            }

            *curp = item->next;
            pthread_mutex_unlock(&g_queue_lock);
            emit_ready_launch_item(item);
            destroy_launch_item(item);

            pthread_mutex_lock(&g_queue_lock);
            curp = &g_pending;
        }
        pthread_mutex_unlock(&g_queue_lock);
    }

    pthread_mutex_lock(&g_queue_lock);
    struct launch_item* item = g_pending;
    g_pending = NULL;
    pthread_mutex_unlock(&g_queue_lock);
    while (item) {
        struct launch_item* next = item->next;
        for (int i = 0; i < 100 && !emit_ready_launch_item(item); i++) {
            struct timespec ts;
            ts.tv_sec = 0;
            ts.tv_nsec = 1000000L;
            nanosleep(&ts, NULL);
        }
        destroy_launch_item(item);
        item = next;
    }
    return NULL;
}

static int cuda_events_ready(void) {
    if (!g_cuda_event_enabled) {
        return 0;
    }
    refresh_driver_symbols();
    return g_cuda_event_enabled && real_cuEventCreate && real_cuEventRecord &&
           real_cuEventQuery && real_cuEventElapsedTime;
}

static void init_runtime(void) {
    g_cuda_event_enabled = env_flag("WEAVER_CUDA_EVENTS", 1);
    g_sync_stream_anchor = env_flag("WEAVER_CUDA_SYNC_ANCHOR", 1);
    g_cuda_event_pool_enabled = env_flag("WEAVER_CUDA_EVENT_POOL", 1);
    g_trace_getproc_errors = env_flag("WEAVER_TRACE_GETPROC_ERRORS", 0);
    g_patch_dlsym = env_flag("WEAVER_PATCH_DLSYM", 0);
    g_patch_getproc = env_flag("WEAVER_PATCH_GETPROC", 1);
    g_disasm_enabled = env_flag("WEAVER_ENABLE_DISASM", 0);
    g_emit_code_events = env_flag("WEAVER_EMIT_CODE_EVENTS", 0);
    g_async_launch_emit = env_flag("WEAVER_ASYNC_LAUNCH_EMIT", 1);
    if (g_disasm_enabled) {
        signal(SIGCHLD, SIG_IGN);
    }

    const char* sock = getenv("WEAVER_SOCK");
    if (!sock || !sock[0]) {
        sock = "/tmp/weaver.sock";
    }
    g_sock = socket(AF_UNIX, SOCK_DGRAM, 0);
    if (g_sock >= 0) {
        memset(&g_addr, 0, sizeof(g_addr));
        g_addr.sun_family = AF_UNIX;
        strncpy(g_addr.sun_path, sock, sizeof(g_addr.sun_path) - 1);
    }

    g_should_run = 1;
    start_launch_emit_thread();
    if (g_cuda_event_enabled && pthread_create(&g_poller, NULL, poller_run, NULL) == 0) {
        g_poller_started = 1;
    }

    send_json(
        "{\"ts_ns\":%lld,\"pid\":%d,\"tid\":%llu,\"layer\":\"hook\",\"kind\":\"init\",\"payload\":{\"status\":\"ok\",\"cuda_events\":%s,\"sync_stream_anchor\":%s,\"cuda_event_pool\":%s,\"patch_dlsym\":%s,\"patch_getproc\":%s,\"disasm\":%s,\"emit_code_events\":%s,\"async_launch_emit\":%s,\"has_cuLaunchKernel\":%s,\"has_cudaLaunchKernel\":%s,\"has_cuGetProcAddress\":%s,\"has_ncclAllReduce\":%s}}",
        now_ns(), getpid(), (unsigned long long)pthread_self(),
        (g_cuda_event_enabled && real_cuEventCreate && real_cuEventRecord &&
         real_cuEventQuery && real_cuEventElapsedTime) ? "true" : "false",
        g_sync_stream_anchor ? "true" : "false",
        g_cuda_event_pool_enabled ? "true" : "false",
        g_patch_dlsym ? "true" : "false",
        g_patch_getproc ? "true" : "false",
        g_disasm_enabled ? "true" : "false",
        g_emit_code_events ? "true" : "false",
        g_async_launch_emit ? "true" : "false",
        real_cuLaunchKernel ? "true" : "false",
        real_cudaLaunchKernel ? "true" : "false",
        real_cuGetProcAddress ? "true" : "false",
        real_ncclAllReduce ? "true" : "false");
}

static void init_once(void) {
    pthread_once(&g_init_once, init_runtime);
}

__attribute__((constructor)) static void weaver_init(void) {
    init_once();
}

__attribute__((destructor)) static void weaver_fini(void) {
    g_should_run = 0;
    pthread_cond_broadcast(&g_queue_cond);
    if (g_poller_started) {
        pthread_join(g_poller, NULL);
    }
    stop_launch_emit_thread();
    destroy_event_pool();
    if (g_sock >= 0) {
        close(g_sock);
        g_sock = -1;
    }
}

CUresult cuModuleLoadData(CUmodule* module, const void* image) {
    init_once();
    refresh_driver_symbols();
    if (!real_cuModuleLoadData) {
        return 1;
    }
    CUresult ret = real_cuModuleLoadData(module, image);
    if (ret == CU_SUCCESS && module && *module) {
        void* code = NULL;
        size_t size = 0;
        if (get_managed_code(image, &code, &size) == 0) {
            code_set(*module, code, size, NULL);
        }
    }
    if (g_emit_code_events) {
        send_json(
            "{\"ts_ns\":%lld,\"pid\":%d,\"tid\":%llu,\"layer\":\"cuda\",\"kind\":\"module_load_data\",\"payload\":{\"ret\":%d,\"module\":\"%p\"}}",
            now_ns(), getpid(), (unsigned long long)pthread_self(), ret, module ? *module : NULL);
    }
    return ret;
}

CUresult cuModuleLoadDataEx(CUmodule* module, const void* image, unsigned int numOptions,
                            void* options, void** optionValues) {
    init_once();
    refresh_driver_symbols();
    if (!real_cuModuleLoadDataEx) {
        return 1;
    }
    CUresult ret = real_cuModuleLoadDataEx(module, image, numOptions, options, optionValues);
    if (ret == CU_SUCCESS && module && *module) {
        void* code = NULL;
        size_t size = 0;
        if (get_managed_code(image, &code, &size) == 0) {
            code_set(*module, code, size, NULL);
        }
    }
    if (g_emit_code_events) {
        send_json(
            "{\"ts_ns\":%lld,\"pid\":%d,\"tid\":%llu,\"layer\":\"cuda\",\"kind\":\"module_load_data_ex\",\"payload\":{\"ret\":%d,\"module\":\"%p\"}}",
            now_ns(), getpid(), (unsigned long long)pthread_self(), ret, module ? *module : NULL);
    }
    return ret;
}

CUresult cuModuleLoadFatBinary(CUmodule* module, const void* fatCubin) {
    init_once();
    refresh_driver_symbols();
    if (!real_cuModuleLoadFatBinary) {
        return 1;
    }
    CUresult ret = real_cuModuleLoadFatBinary(module, fatCubin);
    if (ret == CU_SUCCESS && module && *module) {
        void* code = NULL;
        size_t size = 0;
        if (get_managed_code(fatCubin, &code, &size) == 0) {
            code_set(*module, code, size, NULL);
        }
    }
    if (g_emit_code_events) {
        send_json(
            "{\"ts_ns\":%lld,\"pid\":%d,\"tid\":%llu,\"layer\":\"cuda\",\"kind\":\"module_load_fat_binary\",\"payload\":{\"ret\":%d,\"module\":\"%p\"}}",
            now_ns(), getpid(), (unsigned long long)pthread_self(), ret, module ? *module : NULL);
    }
    return ret;
}

CUresult cuLibraryLoadData(CUlibrary* library, const void* code, void* jitOptions,
                           void** jitOptionsValues, unsigned int numJitOptions,
                           void* libraryOptions, void** libraryOptionValues,
                           unsigned int numLibraryOptions) {
    init_once();
    refresh_driver_symbols();
    if (!real_cuLibraryLoadData) {
        return 1;
    }
    CUresult ret = real_cuLibraryLoadData(library, code, jitOptions, jitOptionsValues,
                                          numJitOptions, libraryOptions,
                                          libraryOptionValues, numLibraryOptions);
    if (ret == CU_SUCCESS && library && *library) {
        void* managed = NULL;
        size_t size = 0;
        if (get_managed_code(code, &managed, &size) == 0) {
            code_set(*library, managed, size, NULL);
        }
    }
    if (g_emit_code_events) {
        send_json(
            "{\"ts_ns\":%lld,\"pid\":%d,\"tid\":%llu,\"layer\":\"cuda\",\"kind\":\"library_load_data\",\"payload\":{\"ret\":%d,\"library\":\"%p\"}}",
            now_ns(), getpid(), (unsigned long long)pthread_self(), ret, library ? *library : NULL);
    }
    return ret;
}

CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) {
    init_once();
    refresh_driver_symbols();
    if (!real_cuModuleGetFunction) {
        return 1;
    }
    CUresult ret = real_cuModuleGetFunction(hfunc, hmod, name);
    if (ret == CU_SUCCESS && hfunc && *hfunc) {
        code_copy_key(hmod, *hfunc, name);
    }
    if (g_emit_code_events) {
        char escaped[1024];
        json_escape(name ? name : "", escaped, sizeof(escaped));
        send_json(
            "{\"ts_ns\":%lld,\"pid\":%d,\"tid\":%llu,\"layer\":\"cuda\",\"kind\":\"module_get_function\",\"kernel_name\":\"%s\",\"payload\":{\"ret\":%d,\"name\":\"%s\",\"func\":\"%p\",\"module\":\"%p\"}}",
            now_ns(), getpid(), (unsigned long long)pthread_self(), escaped,
            ret, escaped, hfunc ? *hfunc : NULL, hmod);
    }
    return ret;
}

CUresult cuLibraryGetKernel(CUkernel* pKernel, CUlibrary library, const char* name) {
    init_once();
    refresh_driver_symbols();
    if (!real_cuLibraryGetKernel) {
        return 1;
    }
    CUresult ret = real_cuLibraryGetKernel(pKernel, library, name);
    if (ret == CU_SUCCESS && pKernel && *pKernel) {
        code_copy_key(library, *pKernel, name);
    }
    if (g_emit_code_events) {
        char escaped[1024];
        json_escape(name ? name : "", escaped, sizeof(escaped));
        send_json(
            "{\"ts_ns\":%lld,\"pid\":%d,\"tid\":%llu,\"layer\":\"cuda\",\"kind\":\"library_get_kernel\",\"kernel_name\":\"%s\",\"payload\":{\"ret\":%d,\"name\":\"%s\",\"kernel\":\"%p\",\"library\":\"%p\"}}",
            now_ns(), getpid(), (unsigned long long)pthread_self(), escaped,
            ret, escaped, pKernel ? *pKernel : NULL, library);
    }
    return ret;
}

CUresult cuKernelGetFunction(CUfunction* pFunc, CUkernel kernel) {
    init_once();
    refresh_driver_symbols();
    if (!real_cuKernelGetFunction) {
        return 1;
    }
    CUresult ret = real_cuKernelGetFunction(pFunc, kernel);
    if (ret == CU_SUCCESS && pFunc && *pFunc) {
        code_copy_key(kernel, *pFunc, NULL);
    }
    if (g_emit_code_events) {
        send_json(
            "{\"ts_ns\":%lld,\"pid\":%d,\"tid\":%llu,\"layer\":\"cuda\",\"kind\":\"kernel_get_function\",\"payload\":{\"ret\":%d,\"kernel\":\"%p\",\"func\":\"%p\"}}",
            now_ns(), getpid(), (unsigned long long)pthread_self(), ret, kernel, pFunc ? *pFunc : NULL);
    }
    return ret;
}

CUresult cuLibraryGetModule(CUmodule* pMod, CUlibrary library) {
    init_once();
    refresh_driver_symbols();
    if (!real_cuLibraryGetModule) {
        return 1;
    }
    CUresult ret = real_cuLibraryGetModule(pMod, library);
    if (ret == CU_SUCCESS && pMod && *pMod) {
        code_copy_key(library, *pMod, NULL);
    }
    if (g_emit_code_events) {
        send_json(
            "{\"ts_ns\":%lld,\"pid\":%d,\"tid\":%llu,\"layer\":\"cuda\",\"kind\":\"library_get_module\",\"payload\":{\"ret\":%d,\"module\":\"%p\",\"library\":\"%p\"}}",
            now_ns(), getpid(), (unsigned long long)pthread_self(), ret, pMod ? *pMod : NULL, library);
    }
    return ret;
}

CUresult cuFuncGetName(const char** name, CUfunction hfunc) {
    init_once();
    refresh_driver_symbols();
    if (!real_cuFuncGetName) {
        return 1;
    }
    CUresult ret = real_cuFuncGetName(name, hfunc);
    if (ret == CU_SUCCESS && name && *name) {
        code_set_name(hfunc, *name);
    }
    if (g_emit_code_events) {
        char escaped[1024];
        json_escape((name && *name) ? *name : "", escaped, sizeof(escaped));
        send_json(
            "{\"ts_ns\":%lld,\"pid\":%d,\"tid\":%llu,\"layer\":\"cuda\",\"kind\":\"func_get_name\",\"kernel_name\":\"%s\",\"payload\":{\"ret\":%d,\"name\":\"%s\",\"func\":\"%p\"}}",
            now_ns(), getpid(), (unsigned long long)pthread_self(), escaped,
            ret, escaped, hfunc);
    }
    return ret;
}

static CUresult handle_launch(CUfunction f,
                              unsigned int gridDimX,
                              unsigned int gridDimY,
                              unsigned int gridDimZ,
                              unsigned int blockDimX,
                              unsigned int blockDimY,
                              unsigned int blockDimZ,
                              unsigned int sharedMemBytes,
                              CUstream hStream,
                              void** kernelParams,
                              void** extra,
                              cuLaunchKernel_t launcher) {
    char* name = code_name_for(f);
    launch_disassembler(f, name);

    unsigned long long block_size =
        (unsigned long long)blockDimX * (unsigned long long)blockDimY * (unsigned long long)blockDimZ;
    unsigned long long grid_size =
        (unsigned long long)gridDimX * (unsigned long long)gridDimY * (unsigned long long)gridDimZ;
    unsigned long long warps_per_block = (block_size + WARP_SIZE - 1ULL) / WARP_SIZE;
    unsigned long long total_warps = warps_per_block * grid_size;

    CUevent start_event = NULL;
    CUevent end_event = NULL;
    struct stream_anchor* anchor = NULL;
    if (cuda_events_ready()) {
        anchor = get_stream_anchor(hStream);
    }
    struct event_pair* event_pair = begin_event_timing(hStream, &start_event, &end_event);
    int use_events = event_pair != NULL;

    long long cpu_start = now_ns();
    CUresult ret = launcher(f, gridDimX, gridDimY, gridDimZ,
                            blockDimX, blockDimY, blockDimZ,
                            sharedMemBytes, hStream, kernelParams, extra);
    long long cpu_end = now_ns();

    if (use_events && real_cuEventRecord(end_event, hStream) == CU_SUCCESS) {
        struct launch_item* item = (struct launch_item*)calloc(1, sizeof(*item));
        item->launch_id = __sync_add_and_fetch(&g_launch_seq, 1);
        item->ret = ret;
        item->kernel_name = name;
        item->func = f;
        item->stream = hStream;
        item->start_event = start_event;
        item->end_event = end_event;
        item->event_pair = event_pair;
        item->anchor = anchor;
        item->grid[0] = gridDimX;
        item->grid[1] = gridDimY;
        item->grid[2] = gridDimZ;
        item->block[0] = blockDimX;
        item->block[1] = blockDimY;
        item->block[2] = blockDimZ;
        item->shared_mem = sharedMemBytes;
        item->warps_per_block = warps_per_block;
        item->total_warps = total_warps;
        item->cpu_enqueue_start_ns = cpu_start;
        item->cpu_enqueue_end_ns = cpu_end;
        queue_launch_item(item);
    } else {
        struct launch_item item;
        memset(&item, 0, sizeof(item));
        item.launch_id = __sync_add_and_fetch(&g_launch_seq, 1);
        item.ret = ret;
        item.kernel_name = name;
        item.func = f;
        item.stream = hStream;
        item.grid[0] = gridDimX;
        item.grid[1] = gridDimY;
        item.grid[2] = gridDimZ;
        item.block[0] = blockDimX;
        item.block[1] = blockDimY;
        item.block[2] = blockDimZ;
        item.shared_mem = sharedMemBytes;
        item.warps_per_block = warps_per_block;
        item.total_warps = total_warps;
        item.cpu_enqueue_start_ns = cpu_start;
        item.cpu_enqueue_end_ns = cpu_end;
        emit_kernel_launch(&item, cpu_end, cpu_end - cpu_start, cpu_start, cpu_end,
                           "cpu_enqueue_only", 0);
        release_event_pair(event_pair, 1);
        free(name);
    }
    return ret;
}

CUresult cuLaunchKernel(CUfunction f,
                        unsigned int gridDimX,
                        unsigned int gridDimY,
                        unsigned int gridDimZ,
                        unsigned int blockDimX,
                        unsigned int blockDimY,
                        unsigned int blockDimZ,
                        unsigned int sharedMemBytes,
                        CUstream hStream,
                        void** kernelParams,
                        void** extra) {
    init_once();
    refresh_driver_symbols();
    if (!real_cuLaunchKernel) {
        emit_launch_resolve_error("cuLaunchKernel");
        return 1;
    }
    return handle_launch(f, gridDimX, gridDimY, gridDimZ,
                         blockDimX, blockDimY, blockDimZ,
                         sharedMemBytes, hStream, kernelParams, extra,
                         real_cuLaunchKernel);
}

CUresult cuLaunchKernel_ptsz(CUfunction f,
                             unsigned int gridDimX,
                             unsigned int gridDimY,
                             unsigned int gridDimZ,
                             unsigned int blockDimX,
                             unsigned int blockDimY,
                             unsigned int blockDimZ,
                             unsigned int sharedMemBytes,
                             CUstream hStream,
                             void** kernelParams,
                             void** extra) {
    init_once();
    refresh_driver_symbols();
    cuLaunchKernel_t launcher = real_cuLaunchKernel_ptsz ? real_cuLaunchKernel_ptsz : real_cuLaunchKernel;
    if (!launcher) {
        emit_launch_resolve_error("cuLaunchKernel_ptsz");
        return 1;
    }
    return handle_launch(f, gridDimX, gridDimY, gridDimZ,
                         blockDimX, blockDimY, blockDimZ,
                         sharedMemBytes, hStream, kernelParams, extra,
                         launcher);
}

static CUresult handle_launch_ex(const CUlaunchConfig* config,
                                 CUfunction f,
                                 void** kernelParams,
                                 void** extra,
                                 cuLaunchKernelEx_t launcher,
                                 const char* api_name) {
    if (!config || !launcher) {
        return 1;
    }

    char* name = code_name_for(f);
    launch_disassembler(f, name);

    unsigned long long block_size =
        (unsigned long long)config->blockDimX * (unsigned long long)config->blockDimY *
        (unsigned long long)config->blockDimZ;
    unsigned long long grid_size =
        (unsigned long long)config->gridDimX * (unsigned long long)config->gridDimY *
        (unsigned long long)config->gridDimZ;
    unsigned long long warps_per_block = (block_size + WARP_SIZE - 1ULL) / WARP_SIZE;
    unsigned long long total_warps = warps_per_block * grid_size;

    CUevent start_event = NULL;
    CUevent end_event = NULL;
    struct stream_anchor* anchor = NULL;
    if (cuda_events_ready()) {
        anchor = get_stream_anchor(config->hStream);
    }
    struct event_pair* event_pair = begin_event_timing(config->hStream, &start_event, &end_event);
    int use_events = event_pair != NULL;

    long long cpu_start = now_ns();
    CUresult ret = launcher(config, f, kernelParams, extra);
    long long cpu_end = now_ns();

    if (use_events && real_cuEventRecord(end_event, config->hStream) == CU_SUCCESS) {
        struct launch_item* item = (struct launch_item*)calloc(1, sizeof(*item));
        item->launch_id = __sync_add_and_fetch(&g_launch_seq, 1);
        item->ret = ret;
        item->kernel_name = name;
        item->func = f;
        item->stream = config->hStream;
        item->start_event = start_event;
        item->end_event = end_event;
        item->event_pair = event_pair;
        item->anchor = anchor;
        item->grid[0] = config->gridDimX;
        item->grid[1] = config->gridDimY;
        item->grid[2] = config->gridDimZ;
        item->block[0] = config->blockDimX;
        item->block[1] = config->blockDimY;
        item->block[2] = config->blockDimZ;
        item->shared_mem = config->sharedMemBytes;
        item->warps_per_block = warps_per_block;
        item->total_warps = total_warps;
        item->cpu_enqueue_start_ns = cpu_start;
        item->cpu_enqueue_end_ns = cpu_end;
        queue_launch_item(item);
    } else {
        struct launch_item item;
        memset(&item, 0, sizeof(item));
        item.launch_id = __sync_add_and_fetch(&g_launch_seq, 1);
        item.ret = ret;
        item.kernel_name = name;
        item.func = f;
        item.stream = config->hStream;
        item.grid[0] = config->gridDimX;
        item.grid[1] = config->gridDimY;
        item.grid[2] = config->gridDimZ;
        item.block[0] = config->blockDimX;
        item.block[1] = config->blockDimY;
        item.block[2] = config->blockDimZ;
        item.shared_mem = config->sharedMemBytes;
        item.warps_per_block = warps_per_block;
        item.total_warps = total_warps;
        item.cpu_enqueue_start_ns = cpu_start;
        item.cpu_enqueue_end_ns = cpu_end;
        emit_kernel_launch(&item, cpu_end, cpu_end - cpu_start, cpu_start, cpu_end, api_name, 0);
        release_event_pair(event_pair, 1);
        free(name);
    }
    return ret;
}

CUresult cuLaunchKernelEx(const CUlaunchConfig* config, CUfunction f, void** kernelParams, void** extra) {
    init_once();
    refresh_driver_symbols();
    if (!real_cuLaunchKernelEx) {
        emit_launch_resolve_error("cuLaunchKernelEx");
        return 1;
    }
    return handle_launch_ex(config, f, kernelParams, extra,
                            real_cuLaunchKernelEx, "cuLaunchKernelEx_cpu_enqueue_fallback");
}

CUresult cuLaunchKernelEx_ptsz(const CUlaunchConfig* config, CUfunction f, void** kernelParams, void** extra) {
    init_once();
    refresh_driver_symbols();
    cuLaunchKernelEx_t launcher = real_cuLaunchKernelEx_ptsz ? real_cuLaunchKernelEx_ptsz : real_cuLaunchKernelEx;
    if (!launcher) {
        emit_launch_resolve_error("cuLaunchKernelEx_ptsz");
        return 1;
    }
    return handle_launch_ex(config, f, kernelParams, extra,
                            launcher, "cuLaunchKernelEx_ptsz_cpu_enqueue_fallback");
}

static cudaError_t handle_runtime_launch(const void* func,
                                         cuda_dim3 gridDim,
                                         cuda_dim3 blockDim,
                                         void** args,
                                         size_t sharedMem,
                                         cudaStream_t stream,
                                         cudaLaunchKernel_runtime_t launcher,
                                         const char* api_name) {
    char* name = runtime_name_for(func);
    launch_disassembler((CUfunction)func, name);

    unsigned long long block_size =
        (unsigned long long)blockDim.x * (unsigned long long)blockDim.y * (unsigned long long)blockDim.z;
    unsigned long long grid_size =
        (unsigned long long)gridDim.x * (unsigned long long)gridDim.y * (unsigned long long)gridDim.z;
    unsigned long long warps_per_block = (block_size + WARP_SIZE - 1ULL) / WARP_SIZE;
    unsigned long long total_warps = warps_per_block * grid_size;

    CUevent start_event = NULL;
    CUevent end_event = NULL;
    struct stream_anchor* anchor = NULL;
    if (cuda_events_ready()) {
        anchor = get_stream_anchor((CUstream)stream);
    }
    struct event_pair* event_pair = begin_event_timing((CUstream)stream, &start_event, &end_event);
    int use_events = event_pair != NULL;

    long long cpu_start = now_ns();
    cudaError_t ret = launcher(func, gridDim, blockDim, args, sharedMem, stream);
    long long cpu_end = now_ns();

    if (use_events && real_cuEventRecord(end_event, (CUstream)stream) == CU_SUCCESS) {
        struct launch_item* item = (struct launch_item*)calloc(1, sizeof(*item));
        item->launch_id = __sync_add_and_fetch(&g_launch_seq, 1);
        item->ret = ret;
        item->kernel_name = name;
        item->func = (CUfunction)func;
        item->stream = (CUstream)stream;
        item->start_event = start_event;
        item->end_event = end_event;
        item->event_pair = event_pair;
        item->anchor = anchor;
        item->grid[0] = gridDim.x;
        item->grid[1] = gridDim.y;
        item->grid[2] = gridDim.z;
        item->block[0] = blockDim.x;
        item->block[1] = blockDim.y;
        item->block[2] = blockDim.z;
        item->shared_mem = (unsigned int)sharedMem;
        item->warps_per_block = warps_per_block;
        item->total_warps = total_warps;
        item->cpu_enqueue_start_ns = cpu_start;
        item->cpu_enqueue_end_ns = cpu_end;
        queue_launch_item(item);
    } else {
        struct launch_item item;
        memset(&item, 0, sizeof(item));
        item.launch_id = __sync_add_and_fetch(&g_launch_seq, 1);
        item.ret = ret;
        item.kernel_name = name;
        item.func = (CUfunction)func;
        item.stream = (CUstream)stream;
        item.grid[0] = gridDim.x;
        item.grid[1] = gridDim.y;
        item.grid[2] = gridDim.z;
        item.block[0] = blockDim.x;
        item.block[1] = blockDim.y;
        item.block[2] = blockDim.z;
        item.shared_mem = (unsigned int)sharedMem;
        item.warps_per_block = warps_per_block;
        item.total_warps = total_warps;
        item.cpu_enqueue_start_ns = cpu_start;
        item.cpu_enqueue_end_ns = cpu_end;
        emit_kernel_launch(&item, cpu_end, cpu_end - cpu_start, cpu_start, cpu_end, api_name, 0);
        release_event_pair(event_pair, 1);
        free(name);
    }
    return ret;
}

cudaError_t cudaLaunchKernel(const void* func,
                             cuda_dim3 gridDim,
                             cuda_dim3 blockDim,
                             void** args,
                             size_t sharedMem,
                             cudaStream_t stream) {
    init_once();
    refresh_runtime_symbols();
    if (!real_cudaLaunchKernel) {
        emit_launch_resolve_error("cudaLaunchKernel");
        return 1;
    }
    return handle_runtime_launch(func, gridDim, blockDim, args, sharedMem, stream,
                                 real_cudaLaunchKernel, "runtime_cpu_enqueue_fallback");
}

cudaError_t cudaLaunchKernel_ptsz(const void* func,
                                  cuda_dim3 gridDim,
                                  cuda_dim3 blockDim,
                                  void** args,
                                  size_t sharedMem,
                                  cudaStream_t stream) {
    init_once();
    refresh_runtime_symbols();
    cudaLaunchKernel_runtime_t launcher =
        real_cudaLaunchKernel_ptsz ? real_cudaLaunchKernel_ptsz : real_cudaLaunchKernel;
    if (!launcher) {
        emit_launch_resolve_error("cudaLaunchKernel_ptsz");
        return 1;
    }
    return handle_runtime_launch(func, gridDim, blockDim, args, sharedMem, stream,
                                 launcher, "runtime_ptsz_cpu_enqueue_fallback");
}

cudaError_t cudaLaunchCooperativeKernel(const void* func,
                                        cuda_dim3 gridDim,
                                        cuda_dim3 blockDim,
                                        void** args,
                                        size_t sharedMem,
                                        cudaStream_t stream) {
    init_once();
    refresh_runtime_symbols();
    if (!real_cudaLaunchCooperativeKernel) {
        emit_launch_resolve_error("cudaLaunchCooperativeKernel");
        return 1;
    }
    return handle_runtime_launch(func, gridDim, blockDim, args, sharedMem, stream,
                                 real_cudaLaunchCooperativeKernel,
                                 "runtime_cooperative_cpu_enqueue_fallback");
}

cudaError_t cudaLaunchCooperativeKernel_ptsz(const void* func,
                                             cuda_dim3 gridDim,
                                             cuda_dim3 blockDim,
                                             void** args,
                                             size_t sharedMem,
                                             cudaStream_t stream) {
    init_once();
    refresh_runtime_symbols();
    cudaLaunchCooperativeKernel_t launcher =
        real_cudaLaunchCooperativeKernel_ptsz ?
        real_cudaLaunchCooperativeKernel_ptsz : real_cudaLaunchCooperativeKernel;
    if (!launcher) {
        emit_launch_resolve_error("cudaLaunchCooperativeKernel_ptsz");
        return 1;
    }
    return handle_runtime_launch(func, gridDim, blockDim, args, sharedMem, stream,
                                 launcher,
                                 "runtime_cooperative_ptsz_cpu_enqueue_fallback");
}

static cudaError_t handle_runtime_launch_ex(const cudaLaunchConfig_runtime* config,
                                            const void* func,
                                            void** args,
                                            cudaLaunchKernelExC_t launcher,
                                            const char* api_name) {
    if (!config || !launcher) {
        return 1;
    }

    char* name = runtime_name_for(func);
    launch_disassembler((CUfunction)func, name);

    unsigned long long block_size =
        (unsigned long long)config->blockDim.x * (unsigned long long)config->blockDim.y *
        (unsigned long long)config->blockDim.z;
    unsigned long long grid_size =
        (unsigned long long)config->gridDim.x * (unsigned long long)config->gridDim.y *
        (unsigned long long)config->gridDim.z;
    unsigned long long warps_per_block = (block_size + WARP_SIZE - 1ULL) / WARP_SIZE;
    unsigned long long total_warps = warps_per_block * grid_size;

    CUevent start_event = NULL;
    CUevent end_event = NULL;
    struct stream_anchor* anchor = NULL;
    if (cuda_events_ready()) {
        anchor = get_stream_anchor((CUstream)config->stream);
    }
    struct event_pair* event_pair = begin_event_timing((CUstream)config->stream, &start_event, &end_event);
    int use_events = event_pair != NULL;

    long long cpu_start = now_ns();
    cudaError_t ret = launcher(config, func, args);
    long long cpu_end = now_ns();

    if (use_events && real_cuEventRecord(end_event, (CUstream)config->stream) == CU_SUCCESS) {
        struct launch_item* item = (struct launch_item*)calloc(1, sizeof(*item));
        item->launch_id = __sync_add_and_fetch(&g_launch_seq, 1);
        item->ret = ret;
        item->kernel_name = name;
        item->func = (CUfunction)func;
        item->stream = (CUstream)config->stream;
        item->start_event = start_event;
        item->end_event = end_event;
        item->event_pair = event_pair;
        item->anchor = anchor;
        item->grid[0] = config->gridDim.x;
        item->grid[1] = config->gridDim.y;
        item->grid[2] = config->gridDim.z;
        item->block[0] = config->blockDim.x;
        item->block[1] = config->blockDim.y;
        item->block[2] = config->blockDim.z;
        item->shared_mem = (unsigned int)config->dynamicSmemBytes;
        item->warps_per_block = warps_per_block;
        item->total_warps = total_warps;
        item->cpu_enqueue_start_ns = cpu_start;
        item->cpu_enqueue_end_ns = cpu_end;
        queue_launch_item(item);
    } else {
        struct launch_item item;
        memset(&item, 0, sizeof(item));
        item.launch_id = __sync_add_and_fetch(&g_launch_seq, 1);
        item.ret = ret;
        item.kernel_name = name;
        item.func = (CUfunction)func;
        item.stream = (CUstream)config->stream;
        item.grid[0] = config->gridDim.x;
        item.grid[1] = config->gridDim.y;
        item.grid[2] = config->gridDim.z;
        item.block[0] = config->blockDim.x;
        item.block[1] = config->blockDim.y;
        item.block[2] = config->blockDim.z;
        item.shared_mem = (unsigned int)config->dynamicSmemBytes;
        item.warps_per_block = warps_per_block;
        item.total_warps = total_warps;
        item.cpu_enqueue_start_ns = cpu_start;
        item.cpu_enqueue_end_ns = cpu_end;
        emit_kernel_launch(&item, cpu_end, cpu_end - cpu_start, cpu_start, cpu_end, api_name, 0);
        release_event_pair(event_pair, 1);
        free(name);
    }
    return ret;
}

cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_runtime* config,
                                const void* func,
                                void** args) {
    init_once();
    refresh_runtime_symbols();
    if (!real_cudaLaunchKernelExC) {
        emit_launch_resolve_error("cudaLaunchKernelExC");
        return 1;
    }
    return handle_runtime_launch_ex(config, func, args,
                                    real_cudaLaunchKernelExC,
                                    "runtime_ex_cpu_enqueue_fallback");
}

cudaError_t cudaLaunchKernelExC_ptsz(const cudaLaunchConfig_runtime* config,
                                     const void* func,
                                     void** args) {
    init_once();
    refresh_runtime_symbols();
    cudaLaunchKernelExC_t launcher =
        real_cudaLaunchKernelExC_ptsz ? real_cudaLaunchKernelExC_ptsz : real_cudaLaunchKernelExC;
    if (!launcher) {
        emit_launch_resolve_error("cudaLaunchKernelExC_ptsz");
        return 1;
    }
    return handle_runtime_launch_ex(config, func, args,
                                    launcher,
                                    "runtime_ex_ptsz_cpu_enqueue_fallback");
}

CUresult cuGetProcAddress(const char* symbol,
                          void** pfn,
                          int cudaVersion,
                          uint64_t flags,
                          void* symbolStatus);
CUresult cuGetProcAddress_v2(const char* symbol,
                             void** pfn,
                             int cudaVersion,
                             uint64_t flags,
                             void* symbolStatus);

static void* patch_symbol_pointer(const char* symbol, void* pfn) {
    if (!symbol || !pfn) {
        return pfn;
    }
    if (strcmp(symbol, "cuModuleLoadData") == 0) {
        if (!real_cuModuleLoadData) {
            real_cuModuleLoadData = (cuModuleLoadData_t)pfn;
        }
        return (void*)cuModuleLoadData;
    } else if (strcmp(symbol, "cuModuleLoadDataEx") == 0) {
        if (!real_cuModuleLoadDataEx) {
            real_cuModuleLoadDataEx = (cuModuleLoadDataEx_t)pfn;
        }
        return (void*)cuModuleLoadDataEx;
    } else if (strcmp(symbol, "cuModuleLoadFatBinary") == 0) {
        if (!real_cuModuleLoadFatBinary) {
            real_cuModuleLoadFatBinary = (cuModuleLoadFatBinary_t)pfn;
        }
        return (void*)cuModuleLoadFatBinary;
    } else if (strcmp(symbol, "cuModuleGetFunction") == 0) {
        if (!real_cuModuleGetFunction) {
            real_cuModuleGetFunction = (cuModuleGetFunction_t)pfn;
        }
        return (void*)cuModuleGetFunction;
    } else if (strcmp(symbol, "cuKernelGetFunction") == 0) {
        if (!real_cuKernelGetFunction) {
            real_cuKernelGetFunction = (cuKernelGetFunction_t)pfn;
        }
        return (void*)cuKernelGetFunction;
    } else if (strcmp(symbol, "cuLibraryLoadData") == 0) {
        if (!real_cuLibraryLoadData) {
            real_cuLibraryLoadData = (cuLibraryLoadData_t)pfn;
        }
        return (void*)cuLibraryLoadData;
    } else if (strcmp(symbol, "cuLibraryGetKernel") == 0) {
        if (!real_cuLibraryGetKernel) {
            real_cuLibraryGetKernel = (cuLibraryGetKernel_t)pfn;
        }
        return (void*)cuLibraryGetKernel;
    } else if (strcmp(symbol, "cuLibraryGetModule") == 0) {
        if (!real_cuLibraryGetModule) {
            real_cuLibraryGetModule = (cuLibraryGetModule_t)pfn;
        }
        return (void*)cuLibraryGetModule;
    } else if (strcmp(symbol, "cuFuncGetName") == 0) {
        if (!real_cuFuncGetName) {
            real_cuFuncGetName = (cuFuncGetName_t)pfn;
        }
        return (void*)cuFuncGetName;
    } else if (strcmp(symbol, "cuLaunchKernel") == 0) {
        if (!real_cuLaunchKernel) {
            real_cuLaunchKernel = (cuLaunchKernel_t)pfn;
        }
        return (void*)cuLaunchKernel;
    } else if (strcmp(symbol, "cuLaunchKernel_ptsz") == 0) {
        if (!real_cuLaunchKernel_ptsz) {
            real_cuLaunchKernel_ptsz = (cuLaunchKernel_t)pfn;
        }
        return (void*)cuLaunchKernel_ptsz;
    } else if (strcmp(symbol, "cuLaunchKernelEx") == 0) {
        if (!real_cuLaunchKernelEx) {
            real_cuLaunchKernelEx = (cuLaunchKernelEx_t)pfn;
        }
        return (void*)cuLaunchKernelEx;
    } else if (strcmp(symbol, "cuLaunchKernelEx_ptsz") == 0) {
        if (!real_cuLaunchKernelEx_ptsz) {
            real_cuLaunchKernelEx_ptsz = (cuLaunchKernelEx_t)pfn;
        }
        return (void*)cuLaunchKernelEx_ptsz;
    } else if (strcmp(symbol, "cudaLaunchKernel") == 0) {
        if (!real_cudaLaunchKernel) {
            real_cudaLaunchKernel = (cudaLaunchKernel_runtime_t)pfn;
        }
        return (void*)cudaLaunchKernel;
    } else if (strcmp(symbol, "cudaLaunchKernel_ptsz") == 0) {
        if (!real_cudaLaunchKernel_ptsz) {
            real_cudaLaunchKernel_ptsz = (cudaLaunchKernel_runtime_t)pfn;
        }
        return (void*)cudaLaunchKernel_ptsz;
    } else if (strcmp(symbol, "cudaLaunchCooperativeKernel") == 0) {
        if (!real_cudaLaunchCooperativeKernel) {
            real_cudaLaunchCooperativeKernel = (cudaLaunchCooperativeKernel_t)pfn;
        }
        return (void*)cudaLaunchCooperativeKernel;
    } else if (strcmp(symbol, "cudaLaunchCooperativeKernel_ptsz") == 0) {
        if (!real_cudaLaunchCooperativeKernel_ptsz) {
            real_cudaLaunchCooperativeKernel_ptsz = (cudaLaunchCooperativeKernel_t)pfn;
        }
        return (void*)cudaLaunchCooperativeKernel_ptsz;
    } else if (strcmp(symbol, "cudaLaunchKernelExC") == 0) {
        if (!real_cudaLaunchKernelExC) {
            real_cudaLaunchKernelExC = (cudaLaunchKernelExC_t)pfn;
        }
        return (void*)cudaLaunchKernelExC;
    } else if (strcmp(symbol, "cudaLaunchKernelExC_ptsz") == 0) {
        if (!real_cudaLaunchKernelExC_ptsz) {
            real_cudaLaunchKernelExC_ptsz = (cudaLaunchKernelExC_t)pfn;
        }
        return (void*)cudaLaunchKernelExC_ptsz;
    } else if (strcmp(symbol, "cuGetProcAddress") == 0) {
        if (!real_cuGetProcAddress && pfn != (void*)cuGetProcAddress) {
            real_cuGetProcAddress = (cuGetProcAddress_t)pfn;
        }
        return (void*)cuGetProcAddress;
    } else if (strcmp(symbol, "cuGetProcAddress_v2") == 0) {
        if (!real_cuGetProcAddress_v2 && pfn != (void*)cuGetProcAddress_v2) {
            real_cuGetProcAddress_v2 = (cuGetProcAddress_t)pfn;
        }
        return (void*)cuGetProcAddress_v2;
    }
    return pfn;
}

static void patch_driver_proc_address(const char* symbol, void** pfn) {
    if (!symbol || !pfn || !*pfn) {
        return;
    }
    *pfn = patch_symbol_pointer(symbol, *pfn);
}

static CUresult call_real_cu_get_proc_address(cuGetProcAddress_t getter,
                                              const char* symbol,
                                              void** pfn,
                                              int cudaVersion,
                                              uint64_t flags,
                                              void* symbolStatus,
                                              int patch_result) {
    if (!getter) {
        refresh_getproc_symbols();
        getter = real_cuGetProcAddress ? real_cuGetProcAddress : real_cuGetProcAddress_v2;
    }
    if (!getter) {
        void* fn = dlsym_next_any(symbol, NULL);
        if (!fn) {
            fn = dlsym_libcuda(symbol);
        }
        if (!fn || !pfn) {
            return 1;
        }
        *pfn = patch_result ? patch_symbol_pointer(symbol, fn) : fn;
        return CU_SUCCESS;
    }
    CUresult ret = getter(symbol, pfn, cudaVersion, flags, symbolStatus);
    if (ret == CU_SUCCESS && patch_result) {
        patch_driver_proc_address(symbol, pfn);
    } else if (g_trace_getproc_errors) {
        char escaped[256];
        json_escape(symbol ? symbol : "", escaped, sizeof(escaped));
        send_json(
            "{\"ts_ns\":%lld,\"pid\":%d,\"tid\":%llu,\"layer\":\"hook\",\"kind\":\"cu_get_proc_address_error\",\"payload\":{\"symbol\":\"%s\",\"ret\":%d}}",
            now_ns(), getpid(), (unsigned long long)pthread_self(), escaped, ret);
    }
    return ret;
}

CUresult cuGetProcAddress(const char* symbol,
                          void** pfn,
                          int cudaVersion,
                          uint64_t flags,
                          void* symbolStatus) {
    init_once();
    refresh_getproc_symbols();
    cuGetProcAddress_t getter = real_cuGetProcAddress ? real_cuGetProcAddress : real_cuGetProcAddress_v2;
    int patch_result = g_patch_getproc && !is_blocked_interposer_caller(__builtin_return_address(0));
    if (!patch_result) {
        if (!getter || getter == cuGetProcAddress || getter == cuGetProcAddress_v2) {
            return 1;
        }
        return getter(symbol, pfn, cudaVersion, flags, symbolStatus);
    }
    return call_real_cu_get_proc_address(getter, symbol, pfn, cudaVersion, flags, symbolStatus, patch_result);
}

CUresult cuGetProcAddress_v2(const char* symbol,
                             void** pfn,
                             int cudaVersion,
                             uint64_t flags,
                             void* symbolStatus) {
    init_once();
    refresh_getproc_symbols();
    cuGetProcAddress_t getter = real_cuGetProcAddress_v2 ? real_cuGetProcAddress_v2 : real_cuGetProcAddress;
    int patch_result = g_patch_getproc && !is_blocked_interposer_caller(__builtin_return_address(0));
    if (!patch_result) {
        if (!getter || getter == cuGetProcAddress || getter == cuGetProcAddress_v2) {
            return 1;
        }
        return getter(symbol, pfn, cudaVersion, flags, symbolStatus);
    }
    return call_real_cu_get_proc_address(getter, symbol, pfn, cudaVersion, flags, symbolStatus, patch_result);
}

#ifdef __linux__
void* dlsym(void* handle, const char* symbol) {
    dlsym_fn_t real_dlsym = get_real_dlsym();
    if (!real_dlsym) {
        return NULL;
    }
    uintptr_t symbol_addr = (uintptr_t)symbol;
    if (symbol_addr == 0) {
        return NULL;
    }
    const char* sym = (const char*)symbol_addr;
    if (strcmp(sym, "dlsym") == 0) {
        return (void*)real_dlsym;
    }
    void* fn = real_dlsym(handle, sym);
    if (!g_patch_dlsym || !is_interposable_cuda_symbol(sym) ||
        is_blocked_interposer_caller(__builtin_return_address(0))) {
        return fn;
    }
    return patch_symbol_pointer(sym, fn);
}
#endif

static void send_nccl_event(const char* kind, size_t count, int ret,
                            long long start_ns, long long end_ns) {
    send_json(
        "{\"ts_ns\":%lld,\"pid\":%d,\"tid\":%llu,\"layer\":\"cuda\",\"kind\":\"%s\",\"source\":\"weaver_hook\",\"kernel_name\":\"nccl::%s\",\"cpu_enqueue_start_ns\":%lld,\"cpu_enqueue_end_ns\":%lld,\"dur_ns\":%lld,\"payload\":{\"ret\":%d,\"count\":%zu,\"component\":\"nccl\",\"cuda_event_timing\":false}}",
        start_ns, getpid(), (unsigned long long)pthread_self(), kind, kind,
        start_ns, end_ns, end_ns - start_ns, ret, count);
}

ncclResult_t ncclAllReduce(const void* sendbuff,
                           void* recvbuff,
                           size_t count,
                           ncclDataType_t datatype,
                           ncclRedOp_t op,
                           ncclComm_t comm,
                           CUstream stream) {
    init_once();
    refresh_nccl_symbols();
    if (!real_ncclAllReduce) {
        return -1;
    }
    long long start = now_ns();
    ncclResult_t ret = real_ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
    long long end = now_ns();
    send_nccl_event("nccl_all_reduce", count, ret, start, end);
    return ret;
}

ncclResult_t ncclAllGather(const void* sendbuff,
                           void* recvbuff,
                           size_t sendcount,
                           ncclDataType_t datatype,
                           ncclComm_t comm,
                           CUstream stream) {
    init_once();
    refresh_nccl_symbols();
    if (!real_ncclAllGather) {
        return -1;
    }
    long long start = now_ns();
    ncclResult_t ret = real_ncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream);
    long long end = now_ns();
    send_nccl_event("nccl_all_gather", sendcount, ret, start, end);
    return ret;
}

ncclResult_t ncclReduceScatter(const void* sendbuff,
                               void* recvbuff,
                               size_t recvcount,
                               ncclDataType_t datatype,
                               ncclRedOp_t op,
                               ncclComm_t comm,
                               CUstream stream) {
    init_once();
    refresh_nccl_symbols();
    if (!real_ncclReduceScatter) {
        return -1;
    }
    long long start = now_ns();
    ncclResult_t ret = real_ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
    long long end = now_ns();
    send_nccl_event("nccl_reduce_scatter", recvcount, ret, start, end);
    return ret;
}

ncclResult_t ncclBroadcast(const void* sendbuff,
                           void* recvbuff,
                           size_t count,
                           ncclDataType_t datatype,
                           int root,
                           ncclComm_t comm,
                           CUstream stream) {
    init_once();
    refresh_nccl_symbols();
    if (!real_ncclBroadcast) {
        return -1;
    }
    long long start = now_ns();
    ncclResult_t ret = real_ncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream);
    long long end = now_ns();
    send_nccl_event("nccl_broadcast", count, ret, start, end);
    return ret;
}
