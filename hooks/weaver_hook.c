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

static cuModuleLoadData_t real_cuModuleLoadData = NULL;
static cuModuleLoadDataEx_t real_cuModuleLoadDataEx = NULL;
static cuModuleLoadFatBinary_t real_cuModuleLoadFatBinary = NULL;
static cuModuleGetFunction_t real_cuModuleGetFunction = NULL;
static cuKernelGetFunction_t real_cuKernelGetFunction = NULL;
static cuLibraryLoadData_t real_cuLibraryLoadData = NULL;
static cuLibraryGetKernel_t real_cuLibraryGetKernel = NULL;
static cuLibraryGetModule_t real_cuLibraryGetModule = NULL;
static cuLaunchKernel_t real_cuLaunchKernel = NULL;
static cuLaunchKernelEx_t real_cuLaunchKernelEx = NULL;
static cuEventCreate_t real_cuEventCreate = NULL;
static cuEventRecord_t real_cuEventRecord = NULL;
static cuEventQuery_t real_cuEventQuery = NULL;
static cuEventSynchronize_t real_cuEventSynchronize = NULL;
static cuEventElapsedTime_t real_cuEventElapsedTime = NULL;
static cuEventDestroy_t real_cuEventDestroy = NULL;
static cuGetProcAddress_t real_cuGetProcAddress = NULL;
static cudaLaunchKernel_runtime_t real_cudaLaunchKernel = NULL;
static cudaLaunchKernelExC_t real_cudaLaunchKernelExC = NULL;
static cudaLaunchCooperativeKernel_t real_cudaLaunchCooperativeKernel = NULL;

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
static int g_sync_stream_anchor = 0;
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

static struct code_item* g_code_map = NULL;
static struct stream_anchor* g_anchors = NULL;
static struct launch_item* g_pending = NULL;

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

static void* dlsym_next_any(const char* name, const char* alt_name) {
    void* fn = dlsym(RTLD_NEXT, name);
    if (!fn && alt_name) {
        fn = dlsym(RTLD_NEXT, alt_name);
    }
    return fn;
}

static void refresh_driver_symbols(void) {
    if (!real_cuModuleLoadData) {
        real_cuModuleLoadData = (cuModuleLoadData_t)dlsym_next_any("cuModuleLoadData", NULL);
    }
    if (!real_cuModuleLoadDataEx) {
        real_cuModuleLoadDataEx = (cuModuleLoadDataEx_t)dlsym_next_any("cuModuleLoadDataEx", NULL);
    }
    if (!real_cuModuleLoadFatBinary) {
        real_cuModuleLoadFatBinary = (cuModuleLoadFatBinary_t)dlsym_next_any("cuModuleLoadFatBinary", NULL);
    }
    if (!real_cuModuleGetFunction) {
        real_cuModuleGetFunction = (cuModuleGetFunction_t)dlsym_next_any("cuModuleGetFunction", NULL);
    }
    if (!real_cuKernelGetFunction) {
        real_cuKernelGetFunction = (cuKernelGetFunction_t)dlsym_next_any("cuKernelGetFunction", NULL);
    }
    if (!real_cuLibraryLoadData) {
        real_cuLibraryLoadData = (cuLibraryLoadData_t)dlsym_next_any("cuLibraryLoadData", NULL);
    }
    if (!real_cuLibraryGetKernel) {
        real_cuLibraryGetKernel = (cuLibraryGetKernel_t)dlsym_next_any("cuLibraryGetKernel", NULL);
    }
    if (!real_cuLibraryGetModule) {
        real_cuLibraryGetModule = (cuLibraryGetModule_t)dlsym_next_any("cuLibraryGetModule", NULL);
    }
    if (!real_cuLaunchKernel) {
        real_cuLaunchKernel = (cuLaunchKernel_t)dlsym_next_any("cuLaunchKernel", NULL);
    }
    if (!real_cuLaunchKernelEx) {
        real_cuLaunchKernelEx = (cuLaunchKernelEx_t)dlsym_next_any("cuLaunchKernelEx", NULL);
    }
    if (!real_cuEventCreate) {
        real_cuEventCreate = (cuEventCreate_t)dlsym_next_any("cuEventCreate", NULL);
    }
    if (!real_cuEventRecord) {
        real_cuEventRecord = (cuEventRecord_t)dlsym_next_any("cuEventRecord", NULL);
    }
    if (!real_cuEventQuery) {
        real_cuEventQuery = (cuEventQuery_t)dlsym_next_any("cuEventQuery", NULL);
    }
    if (!real_cuEventSynchronize) {
        real_cuEventSynchronize = (cuEventSynchronize_t)dlsym_next_any("cuEventSynchronize", NULL);
    }
    if (!real_cuEventElapsedTime) {
        real_cuEventElapsedTime = (cuEventElapsedTime_t)dlsym_next_any("cuEventElapsedTime", NULL);
    }
    if (!real_cuEventDestroy) {
        real_cuEventDestroy = (cuEventDestroy_t)dlsym_next_any("cuEventDestroy_v2", "cuEventDestroy");
    }
    if (!real_cuGetProcAddress) {
        real_cuGetProcAddress = (cuGetProcAddress_t)dlsym_next_any("cuGetProcAddress", NULL);
    }
}

static void refresh_runtime_symbols(void) {
    if (!real_cudaLaunchKernel) {
        real_cudaLaunchKernel = (cudaLaunchKernel_runtime_t)dlsym_next_any("cudaLaunchKernel", NULL);
    }
    if (!real_cudaLaunchKernelExC) {
        real_cudaLaunchKernelExC = (cudaLaunchKernelExC_t)dlsym_next_any("cudaLaunchKernelExC", NULL);
    }
    if (!real_cudaLaunchCooperativeKernel) {
        real_cudaLaunchCooperativeKernel =
            (cudaLaunchCooperativeKernel_t)dlsym_next_any("cudaLaunchCooperativeKernel", NULL);
    }
}

static void refresh_nccl_symbols(void) {
    if (!real_ncclAllReduce) {
        real_ncclAllReduce = (ncclAllReduce_t)dlsym(RTLD_NEXT, "ncclAllReduce");
    }
    if (!real_ncclAllGather) {
        real_ncclAllGather = (ncclAllGather_t)dlsym(RTLD_NEXT, "ncclAllGather");
    }
    if (!real_ncclReduceScatter) {
        real_ncclReduceScatter = (ncclReduceScatter_t)dlsym(RTLD_NEXT, "ncclReduceScatter");
    }
    if (!real_ncclBroadcast) {
        real_ncclBroadcast = (ncclBroadcast_t)dlsym(RTLD_NEXT, "ncclBroadcast");
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

static char* code_name_for(CUfunction func) {
    pthread_mutex_lock(&g_map_lock);
    struct code_item* item = code_find_locked(func);
    char* ret = item && item->name ? strdup(item->name) : strdup("<unknown>");
    pthread_mutex_unlock(&g_map_lock);
    return ret;
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
        return strdup(info.dli_sname);
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
    if (!env_flag("WEAVER_ENABLE_DISASM", 1)) {
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

static void emit_kernel_launch(struct launch_item* item, long long ready_ns,
                               long long gpu_duration_ns, long long gpu_start_ns,
                               long long gpu_end_ns, const char* alignment) {
    char escaped[1024];
    json_escape(item->kernel_name, escaped, sizeof(escaped));
    send_json(
        "{\"ts_ns\":%lld,\"pid\":%d,\"tid\":%llu,\"layer\":\"cuda\",\"kind\":\"kernel_launch\",\"source\":\"weaver_hook\",\"kernel_name\":\"%s\",\"gpu_start_ns\":%lld,\"gpu_end_ns\":%lld,\"dur_ns\":%lld,\"cpu_enqueue_start_ns\":%lld,\"cpu_enqueue_end_ns\":%lld,\"stream\":\"%p\",\"payload\":{\"launch_id\":%llu,\"ret\":%d,\"kernel\":\"%s\",\"kernel_name\":\"%s\",\"func\":\"%p\",\"stream\":\"%p\",\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],\"shared_mem\":%u,\"shared_memory\":%u,\"threads_per_block\":%llu,\"blocks_total\":%llu,\"warps_per_block\":%llu,\"total_warps\":%llu,\"warp_size\":32,\"warp_scope\":\"block_runtime\",\"gpu_duration_ns\":%lld,\"gpu_start_ns\":%lld,\"gpu_end_ns\":%lld,\"cuda_event_timing\":true,\"time_alignment\":\"%s\",\"poll_ready_ns\":%lld,\"cpu_enqueue_start_ns\":%lld,\"cpu_enqueue_end_ns\":%lld}}",
        gpu_start_ns, getpid(), (unsigned long long)pthread_self(), escaped,
        gpu_start_ns, gpu_end_ns, gpu_duration_ns,
        item->cpu_enqueue_start_ns, item->cpu_enqueue_end_ns, item->stream,
        item->launch_id, item->ret, escaped, escaped, item->func, item->stream,
        item->grid[0], item->grid[1], item->grid[2],
        item->block[0], item->block[1], item->block[2],
        item->shared_mem, item->shared_mem,
        (unsigned long long)item->block[0] * item->block[1] * item->block[2],
        (unsigned long long)item->grid[0] * item->grid[1] * item->grid[2],
        item->warps_per_block, item->total_warps,
        gpu_duration_ns, gpu_start_ns, gpu_end_ns, alignment, ready_ns,
        item->cpu_enqueue_start_ns, item->cpu_enqueue_end_ns);
}

static void destroy_launch_item(struct launch_item* item) {
    if (!item) {
        return;
    }
    if (real_cuEventDestroy) {
        if (item->start_event) {
            real_cuEventDestroy(item->start_event);
        }
        if (item->end_event) {
            real_cuEventDestroy(item->end_event);
        }
    }
    free(item->kernel_name);
    free(item);
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

            emit_kernel_launch(item, ready_ns, dur_ns, gpu_start_ns, gpu_end_ns, alignment);
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
        destroy_launch_item(item);
        item = next;
    }
    return NULL;
}

static int cuda_events_ready(void) {
    refresh_driver_symbols();
    return g_cuda_event_enabled && real_cuEventCreate && real_cuEventRecord &&
           real_cuEventQuery && real_cuEventElapsedTime;
}

static void init_symbols(void) {
    refresh_driver_symbols();
    refresh_runtime_symbols();
    refresh_nccl_symbols();
}

static void init_runtime(void) {
    init_symbols();
    signal(SIGCHLD, SIG_IGN);
    g_cuda_event_enabled = env_flag("WEAVER_CUDA_EVENTS", 1);
    g_sync_stream_anchor = env_flag("WEAVER_CUDA_SYNC_ANCHOR", 0);

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
    if (pthread_create(&g_poller, NULL, poller_run, NULL) == 0) {
        g_poller_started = 1;
    }

    send_json(
        "{\"ts_ns\":%lld,\"pid\":%d,\"tid\":%llu,\"layer\":\"hook\",\"kind\":\"init\",\"payload\":{\"status\":\"ok\",\"cuda_events\":%s,\"sync_stream_anchor\":%s,\"has_cuLaunchKernel\":%s,\"has_cudaLaunchKernel\":%s,\"has_cuGetProcAddress\":%s,\"has_ncclAllReduce\":%s}}",
        now_ns(), getpid(), (unsigned long long)pthread_self(),
        cuda_events_ready() ? "true" : "false", g_sync_stream_anchor ? "true" : "false",
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
    send_json(
        "{\"ts_ns\":%lld,\"pid\":%d,\"tid\":%llu,\"layer\":\"cuda\",\"kind\":\"module_load_data\",\"payload\":{\"ret\":%d,\"module\":\"%p\"}}",
        now_ns(), getpid(), (unsigned long long)pthread_self(), ret, module ? *module : NULL);
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
    send_json(
        "{\"ts_ns\":%lld,\"pid\":%d,\"tid\":%llu,\"layer\":\"cuda\",\"kind\":\"module_load_data_ex\",\"payload\":{\"ret\":%d,\"module\":\"%p\"}}",
        now_ns(), getpid(), (unsigned long long)pthread_self(), ret, module ? *module : NULL);
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
    send_json(
        "{\"ts_ns\":%lld,\"pid\":%d,\"tid\":%llu,\"layer\":\"cuda\",\"kind\":\"module_load_fat_binary\",\"payload\":{\"ret\":%d,\"module\":\"%p\"}}",
        now_ns(), getpid(), (unsigned long long)pthread_self(), ret, module ? *module : NULL);
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
    send_json(
        "{\"ts_ns\":%lld,\"pid\":%d,\"tid\":%llu,\"layer\":\"cuda\",\"kind\":\"library_load_data\",\"payload\":{\"ret\":%d,\"library\":\"%p\"}}",
        now_ns(), getpid(), (unsigned long long)pthread_self(), ret, library ? *library : NULL);
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
    char escaped[1024];
    json_escape(name ? name : "", escaped, sizeof(escaped));
    send_json(
        "{\"ts_ns\":%lld,\"pid\":%d,\"tid\":%llu,\"layer\":\"cuda\",\"kind\":\"module_get_function\",\"kernel_name\":\"%s\",\"payload\":{\"ret\":%d,\"name\":\"%s\",\"func\":\"%p\",\"module\":\"%p\"}}",
        now_ns(), getpid(), (unsigned long long)pthread_self(), escaped,
        ret, escaped, hfunc ? *hfunc : NULL, hmod);
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
    char escaped[1024];
    json_escape(name ? name : "", escaped, sizeof(escaped));
    send_json(
        "{\"ts_ns\":%lld,\"pid\":%d,\"tid\":%llu,\"layer\":\"cuda\",\"kind\":\"library_get_kernel\",\"kernel_name\":\"%s\",\"payload\":{\"ret\":%d,\"name\":\"%s\",\"kernel\":\"%p\",\"library\":\"%p\"}}",
        now_ns(), getpid(), (unsigned long long)pthread_self(), escaped,
        ret, escaped, pKernel ? *pKernel : NULL, library);
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
    send_json(
        "{\"ts_ns\":%lld,\"pid\":%d,\"tid\":%llu,\"layer\":\"cuda\",\"kind\":\"kernel_get_function\",\"payload\":{\"ret\":%d,\"kernel\":\"%p\",\"func\":\"%p\"}}",
        now_ns(), getpid(), (unsigned long long)pthread_self(), ret, kernel, pFunc ? *pFunc : NULL);
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
    send_json(
        "{\"ts_ns\":%lld,\"pid\":%d,\"tid\":%llu,\"layer\":\"cuda\",\"kind\":\"library_get_module\",\"payload\":{\"ret\":%d,\"module\":\"%p\",\"library\":\"%p\"}}",
        now_ns(), getpid(), (unsigned long long)pthread_self(), ret, pMod ? *pMod : NULL, library);
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
    int use_events = cuda_events_ready() &&
                     real_cuEventCreate(&start_event, CU_EVENT_DEFAULT) == CU_SUCCESS &&
                     real_cuEventCreate(&end_event, CU_EVENT_DEFAULT) == CU_SUCCESS &&
                     real_cuEventRecord(start_event, hStream) == CU_SUCCESS;
    if (use_events) {
        anchor = get_stream_anchor(hStream);
    }

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
                           "cpu_enqueue_fallback");
        if (real_cuEventDestroy) {
            if (start_event) {
                real_cuEventDestroy(start_event);
            }
            if (end_event) {
                real_cuEventDestroy(end_event);
            }
        }
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
        return 1;
    }
    return handle_launch(f, gridDimX, gridDimY, gridDimZ,
                         blockDimX, blockDimY, blockDimZ,
                         sharedMemBytes, hStream, kernelParams, extra,
                         real_cuLaunchKernel);
}

CUresult cuLaunchKernelEx(const CUlaunchConfig* config, CUfunction f, void** kernelParams, void** extra) {
    init_once();
    refresh_driver_symbols();
    if (!real_cuLaunchKernelEx || !config) {
        return 1;
    }
    char* name = code_name_for(f);
    launch_disassembler(f, name);
    free(name);

    // The Ex API is timed with CPU enqueue metadata only. The driver eventually
    // dispatches through lower launch paths on most frameworks, so regular
    // cuLaunchKernel events still cover common CUDA kernels.
    long long start = now_ns();
    CUresult ret = real_cuLaunchKernelEx(config, f, kernelParams, extra);
    long long end = now_ns();
    unsigned long long block_size =
        (unsigned long long)config->blockDimX * config->blockDimY * config->blockDimZ;
    unsigned long long grid_size =
        (unsigned long long)config->gridDimX * config->gridDimY * config->gridDimZ;
    unsigned long long warps = ((block_size + WARP_SIZE - 1ULL) / WARP_SIZE) * grid_size;
    send_json(
        "{\"ts_ns\":%lld,\"pid\":%d,\"tid\":%llu,\"layer\":\"cuda\",\"kind\":\"kernel_launch_ex\",\"source\":\"weaver_hook\",\"kernel_name\":\"<cuLaunchKernelEx>\",\"cpu_enqueue_start_ns\":%lld,\"cpu_enqueue_end_ns\":%lld,\"dur_ns\":%lld,\"payload\":{\"ret\":%d,\"func\":\"%p\",\"stream\":\"%p\",\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],\"shared_mem\":%u,\"total_warps\":%llu,\"warp_scope\":\"block_runtime\",\"cuda_event_timing\":false}}",
        start, getpid(), (unsigned long long)pthread_self(), start, end, end - start,
        ret, f, config->hStream, config->gridDimX, config->gridDimY, config->gridDimZ,
        config->blockDimX, config->blockDimY, config->blockDimZ, config->sharedMemBytes, warps);
    return ret;
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
    int use_events = cuda_events_ready() &&
                     real_cuEventCreate(&start_event, CU_EVENT_DEFAULT) == CU_SUCCESS &&
                     real_cuEventCreate(&end_event, CU_EVENT_DEFAULT) == CU_SUCCESS &&
                     real_cuEventRecord(start_event, (CUstream)stream) == CU_SUCCESS;
    if (use_events) {
        anchor = get_stream_anchor((CUstream)stream);
    }

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
        emit_kernel_launch(&item, cpu_end, cpu_end - cpu_start, cpu_start, cpu_end, api_name);
        if (real_cuEventDestroy) {
            if (start_event) {
                real_cuEventDestroy(start_event);
            }
            if (end_event) {
                real_cuEventDestroy(end_event);
            }
        }
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
        return 1;
    }
    return handle_runtime_launch(func, gridDim, blockDim, args, sharedMem, stream,
                                 real_cudaLaunchKernel, "runtime_cpu_enqueue_fallback");
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
        return 1;
    }
    return handle_runtime_launch(func, gridDim, blockDim, args, sharedMem, stream,
                                 real_cudaLaunchCooperativeKernel,
                                 "runtime_cooperative_cpu_enqueue_fallback");
}

cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_runtime* config,
                                const void* func,
                                void** args) {
    init_once();
    refresh_runtime_symbols();
    if (!real_cudaLaunchKernelExC || !config) {
        return 1;
    }
    char* name = runtime_name_for(func);
    launch_disassembler((CUfunction)func, name);
    free(name);

    long long start = now_ns();
    cudaError_t ret = real_cudaLaunchKernelExC(config, func, args);
    long long end = now_ns();
    unsigned long long block_size =
        (unsigned long long)config->blockDim.x * config->blockDim.y * config->blockDim.z;
    unsigned long long grid_size =
        (unsigned long long)config->gridDim.x * config->gridDim.y * config->gridDim.z;
    unsigned long long warps = ((block_size + WARP_SIZE - 1ULL) / WARP_SIZE) * grid_size;
    send_json(
        "{\"ts_ns\":%lld,\"pid\":%d,\"tid\":%llu,\"layer\":\"cuda\",\"kind\":\"kernel_launch_ex\",\"source\":\"weaver_hook_runtime\",\"kernel_name\":\"<cudaLaunchKernelExC>\",\"cpu_enqueue_start_ns\":%lld,\"cpu_enqueue_end_ns\":%lld,\"dur_ns\":%lld,\"payload\":{\"ret\":%d,\"func\":\"%p\",\"stream\":\"%p\",\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],\"shared_mem\":%zu,\"total_warps\":%llu,\"warp_scope\":\"block_runtime\",\"cuda_event_timing\":false}}",
        start, getpid(), (unsigned long long)pthread_self(), start, end, end - start,
        ret, func, config->stream, config->gridDim.x, config->gridDim.y, config->gridDim.z,
        config->blockDim.x, config->blockDim.y, config->blockDim.z,
        config->dynamicSmemBytes, warps);
    return ret;
}

CUresult cuGetProcAddress(const char* symbol,
                          void** pfn,
                          int cudaVersion,
                          uint64_t flags,
                          void* symbolStatus) {
    init_once();
    refresh_driver_symbols();
    if (!real_cuGetProcAddress) {
        return 1;
    }
    CUresult ret = real_cuGetProcAddress(symbol, pfn, cudaVersion, flags, symbolStatus);
    if (ret == CU_SUCCESS && pfn && symbol) {
        if (strcmp(symbol, "cuLaunchKernel") == 0) {
            if (!real_cuLaunchKernel && *pfn) {
                real_cuLaunchKernel = (cuLaunchKernel_t)(*pfn);
            }
            *pfn = (void*)cuLaunchKernel;
        } else if (strcmp(symbol, "cuLaunchKernelEx") == 0) {
            if (!real_cuLaunchKernelEx && *pfn) {
                real_cuLaunchKernelEx = (cuLaunchKernelEx_t)(*pfn);
            }
            *pfn = (void*)cuLaunchKernelEx;
        }
    }
    return ret;
}

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
