#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

// Minimal type aliases to avoid strict toolkit header dependency.
typedef int CUresult;
typedef void* CUfunction;
typedef void* CUmodule;
typedef void* CUstream;

typedef int ncclResult_t;
typedef int ncclDataType_t;
typedef int ncclRedOp_t;
typedef void* ncclComm_t;

typedef CUresult (*cuModuleGetFunction_t)(CUfunction*, CUmodule, const char*);
typedef CUresult (*cuLaunchKernel_t)(CUfunction, unsigned int, unsigned int,
                                     unsigned int, unsigned int, unsigned int,
                                     unsigned int, unsigned int, CUstream,
                                     void**, void**);

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

static cuModuleGetFunction_t real_cuModuleGetFunction = NULL;
static cuLaunchKernel_t real_cuLaunchKernel = NULL;

static ncclAllReduce_t real_ncclAllReduce = NULL;
static ncclAllGather_t real_ncclAllGather = NULL;
static ncclReduceScatter_t real_ncclReduceScatter = NULL;
static ncclBroadcast_t real_ncclBroadcast = NULL;

static int g_sock = -1;
static struct sockaddr_un g_addr;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static int g_initialized = 0;

struct fn_name_item {
    CUfunction func;
    char* name;
    struct fn_name_item* next;
};

static struct fn_name_item* g_fn_map = NULL;

static long long now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (long long)ts.tv_sec * 1000000000LL + (long long)ts.tv_nsec;
}

static void map_set(CUfunction func, const char* name) {
    if (!func || !name) {
        return;
    }
    pthread_mutex_lock(&g_lock);
    struct fn_name_item* cur = g_fn_map;
    while (cur) {
        if (cur->func == func) {
            free(cur->name);
            cur->name = strdup(name);
            pthread_mutex_unlock(&g_lock);
            return;
        }
        cur = cur->next;
    }
    struct fn_name_item* n = (struct fn_name_item*)calloc(1, sizeof(*n));
    n->func = func;
    n->name = strdup(name);
    n->next = g_fn_map;
    g_fn_map = n;
    pthread_mutex_unlock(&g_lock);
}

static const char* map_get(CUfunction func) {
    pthread_mutex_lock(&g_lock);
    struct fn_name_item* cur = g_fn_map;
    while (cur) {
        if (cur->func == func) {
            const char* ret = cur->name;
            pthread_mutex_unlock(&g_lock);
            return ret;
        }
        cur = cur->next;
    }
    pthread_mutex_unlock(&g_lock);
    return "<unknown>";
}

static void init_once() {
    if (g_initialized) {
        return;
    }
    g_initialized = 1;

    real_cuModuleGetFunction = (cuModuleGetFunction_t)dlsym(RTLD_NEXT, "cuModuleGetFunction");
    real_cuLaunchKernel = (cuLaunchKernel_t)dlsym(RTLD_NEXT, "cuLaunchKernel");

    real_ncclAllReduce = (ncclAllReduce_t)dlsym(RTLD_NEXT, "ncclAllReduce");
    real_ncclAllGather = (ncclAllGather_t)dlsym(RTLD_NEXT, "ncclAllGather");
    real_ncclReduceScatter = (ncclReduceScatter_t)dlsym(RTLD_NEXT, "ncclReduceScatter");
    real_ncclBroadcast = (ncclBroadcast_t)dlsym(RTLD_NEXT, "ncclBroadcast");

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
}

static void send_json(const char* fmt, ...) {
    if (g_sock < 0) {
        return;
    }
    char buf[2048];
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
    sendto(g_sock, buf, (size_t)n, 0, (struct sockaddr*)&g_addr, sizeof(g_addr));
}

__attribute__((constructor)) static void weaver_init() {
    init_once();
    send_json(
        "{\"ts_ns\":%lld,\"pid\":%d,\"tid\":%llu,\"layer\":\"hook\",\"kind\":\"init\",\"payload\":{\"status\":\"ok\"}}",
        now_ns(), getpid(), (unsigned long long)pthread_self());
}

__attribute__((destructor)) static void weaver_fini() {
    if (g_sock >= 0) {
        close(g_sock);
        g_sock = -1;
    }
}

CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) {
    init_once();
    if (!real_cuModuleGetFunction) {
        return 1;
    }
    CUresult ret = real_cuModuleGetFunction(hfunc, hmod, name);
    if (ret == 0 && hfunc && *hfunc && name) {
        map_set(*hfunc, name);
    }
    send_json(
        "{\"ts_ns\":%lld,\"pid\":%d,\"tid\":%llu,\"layer\":\"cuda\",\"kind\":\"module_get_function\",\"payload\":{\"ret\":%d,\"name\":\"%s\",\"func\":\"%p\"}}",
        now_ns(), getpid(), (unsigned long long)pthread_self(), ret,
        name ? name : "", hfunc ? (void*)(*hfunc) : NULL);
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
    if (!real_cuLaunchKernel) {
        return 1;
    }

    unsigned long long block_size = (unsigned long long)blockDimX * (unsigned long long)blockDimY * (unsigned long long)blockDimZ;
    unsigned long long grid_size = (unsigned long long)gridDimX * (unsigned long long)gridDimY * (unsigned long long)gridDimZ;
    unsigned long long warps_per_block = (block_size + 31ULL) / 32ULL;
    unsigned long long total_warps = warps_per_block * grid_size;
    const char* name = map_get(f);

    long long start = now_ns();
    CUresult ret = real_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ,
                                       blockDimX, blockDimY, blockDimZ,
                                       sharedMemBytes, hStream,
                                       kernelParams, extra);
    long long end = now_ns();

    send_json(
        "{\"ts_ns\":%lld,\"pid\":%d,\"tid\":%llu,\"layer\":\"cuda\",\"kind\":\"launch\",\"payload\":{\"ret\":%d,\"kernel\":\"%s\",\"func\":\"%p\",\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],\"shared_mem\":%u,\"warps_per_block\":%llu,\"total_warps\":%llu,\"warp_scope\":\"estimated\",\"start_ns\":%lld,\"end_ns\":%lld,\"dur_ns\":%lld}}",
        end, getpid(), (unsigned long long)pthread_self(), ret,
        name ? name : "", f,
        gridDimX, gridDimY, gridDimZ,
        blockDimX, blockDimY, blockDimZ,
        sharedMemBytes,
        warps_per_block, total_warps,
        start, end, (end - start));

    return ret;
}

static void send_nccl_event(const char* kind, size_t count, int ret) {
    send_json(
        "{\"ts_ns\":%lld,\"pid\":%d,\"tid\":%llu,\"layer\":\"cuda\",\"kind\":\"%s\",\"payload\":{\"ret\":%d,\"count\":%zu,\"component\":\"nccl\"}}",
        now_ns(), getpid(), (unsigned long long)pthread_self(), kind, ret, count);
}

ncclResult_t ncclAllReduce(const void* sendbuff,
                           void* recvbuff,
                           size_t count,
                           ncclDataType_t datatype,
                           ncclRedOp_t op,
                           ncclComm_t comm,
                           CUstream stream) {
    init_once();
    if (!real_ncclAllReduce) {
        return -1;
    }
    ncclResult_t ret = real_ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
    send_nccl_event("nccl_all_reduce", count, ret);
    return ret;
}

ncclResult_t ncclAllGather(const void* sendbuff,
                           void* recvbuff,
                           size_t sendcount,
                           ncclDataType_t datatype,
                           ncclComm_t comm,
                           CUstream stream) {
    init_once();
    if (!real_ncclAllGather) {
        return -1;
    }
    ncclResult_t ret = real_ncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream);
    send_nccl_event("nccl_all_gather", sendcount, ret);
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
    if (!real_ncclReduceScatter) {
        return -1;
    }
    ncclResult_t ret = real_ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
    send_nccl_event("nccl_reduce_scatter", recvcount, ret);
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
    if (!real_ncclBroadcast) {
        return -1;
    }
    ncclResult_t ret = real_ncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream);
    send_nccl_event("nccl_broadcast", count, ret);
    return ret;
}
