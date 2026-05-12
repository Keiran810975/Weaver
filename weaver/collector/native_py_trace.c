#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <frameobject.h>

#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#define CODE_MAP_SIZE 8192U
#define ACTIVE_MAP_SIZE 8192U
#define DEFAULT_QUEUE_SIZE 65536U
#define CODE_MAP_MISS -32768

typedef struct {
    char* spec;
    char* leaf;
    int seen;
    uint64_t calls;
} TargetSpec;

typedef struct {
    uintptr_t code;
    int target;
} CodeSlot;

typedef struct {
    uintptr_t frame;
    uint64_t start_ns;
    int target;
} ActiveSlot;

typedef struct {
    uint64_t start_ns;
    uint64_t end_ns;
    uint64_t tid;
    uint64_t dropped_snapshot;
    int pid;
    int target;
} TraceEvent;

static TargetSpec* g_targets = NULL;
static int g_target_count = 0;
static CodeSlot g_code_map[CODE_MAP_SIZE];
static ActiveSlot g_active_map[ACTIVE_MAP_SIZE];

static TraceEvent* g_queue = NULL;
static size_t g_queue_cap = 0;
static size_t g_queue_head = 0;
static size_t g_queue_tail = 0;
static size_t g_queue_count = 0;
static uint64_t g_dropped = 0;
static pthread_mutex_t g_queue_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_queue_cond = PTHREAD_COND_INITIALIZER;
static pthread_t g_flush_thread;
static int g_flush_started = 0;
static int g_stop_flush = 0;
static int g_running = 0;
static unsigned long g_sample_rate = 1;

static int g_sock = -1;
static struct sockaddr_un g_addr;

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static uint64_t thread_id_u64(void) {
    return (uint64_t)(uintptr_t)pthread_self();
}

static size_t hash_ptr(uintptr_t value, size_t size) {
    value >>= 4;
    value ^= value >> 33;
    value *= 0xff51afd7ed558ccdULL;
    value ^= value >> 33;
    return (size_t)(value & (size - 1U));
}

static char* dup_normalized(const char* in) {
    size_t len = strlen(in);
    char* out = (char*)malloc(len + 1U);
    if (!out) {
        return NULL;
    }
    for (size_t i = 0; i < len; i++) {
        out[i] = in[i] == '@' ? '.' : in[i];
    }
    out[len] = '\0';
    return out;
}

static const char* leaf_name(const char* spec) {
    const char* dot = strrchr(spec, '.');
    return dot ? dot + 1 : spec;
}

static int suffix_component_match(const char* text, const char* suffix) {
    size_t text_len = strlen(text);
    size_t suffix_len = strlen(suffix);
    if (text_len == suffix_len) {
        return strcmp(text, suffix) == 0;
    }
    if (text_len > suffix_len && text[text_len - suffix_len - 1U] == '.') {
        return strcmp(text + text_len - suffix_len, suffix) == 0;
    }
    return 0;
}

static int target_matches_strings(int target, const char* name, const char* qualname) {
    TargetSpec* spec = &g_targets[target];
    if (name && (strcmp(name, spec->spec) == 0 || strcmp(name, spec->leaf) == 0)) {
        return 1;
    }
    if (qualname) {
        if (suffix_component_match(qualname, spec->spec)) {
            return 1;
        }
        if (suffix_component_match(spec->spec, qualname)) {
            return 1;
        }
    }
    return 0;
}

static int code_map_lookup(uintptr_t code) {
    if (!code) {
        return -1;
    }
    size_t idx = hash_ptr(code, CODE_MAP_SIZE);
    for (size_t i = 0; i < CODE_MAP_SIZE; i++) {
        CodeSlot* slot = &g_code_map[(idx + i) & (CODE_MAP_SIZE - 1U)];
        if (slot->code == code) {
            return slot->target;
        }
        if (slot->code == 0) {
            return CODE_MAP_MISS;
        }
    }
    return CODE_MAP_MISS;
}

static void code_map_insert(uintptr_t code, int target) {
    if (!code || target < -1) {
        return;
    }
    size_t idx = hash_ptr(code, CODE_MAP_SIZE);
    for (size_t i = 0; i < CODE_MAP_SIZE; i++) {
        CodeSlot* slot = &g_code_map[(idx + i) & (CODE_MAP_SIZE - 1U)];
        if (slot->code == 0 || slot->code == code) {
            slot->code = code;
            slot->target = target;
            return;
        }
    }
}

static const char* unicode_attr_utf8(PyObject* obj, const char* attr, PyObject** holder) {
    *holder = PyObject_GetAttrString(obj, attr);
    if (!*holder) {
        PyErr_Clear();
        return NULL;
    }
    const char* text = PyUnicode_AsUTF8(*holder);
    if (!text) {
        PyErr_Clear();
        Py_DECREF(*holder);
        *holder = NULL;
        return NULL;
    }
    return text;
}

static int resolve_target_for_code(PyFrameObject* frame, PyCodeObject* code, uintptr_t code_key) {
    PyObject* name_obj = NULL;
    PyObject* qualname_obj = NULL;
    const char* name = unicode_attr_utf8((PyObject*)code, "co_name", &name_obj);
    const char* qualname = unicode_attr_utf8((PyObject*)code, "co_qualname", &qualname_obj);
    (void)frame;

    int matched = -1;
    for (int i = 0; i < g_target_count; i++) {
        if (target_matches_strings(i, name, qualname)) {
            matched = i;
            if (!g_targets[i].seen) {
                g_targets[i].seen = 1;
            }
            code_map_insert(code_key, i);
            break;
        }
    }
    if (matched < 0) {
        code_map_insert(code_key, -1);
    }

    Py_XDECREF(name_obj);
    Py_XDECREF(qualname_obj);
    return matched;
}

static int active_lookup(uintptr_t frame, uint64_t* start_ns, int* target) {
    size_t idx = hash_ptr(frame, ACTIVE_MAP_SIZE);
    for (size_t i = 0; i < ACTIVE_MAP_SIZE; i++) {
        ActiveSlot* slot = &g_active_map[(idx + i) & (ACTIVE_MAP_SIZE - 1U)];
        if (slot->frame == frame) {
            *start_ns = slot->start_ns;
            *target = slot->target;
            slot->frame = 0;
            return 1;
        }
    }
    return 0;
}

static void active_insert(uintptr_t frame, uint64_t start_ns, int target) {
    size_t idx = hash_ptr(frame, ACTIVE_MAP_SIZE);
    for (size_t i = 0; i < ACTIVE_MAP_SIZE; i++) {
        ActiveSlot* slot = &g_active_map[(idx + i) & (ACTIVE_MAP_SIZE - 1U)];
        if (slot->frame == 0 || slot->frame == frame) {
            slot->frame = frame;
            slot->start_ns = start_ns;
            slot->target = target;
            return;
        }
    }
}

static void queue_event(const TraceEvent* event) {
    pthread_mutex_lock(&g_queue_lock);
    if (!g_queue || g_queue_count == g_queue_cap) {
        g_dropped++;
        pthread_mutex_unlock(&g_queue_lock);
        return;
    }
    g_queue[g_queue_tail] = *event;
    g_queue_tail = (g_queue_tail + 1U) % g_queue_cap;
    g_queue_count++;
    pthread_cond_signal(&g_queue_cond);
    pthread_mutex_unlock(&g_queue_lock);
}

static void json_escape(const char* in, char* out, size_t out_size) {
    if (out_size == 0) {
        return;
    }
    size_t j = 0;
    for (size_t i = 0; in && in[i] && j + 2U < out_size; i++) {
        unsigned char c = (unsigned char)in[i];
        if (c == '"' || c == '\\') {
            if (j + 2U >= out_size) {
                break;
            }
            out[j++] = '\\';
            out[j++] = (char)c;
        } else if (c >= 0x20 && c < 0x7f) {
            out[j++] = (char)c;
        }
    }
    out[j] = '\0';
}

static void emit_event(const TraceEvent* event) {
    if (g_sock < 0 || event->target < 0 || event->target >= g_target_count) {
        return;
    }
    const char* name = g_targets[event->target].spec;
    char escaped[512];
    json_escape(name, escaped, sizeof(escaped));
    char buf[2048];
    uint64_t dur = event->end_ns >= event->start_ns ? event->end_ns - event->start_ns : 0;
    int n = snprintf(
        buf, sizeof(buf),
        "{\"ts_ns\":%llu,\"pid\":%d,\"tid\":%llu,\"layer\":\"python\",\"kind\":\"operator\",\"source\":\"weaver_native_py_trace\",\"operator_name\":\"%s\",\"ts_end_ns\":%llu,\"dur_ns\":%llu,\"payload\":{\"operator_name\":\"%s\",\"start_ns\":%llu,\"end_ns\":%llu,\"dur_ns\":%llu,\"collector\":\"native_cprofile\",\"dropped_events\":%llu}}",
        (unsigned long long)event->start_ns, event->pid,
        (unsigned long long)event->tid, escaped,
        (unsigned long long)event->end_ns, (unsigned long long)dur,
        escaped, (unsigned long long)event->start_ns,
        (unsigned long long)event->end_ns, (unsigned long long)dur,
        (unsigned long long)event->dropped_snapshot);
    if (n > 0) {
        if (n >= (int)sizeof(buf)) {
            n = (int)sizeof(buf) - 1;
        }
        sendto(g_sock, buf, (size_t)n, 0, (struct sockaddr*)&g_addr, sizeof(g_addr));
    }
}

static void* flush_loop(void* unused) {
    (void)unused;
    while (1) {
        pthread_mutex_lock(&g_queue_lock);
        while (g_queue_count == 0 && !g_stop_flush) {
            pthread_cond_wait(&g_queue_cond, &g_queue_lock);
        }
        if (g_queue_count == 0 && g_stop_flush) {
            pthread_mutex_unlock(&g_queue_lock);
            break;
        }
        TraceEvent event = g_queue[g_queue_head];
        g_queue_head = (g_queue_head + 1U) % g_queue_cap;
        g_queue_count--;
        pthread_mutex_unlock(&g_queue_lock);
        emit_event(&event);
    }
    return NULL;
}

static int profiler(PyObject* obj, PyFrameObject* frame, int what, PyObject* arg) {
    (void)obj;
    (void)arg;
    if (!g_running || !frame) {
        return 0;
    }

    uintptr_t frame_key = (uintptr_t)frame;
    if (what == PyTrace_RETURN || what == PyTrace_EXCEPTION) {
        uint64_t start = 0;
        int target = -1;
        if (!active_lookup(frame_key, &start, &target)) {
            return 0;
        }
        TraceEvent event;
        event.start_ns = start;
        event.end_ns = now_ns();
        event.tid = thread_id_u64();
        event.pid = getpid();
        event.target = target;
        event.dropped_snapshot = g_dropped;
        queue_event(&event);
        return 0;
    }
    if (what != PyTrace_CALL) {
        return 0;
    }

    PyCodeObject* code = PyFrame_GetCode(frame);
    if (!code) {
        PyErr_Clear();
        return 0;
    }
    uintptr_t code_key = (uintptr_t)code;
    int target = code_map_lookup(code_key);
    if (target == CODE_MAP_MISS) {
        target = resolve_target_for_code(frame, code, code_key);
    }
    Py_DECREF(code);
    if (target < 0) {
        return 0;
    }

    TargetSpec* spec = &g_targets[target];
    spec->calls++;
    if ((spec->calls % g_sample_rate) != 0) {
        return 0;
    }
    active_insert(frame_key, now_ns(), target);
    return 0;
}

static void set_profile_for_threads(Py_tracefunc tracefunc) {
    PyThreadState* current = PyThreadState_Get();
    if (!current) {
        return;
    }
    PyThreadState* tstate = current;
    int count = 0;
    while (tstate && count < 256) {
        PyThreadState_Swap(tstate);
        PyEval_SetProfile(tracefunc, NULL);
        tstate = PyThreadState_Next(tstate);
        count++;
    }
    PyThreadState_Swap(current);
}

static void free_targets(void) {
    if (!g_targets) {
        return;
    }
    for (int i = 0; i < g_target_count; i++) {
        free(g_targets[i].spec);
        free(g_targets[i].leaf);
    }
    free(g_targets);
    g_targets = NULL;
    g_target_count = 0;
}

static int init_targets(PyObject* targets) {
    PyObject* seq = PySequence_Fast(targets, "targets must be a sequence of strings");
    if (!seq) {
        return -1;
    }
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    if (n <= 0) {
        Py_DECREF(seq);
        PyErr_SetString(PyExc_ValueError, "native Weaver Python tracing requires at least one target function");
        return -1;
    }
    g_targets = (TargetSpec*)calloc((size_t)n, sizeof(TargetSpec));
    if (!g_targets) {
        Py_DECREF(seq);
        PyErr_NoMemory();
        return -1;
    }
    g_target_count = (int)n;
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject* item = PySequence_Fast_GET_ITEM(seq, i);
        const char* text = PyUnicode_AsUTF8(item);
        if (!text || !text[0]) {
            Py_DECREF(seq);
            PyErr_SetString(PyExc_ValueError, "targets must be non-empty strings");
            return -1;
        }
        g_targets[i].spec = dup_normalized(text);
        if (!g_targets[i].spec) {
            Py_DECREF(seq);
            PyErr_NoMemory();
            return -1;
        }
        g_targets[i].leaf = strdup(leaf_name(g_targets[i].spec));
        if (!g_targets[i].leaf) {
            Py_DECREF(seq);
            PyErr_NoMemory();
            return -1;
        }
    }
    Py_DECREF(seq);
    return 0;
}

static void reset_state(void) {
    memset(g_code_map, 0, sizeof(g_code_map));
    memset(g_active_map, 0, sizeof(g_active_map));
    g_queue_head = 0;
    g_queue_tail = 0;
    g_queue_count = 0;
    g_dropped = 0;
}

static PyObject* native_start(PyObject* self, PyObject* args, PyObject* kwargs) {
    (void)self;
    const char* sock_path = "/tmp/weaver.sock";
    PyObject* targets = NULL;
    long sample_rate = 1;
    unsigned long queue_size = DEFAULT_QUEUE_SIZE;
    static char* kwlist[] = {"sock_path", "targets", "sample_rate", "queue_size", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sO|lk", kwlist,
                                     &sock_path, &targets, &sample_rate, &queue_size)) {
        return NULL;
    }
    if (sample_rate < 1) {
        sample_rate = 1;
    }
    g_sample_rate = (unsigned long)sample_rate;
    if (queue_size < 1024U) {
        queue_size = 1024U;
    }

    if (g_running) {
        Py_RETURN_NONE;
    }

    reset_state();
    free_targets();
    if (init_targets(targets) != 0) {
        free_targets();
        return NULL;
    }

    g_queue = (TraceEvent*)calloc(queue_size, sizeof(TraceEvent));
    if (!g_queue) {
        free_targets();
        PyErr_NoMemory();
        return NULL;
    }
    g_queue_cap = queue_size;

    g_sock = socket(AF_UNIX, SOCK_DGRAM, 0);
    if (g_sock >= 0) {
        memset(&g_addr, 0, sizeof(g_addr));
        g_addr.sun_family = AF_UNIX;
        strncpy(g_addr.sun_path, sock_path, sizeof(g_addr.sun_path) - 1U);
    }

    g_stop_flush = 0;
    if (pthread_create(&g_flush_thread, NULL, flush_loop, NULL) == 0) {
        g_flush_started = 1;
    } else {
        g_flush_started = 0;
    }

    g_running = 1;
    set_profile_for_threads(profiler);
    Py_RETURN_NONE;
}

static PyObject* native_stop(PyObject* self, PyObject* Py_UNUSED(ignored)) {
    (void)self;
    if (!g_running) {
        Py_RETURN_NONE;
    }
    set_profile_for_threads(NULL);
    g_running = 0;

    pthread_mutex_lock(&g_queue_lock);
    g_stop_flush = 1;
    pthread_cond_broadcast(&g_queue_cond);
    pthread_mutex_unlock(&g_queue_lock);
    if (g_flush_started) {
        pthread_join(g_flush_thread, NULL);
        g_flush_started = 0;
    }
    if (g_sock >= 0) {
        close(g_sock);
        g_sock = -1;
    }
    free(g_queue);
    g_queue = NULL;
    g_queue_cap = 0;
    free_targets();
    reset_state();
    Py_RETURN_NONE;
}

static PyObject* native_stats(PyObject* self, PyObject* Py_UNUSED(ignored)) {
    (void)self;
    return Py_BuildValue("{s:i,s:K,s:n}",
                         "running", g_running,
                         "dropped", (unsigned long long)g_dropped,
                         "queued", (Py_ssize_t)g_queue_count);
}

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcast-function-type-mismatch"
#endif
static PyMethodDef Methods[] = {
    {"start", (PyCFunction)native_start, METH_VARARGS | METH_KEYWORDS, "Start native low-overhead Python tracing."},
    {"stop", native_stop, METH_NOARGS, "Stop native Python tracing."},
    {"stats", native_stats, METH_NOARGS, "Return native tracing stats."},
    {NULL, NULL, 0, NULL},
};
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

static struct PyModuleDef Module = {
    PyModuleDef_HEAD_INIT,
    "_native_py_trace",
    "Native Weaver Python tracing based on PyEval_SetProfile.",
    -1,
    Methods,
    NULL,
    NULL,
    NULL,
    NULL,
};

PyMODINIT_FUNC PyInit__native_py_trace(void) {
    return PyModule_Create(&Module);
}
