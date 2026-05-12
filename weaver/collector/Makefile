PYTHON ?= python3
CC ?= gcc

EXT_SUFFIX := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')")
PY_INCLUDE := $(shell $(PYTHON) -c "import sysconfig; print('-I' + sysconfig.get_path('include'))")
DARWIN_LDFLAGS := $(shell $(PYTHON) -c "import sys; print('-undefined dynamic_lookup' if sys.platform == 'darwin' else '')")

CFLAGS ?= -O3 -fPIC -Wall -Wextra
LDFLAGS ?= -shared

TARGET = _native_py_trace$(EXT_SUFFIX)
SRC = native_py_trace.c

.PHONY: all native clean

all: native
native: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(PY_INCLUDE) -o $@ $^ $(LDFLAGS) $(DARWIN_LDFLAGS) -lpthread

clean:
	rm -f _native_py_trace*.so _native_py_trace*.dylib
