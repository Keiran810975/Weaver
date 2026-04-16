CC ?= gcc
CFLAGS ?= -O2 -fPIC -Wall -Wextra
LDFLAGS ?= -shared -ldl -lpthread
TARGET = libweaver_hook.so
SRC = weaver_hook.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)
