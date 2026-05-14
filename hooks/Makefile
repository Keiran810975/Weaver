CC ?= gcc
CFLAGS ?= -O2 -fPIC -Wall -Wextra
LDFLAGS ?= -shared -ldl -lpthread
TARGET = libweaver_hook.so
SRC = weaver_hook.c

.PHONY: all clean preflight

all: preflight $(TARGET)

preflight:
	@if [ -f "$(TARGET)" ]; then \
		os=$$(uname -s); \
		desc=$$(file -b "$(TARGET)" 2>/dev/null || true); \
		case "$$os:$$desc" in \
			Linux*:ELF*) ;; \
			Darwin*:Mach-O*) ;; \
			*) echo "Removing incompatible $(TARGET): $$desc"; rm -f "$(TARGET)" ;; \
		esac; \
	fi

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)
