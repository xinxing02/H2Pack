LIB_A   = libH2Pack.a
LIB_SO  = libH2Pack.so

C_SRCS  = $(wildcard *.c)
C_OBJS  = $(C_SRCS:.c=.c.o)

AR      = ar rcs
DEFS    = 
INCS    = 
CFLAGS  = $(INCS) -Wall -g -std=gnu11 -O3 -fPIC $(DEFS)

ifeq ($(shell $(CC) --version 2>&1 | grep -c "icc"), 1)
CFLAGS += -qopenmp -xHost
endif

ifeq ($(shell $(CC) --version 2>&1 | grep -c "gcc"), 1)
CFLAGS += -fopenmp -march=native -Wno-unused-result -Wno-unused-function
endif

# Apple Clang support for macOS (including Apple Silicon)
ifeq ($(shell $(CC) --version 2>&1 | grep -c "Apple clang"), 1)
CFLAGS += -Xpreprocessor -fopenmp -march=native -Wno-unused-result -Wno-unused-function
INCS   += -I/opt/homebrew/opt/libomp/include
LDFLAGS += -L/opt/homebrew/opt/libomp/lib -lomp
endif

ifeq ($(strip $(USE_MKL)), 1)
DEFS   += -DUSE_MKL
CFLAGS += -mkl
endif

# If you use OpenBLAS, modify OPENBLAS_INSTALL_DIR here
OPENBLAS_INSTALL_DIR = /opt/homebrew/opt/openblas
ifeq ($(strip $(USE_OPENBLAS)), 1)
DEFS    += -DUSE_OPENBLAS
INCS    += -I$(OPENBLAS_INSTALL_DIR)/include
LDFLAGS += -L$(OPENBLAS_INSTALL_DIR)/lib -lopenblas
endif

# Delete the default old-fashion double-suffix rules
.SUFFIXES:

.SECONDARY: $(C_OBJS)

all: install

install: $(LIB_A) $(LIB_SO)
	mkdir -p ../lib
	mkdir -p ../include
	mkdir -p ../include/ASTER/include
	cp -u $(LIB_A)  ../lib/$(LIB_A)
	cp -u $(LIB_SO) ../lib/$(LIB_SO)
	cp -u *.h ../include/
	cp -u ASTER/include/*.h ../include/ASTER/include

$(LIB_A): $(C_OBJS) 
	$(AR) $@ $^

$(LIB_SO): $(C_OBJS)
	$(CC) -shared -o $@ $^ $(LDFLAGS)

%.c.o: %.c
	$(CC) $(CFLAGS) -c $^ -o $@

clean:
	rm -f $(C_OBJS) $(LIB_A) $(LIB_SO)
