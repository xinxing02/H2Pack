H2PACK_INSTALL_DIR = ..

DEFS    = 
INCS    = -I$(H2PACK_INSTALL_DIR)/include
CFLAGS  = $(INCS) -Wall -g -std=gnu11 -O3 -fPIC $(DEFS)
LDFLAGS = -g -O3 -fopenmp
LIBS    = $(H2PACK_INSTALL_DIR)/lib/libH2Pack.a

ifeq ($(shell $(CC) --version 2>&1 | grep -c "icc"), 1)
CFLAGS  += -fopenmp -xHost
else ifeq ($(shell $(CC) --version 2>&1 | grep -c "Homebrew GCC"), 1)
# Homebrew GCC support for macOS (including Apple Silicon)
    CFLAGS += -fopenmp
    # Optimize for Apple Silicon if detected
    ifeq ($(shell uname -m), arm64)
        CFLAGS += -mcpu=apple-m1
    else
        CFLAGS += -march=native
    endif
    CFLAGS += -Wno-unused-result -Wno-unused-function
    LIBS   += -lgfortran -lm
else ifeq ($(shell $(CC) --version 2>&1 | grep -c "gcc"), 1)
# Generic GCC (including Linux GCC)
    CFLAGS  += -fopenmp -march=native -Wno-unused-result -Wno-unused-function
    LIBS    += -lgfortran -lm
endif

ifeq ($(strip $(USE_MKL)), 1)
DEFS    += -DUSE_MKL
CFLAGS  += -mkl
LDFLAGS += -mkl
endif

ifeq ($(strip $(USE_OPENBLAS)), 1)
# For Homebrew on macOS: /opt/homebrew/opt/openblas
# For Linux: ../../OpenBLAS-git/install or custom path
OPENBLAS_INSTALL_DIR = /opt/homebrew/opt/openblas
DEFS    += -DUSE_OPENBLAS
INCS    += -I$(OPENBLAS_INSTALL_DIR)/include
LDFLAGS += -L$(OPENBLAS_INSTALL_DIR)/lib
LIBS    += -lopenblas
endif

C_SRCS 	= $(wildcard *.c)
C_OBJS  = $(C_SRCS:.c=.c.o)
EXES    = $(C_SRCS:.c=.exe)

# Delete the default old-fashion double-suffix rules
.SUFFIXES:

.SECONDARY: $(C_OBJS)

all: $(EXES)

%.c.o: %.c
	$(CC) $(CFLAGS) -c $^ -o $@

%.exe: %.c.o
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)

clean:
	rm -f $(EXES) $(C_OBJS)