CC = icc
CXXFLAGS  = -Wall -O2 -g

INCLUDES = -I  /opt/intel/mkl/include
LIBS = -L /opt/intel/mkl/lib/intel64 -lmkl_rt
OUTFILE = sru

SOURCES = $(wildcard *.c) 
OBJS = $(patsubst %.c,%.o,$(SOURCES))

all: $(OUTFILE)

$(OUTFILE):$(SOURCES)
	$(CC) $^ -o $@ $(CXXFLAGS) $(INCLUDES) $(LIBS)



.PHONY:clean
clean:
	rm -rf $(OBJS) $(OUTFILE)
