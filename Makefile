CC = icc
CXXFLAGS  = -O3 -restrict -qopenmp -lpthread


INCLUDES = -I  /home/shuzhan1/intel/mkl/include
LIBS = -L /home/shuzhan1/intel/mkl/lib/intel64 -lmkl_rt 

OUTFILE = sru

SOURCES = $(wildcard *.c) 
OBJS = $(patsubst %.c,%.o,$(SOURCES))

all: $(OUTFILE)

$(OUTFILE):$(SOURCES)
	$(CC) $^ -o $@ $(CXXFLAGS) $(INCLUDES) $(LIBS)



.PHONY:clean
clean:
	rm -rf $(OBJS) $(OUTFILE)
