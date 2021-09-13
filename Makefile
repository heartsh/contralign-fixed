######################################################################
# Makefile
######################################################################

######################################################################
# Alignment model type
#
#   0 = protein
#   1 = RNA
#
# Please make sure you enable one of these options before compiling!
######################################################################

MODEL_TYPE = -DRNA=1

######################################################################
# Compilation flags
######################################################################

CXX = g++

CXX_FLAGS = -Wall -Wundef -O3 -DNDEBUG -fomit-frame-pointer \
	-ffast-math -funroll-all-loops -funsafe-math-optimizations \
	-fpeel-loops -Winline --param large-function-growth=100000 \
	--param max-inline-insns-single=100000 \
	--param inline-unit-growth=100000 -fpermissive

OTHER_FLAGS = 

LINK_FLAGS = -lm

######################################################################
# Compilation rules
######################################################################

ifndef MODEL_TYPE
$(error MODEL_TYPE variable (which determines whether the program \
is being compiled for protein or RNA sequence input) not \
defined!  Please edit the Makefile and follow the instructions at \
the top.)
endif

CONTRALIGN_SRCS = \
	Contralign.cpp \
	FileDescription.cpp \
	Options.cpp \
	MultiSequence.cpp \
	Sequence.cpp \
	Utilities.cpp

CONTRALIGN_OBJS = $(CONTRALIGN_SRCS:%.cpp=%.o)

.PHONY: all clean

all: contralign

%.o: %.cpp *.hpp *.ipp Makefile *.params.* Defaults.ipp
	$(CXX) $(MODEL_TYPE) $(CXX_FLAGS) $(OTHER_FLAGS) -c $<

Defaults.ipp: MakeDefaults.pl *.params.*
	perl MakeDefaults.pl contralign.params.protein contralign.params.rna

contralign: $(CONTRALIGN_OBJS)
	$(CXX) $(MODEL_TYPE) $(CXX_FLAGS) $(OTHER_FLAGS) $(CONTRALIGN_OBJS) $(LINK_FLAGS) -o contralign

clean:
	rm -f contralign *.o Defaults.ipp

######################################################################
# Machine-specific rules
######################################################################

native:
	make all OTHER_FLAGS="-mtune=native"

# default

profile:
	make all OTHER_FLAGS="-pg -g"

multi:
	make all CXX="mpiCC" OTHER_FLAGS="-DMULTI"

# debugging

debug:
	make all CXX_FLAGS="-g -fno-inline -W -Wall"

debugmulti:
	make all CXX="mpiCC" OTHER_FLAGS="-DMULTI" CXX_FLAGS="-g -fno-inline -W -Wall"

assembly:
	make all OTHER_FLAGS="-Wa,-a,-ad"

# pentium 4

gccp4:
	make all OTHER_FLAGS="-march=pentium4 -mtune=pentium4"

gccp4profile:
	make all OTHER_FLAGS="-march=pentium4 -mtune=pentium4 -pg -g"

gccp4multi:
	make all CXX="mpiCC" OTHER_FLAGS="-DMULTI -march=pentium4 -mtune=pentium4"

# athlon64

defaultgccathlon64:
	make all OTHER_FLAGS="-march=athlon64 -mtune=athlon64"

gccathlon64profile:
	make all OTHER_FLAGS="-march=athlon64 -mtune=athlon64 -pg -g"

gccathlon64multi:
	make all CXX="mpiCC" OTHER_FLAGS="-DMULTI -march=athlon64 -mtune=athlon64"

# intel

intel:
	make all CXX="icpc" OTHER_FLAGS="-xN -no-ipo -static"

intelmulti:
	make all LAMHCP="icpc" CXX="mpiCC" OTHER_FLAGS="-DMULTI -xN -no-ipo"
