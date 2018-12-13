CC = g++ # define the C compiler to use
CFLAGS = -std=c++11 -O3 -Wall -Wextra -pedantic -DNDEBUG 
# define any directories containing header files other than /usr/include
INCLUDES = -I/home/fran/drive/projects/DNest4/code -I/home/fran/drive/projects/VBBinaryLensing/VBBinaryLensing/lib
# define library paths in addition to /usr/lib
LFLAGS = -L/home/fran/drive/projects/DNest4/code
# define any libraries to link into executable:
LIBS = -ldnest4 -lpthread -lVBB
# define the C++ source files
SRCS = main.cpp Data.cpp ESPL.cpp 
# define the C object files 
# This uses Suffix Replacement within a macro:
#   $(name:string1=string2)
#         For each word in 'name' replace 'string1' with 'string2'
OBJS = $(SRCS:.c=.o)
# define the executable file 
MAIN = main

.PHONY: clean

all:    $(MAIN)
	@echo  Successfully compiled.

$(MAIN): $(OBJS) 
	$(CC) $(CFLAGS) $(INCLUDES) -o $(MAIN) $(OBJS) $(LFLAGS) $(LIBS)

# this is a suffix replacement rule for building .o's from .c's
# it uses automatic variables $<: the name of the prerequisite of
# the rule(a .c file) and $@: the name of the target of the rule (a .o file) 
# (see the gnu make manual section about automatic variables)
.c.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c $<  -o $@
clean:
	$(RM) *.o *~ $(MAIN)
