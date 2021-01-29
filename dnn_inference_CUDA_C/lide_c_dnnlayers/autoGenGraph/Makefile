mlpsingle: lide_c_mlpsingle_graph.c
	g++ -g -o lide_c_mlpsingle_graph.o  -c lide_c_mlpsingle_graph.c \
	-I. -I../actor -I../includes -I../../cnnnsight \

libr: lide_c_mlpsingle_graph.o
	ar rcs libgraph.a lide_c_mlpsingle_graph.o

all: mlpsingle libr

.PHONY: clean

clean:
	rm lide_c_mlpsingle_graph.o libgraph.a
