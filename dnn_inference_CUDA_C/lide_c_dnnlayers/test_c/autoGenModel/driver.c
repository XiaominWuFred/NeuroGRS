#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include "lide_c_util.h"
#include "lide_c_mlpsingle_graph.h"
int main(int argc, char **argv) {
    const char* InputsFile = "../../../../graphGen/extractedParas/testInputXDEN.csv";
    const char* OutFile = "../result.csv";
    const char* Wfst="../../../../graphGen/extractedParas/fstWeight.csv";
    const char* Bfst = "../../../../graphGen/extractedParas/fstBias.csv";
    const char* Wsnd="../../../../graphGen/extractedParas/sndWeight.csv";
    const char* Bsnd = "../../../../graphGen/extractedParas/sndBias.csv";
    lide_c_graph_context_type *graph = NULL;
    graph = (lide_c_graph_context_type*)lide_c_mlpsingle_graph_new(InputsFile,
    Wfst, Bfst,
    Wsnd, Bsnd,
    OutFile);
    graph->scheduler(graph);
    return 0;
}
