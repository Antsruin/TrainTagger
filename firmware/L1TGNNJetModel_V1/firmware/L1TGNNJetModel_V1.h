#ifndef L1TGNNJETMODEL_V1_H_
#define L1TGNNJETMODEL_V1_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"


// Prototype of top level function for C-synthesis
void L1TGNNJetModel_V1(
    input_t model_input[N_INPUT_1_1*N_INPUT_2_1], input3_t adj_input[N_INPUT_1_3*N_INPUT_2_3],
    layer25_t layer25_out[N_LAYER_22], layer27_t layer27_out[N_LAYER_26]
);


#endif
