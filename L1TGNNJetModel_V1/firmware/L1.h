#ifndef L1_H_
#define L1_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"


// Prototype of top level function for C-synthesis
void L1(
    input_t model_input[N_INPUT_1_1*N_INPUT_2_1], input3_t adj_input[N_INPUT_1_3*N_INPUT_2_3],
    layer21_t layer21_out[N_LAYER_18], layer23_t layer23_out[N_LAYER_22]
);


#endif
