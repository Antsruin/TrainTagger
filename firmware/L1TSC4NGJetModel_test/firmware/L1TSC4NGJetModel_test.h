#ifndef L1TSC4NGJETMODEL_TEST_H_
#define L1TSC4NGJETMODEL_TEST_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"


// Prototype of top level function for C-synthesis
void L1TSC4NGJetModel_test(
    input_t model_input[N_INPUT_1_1*N_INPUT_2_1],
    layer34_t layer34_out[N_LAYER_31], layer36_t layer36_out[N_LAYER_35]
);


#endif
