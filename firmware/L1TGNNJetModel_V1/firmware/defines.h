#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 16
#define N_INPUT_2_1 20
#define N_INPUT_1_1 16
#define N_INPUT_2_1 20
#define N_INPUT_1_3 16
#define N_INPUT_2_3 16
#define N_OUTPUTS_30 16
#define N_FILT_30 32
#define N_INPUT_1_3 16
#define N_LAYER_2_4 32
#define N_INPUT_1_3 16
#define N_LAYER_2_4 32
#define N_OUTPUTS_31 16
#define N_FILT_31 32
#define N_INPUT_1_3 16
#define N_LAYER_2_8 32
#define N_INPUT_1_3 16
#define N_LAYER_2_8 32
#define N_INPUT_1_3 16
#define N_LAYER_2_8 32
#define N_FILT_13 32
#define N_LAYER_14 32
#define N_LAYER_14 32
#define N_LAYER_17 16
#define N_LAYER_17 16
#define N_LAYER_20 8
#define N_LAYER_22 8
#define N_LAYER_20 8
#define N_LAYER_22 8
#define N_LAYER_26 1
#define N_LAYER_26 1


// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<24,12,AP_RND,AP_SAT,0> input_t;
typedef ap_fixed<24,12,AP_RND,AP_SAT,0> layer2_t;
typedef ap_fixed<24,12,AP_RND,AP_SAT,0> bn_input_default_t;
typedef ap_fixed<24,12,AP_RND,AP_SAT,0> input3_t;
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<39,21> gcn_dense_0_result_t;
typedef ap_fixed<9,3> gcn_dense_0_weight_t;
typedef ap_fixed<9,3> gcn_dense_0_bias_t;
typedef ap_fixed<24,12,AP_RND,AP_SAT,0> gcn_agg_0_default_t;
typedef ap_fixed<24,12,AP_RND,AP_SAT,0> layer6_t;
typedef ap_ufixed<9,0,AP_RND_CONV,AP_SAT,0> layer7_t;
typedef ap_fixed<18,8> gcn_act_0_table_t;
typedef ap_fixed<24,9> gcn_dense_1_result_t;
typedef ap_fixed<9,3> gcn_dense_1_weight_t;
typedef ap_fixed<9,3> gcn_dense_1_bias_t;
typedef ap_fixed<24,12,AP_RND,AP_SAT,0> gcn_agg_1_default_t;
typedef ap_fixed<24,12,AP_RND,AP_SAT,0> layer10_t;
typedef ap_ufixed<9,0,AP_RND_CONV,AP_SAT,0> layer11_t;
typedef ap_fixed<18,8> gcn_act_1_table_t;
typedef ap_fixed<18,9,AP_RND_CONV,AP_SAT,0> layer12_t;
typedef ap_fixed<18,8> act_pool_table_t;
typedef ap_fixed<16,6> layer13_t;
typedef ap_fixed<31,15> dense_0_result_t;
typedef ap_fixed<9,3> weight14_t;
typedef ap_fixed<9,3> bias14_t;
typedef ap_uint<1> layer14_index;
typedef ap_ufixed<9,0,AP_RND_CONV,AP_SAT,0> layer16_t;
typedef ap_fixed<18,8> act_0_table_t;
typedef ap_fixed<24,9> dense_1_result_t;
typedef ap_fixed<9,3> weight17_t;
typedef ap_fixed<9,3> bias17_t;
typedef ap_uint<1> layer17_index;
typedef ap_ufixed<9,0,AP_RND_CONV,AP_SAT,0> layer19_t;
typedef ap_fixed<18,8> act_1_table_t;
typedef ap_fixed<23,8> jet_pt_regression_dense_result_t;
typedef ap_fixed<9,3> weight20_t;
typedef ap_fixed<9,3> bias20_t;
typedef ap_uint<1> layer20_index;
typedef ap_fixed<23,8> jet_id_classification_result_t;
typedef ap_fixed<9,3> weight22_t;
typedef ap_fixed<9,3> bias22_t;
typedef ap_uint<1> layer22_index;
typedef ap_ufixed<9,0,AP_RND_CONV,AP_SAT,0> layer24_t;
typedef ap_fixed<18,8> jet_pt_regression_act_table_t;
typedef ap_ufixed<24,12,AP_RND,AP_SAT,0> layer25_t;
typedef ap_fixed<18,8> jet_id_output_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT,0> jet_id_output_exp_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT,0> jet_id_output_inv_table_t;
typedef ap_fixed<24,12,AP_RND,AP_SAT,0> layer26_t;
typedef ap_fixed<16,7> weight26_t;
typedef ap_fixed<16,7> bias26_t;
typedef ap_uint<1> layer26_index;
typedef ap_fixed<16,6> layer27_t;
typedef ap_fixed<18,8> pT_output_linear_table_t;


#endif
