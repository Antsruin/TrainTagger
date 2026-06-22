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
#define N_OUTPUTS_26 16
#define N_FILT_26 32
#define N_INPUT_1_3 16
#define N_LAYER_2_4 32
#define N_INPUT_1_3 16
#define N_LAYER_2_4 32
#define N_OUTPUTS_27 16
#define N_FILT_27 16
#define N_INPUT_1_3 16
#define N_LAYER_2_8 16
#define N_INPUT_1_3 16
#define N_LAYER_2_8 16
#define N_FILT_12 16
#define N_LAYER_13 16
#define N_LAYER_13 16
#define N_LAYER_16 8
#define N_LAYER_18 8
#define N_LAYER_16 8
#define N_LAYER_18 8
#define N_LAYER_22 1
#define N_LAYER_22 1


// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<24,12,AP_RND,AP_SAT,0> input_t;
typedef ap_fixed<24,12,AP_RND,AP_SAT,0> layer2_t;
typedef ap_fixed<24,12,AP_RND,AP_SAT,0> bn_input_default_t;
typedef ap_fixed<24,12,AP_RND,AP_SAT,0> input3_t;
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<39,20> gcn_dense_0_result_t;
typedef ap_fixed<9,2,AP_RND,AP_SAT,0> gcn_dense_0_weight_t;
typedef ap_fixed<9,2,AP_RND,AP_SAT,0> gcn_dense_0_bias_t;
typedef ap_fixed<24,12,AP_RND,AP_SAT,0> gcn_agg_0_default_t;
typedef ap_fixed<24,12,AP_RND,AP_SAT,0> layer6_t;
typedef ap_ufixed<9,0,AP_RND_CONV,AP_SAT,0> layer7_t;
typedef ap_fixed<18,8> gcn_act_0_table_t;
typedef ap_fixed<24,8> gcn_dense_1_result_t;
typedef ap_fixed<9,2,AP_RND,AP_SAT,0> gcn_dense_1_weight_t;
typedef ap_fixed<9,2,AP_RND,AP_SAT,0> gcn_dense_1_bias_t;
typedef ap_fixed<24,12,AP_RND,AP_SAT,0> gcn_agg_1_default_t;
typedef ap_fixed<24,12,AP_RND,AP_SAT,0> layer10_t;
typedef ap_ufixed<9,0,AP_RND_CONV,AP_SAT,0> layer11_t;
typedef ap_fixed<18,8> gcn_act_1_table_t;
typedef ap_fixed<16,6> layer12_t;
typedef ap_fixed<30,14> dense_0_result_t;
typedef ap_fixed<9,3> weight13_t;
typedef ap_fixed<9,3> bias13_t;
typedef ap_uint<1> layer13_index;
typedef ap_ufixed<9,0,AP_RND_CONV,AP_SAT,0> layer15_t;
typedef ap_fixed<18,8> act_0_table_t;
typedef ap_fixed<23,8> jet_pt_reg_0_result_t;
typedef ap_fixed<9,3> weight16_t;
typedef ap_fixed<9,3> bias16_t;
typedef ap_uint<1> layer16_index;
typedef ap_fixed<23,8> jet_id_classification_result_t;
typedef ap_fixed<9,3> weight18_t;
typedef ap_fixed<9,3> bias18_t;
typedef ap_uint<1> layer18_index;
typedef ap_ufixed<9,0,AP_RND_CONV,AP_SAT,0> layer20_t;
typedef ap_fixed<18,8> jet_pt_reg_act_0_table_t;
typedef ap_ufixed<24,12,AP_RND,AP_SAT,0> layer21_t;
typedef ap_fixed<18,8> jet_id_output_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT,0> jet_id_output_exp_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT,0> jet_id_output_inv_table_t;
typedef ap_fixed<16,6,AP_RND,AP_SAT,0> layer22_t;
typedef ap_fixed<16,7> weight22_t;
typedef ap_fixed<16,7> bias22_t;
typedef ap_uint<1> layer22_index;
typedef ap_fixed<16,6> layer23_t;
typedef ap_fixed<18,8> pT_output_linear_table_t;


#endif
