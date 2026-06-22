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
#define N_OUTPUTS_37 16
#define N_FILT_37 30
#define N_OUTPUTS_3 16
#define N_FILT_3 30
#define N_OUTPUTS_38 16
#define N_FILT_38 15
#define N_OUTPUTS_6 16
#define N_FILT_6 15
#define N_OUTPUTS_39 16
#define N_FILT_39 10
#define N_OUTPUTS_9 16
#define N_FILT_9 10
#define N_OUTPUTS_9 16
#define N_FILT_9 10
#define N_FILT_13 10
#define N_LAYER_14 32
#define N_LAYER_14 32
#define N_LAYER_17 16
#define N_LAYER_19 16
#define N_LAYER_17 16
#define N_LAYER_19 16
#define N_LAYER_23 8
#define N_LAYER_25 8
#define N_LAYER_23 8
#define N_LAYER_25 8
#define N_LAYER_29 4
#define N_LAYER_31 8
#define N_LAYER_29 4
#define N_LAYER_31 8
#define N_LAYER_35 1
#define N_LAYER_35 1


// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<24,12,AP_RND,AP_SAT,0> input_t;
typedef ap_fixed<24,12,AP_RND,AP_SAT,0> layer2_t;
typedef ap_fixed<24,12,AP_RND,AP_SAT,0> norm_input_default_t;
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<39,21> Conv1D_1_result_t;
typedef ap_fixed<9,3> weight37_t;
typedef ap_fixed<9,3> bias37_t;
typedef ap_ufixed<9,0,AP_RND_CONV,AP_SAT,0> layer5_t;
typedef ap_fixed<18,8> relu_1_table_t;
typedef ap_fixed<24,9> Conv1D_2_result_t;
typedef ap_fixed<9,3> weight38_t;
typedef ap_fixed<9,3> bias38_t;
typedef ap_ufixed<9,0,AP_RND_CONV,AP_SAT,0> layer8_t;
typedef ap_fixed<18,8> relu_2_table_t;
typedef ap_fixed<23,8> Conv1D_3_result_t;
typedef ap_fixed<9,3> weight39_t;
typedef ap_fixed<9,3> bias39_t;
typedef ap_ufixed<9,0,AP_RND_CONV,AP_SAT,0> layer11_t;
typedef ap_fixed<18,8> relu_3_table_t;
typedef ap_fixed<18,9,AP_RND_CONV,AP_SAT,0> layer12_t;
typedef ap_fixed<18,8> act_pool_table_t;
typedef ap_fixed<16,6> layer13_t;
typedef ap_fixed<30,14> Dense_1_jetID_result_t;
typedef ap_fixed<9,3> weight14_t;
typedef ap_fixed<9,3> bias14_t;
typedef ap_uint<1> layer14_index;
typedef ap_ufixed<9,0,AP_RND_CONV,AP_SAT,0> layer16_t;
typedef ap_fixed<18,8> relu_1_jetID_table_t;
typedef ap_fixed<30,14> Dense_1_pT_result_t;
typedef ap_fixed<9,3> weight17_t;
typedef ap_fixed<9,3> bias17_t;
typedef ap_uint<1> layer17_index;
typedef ap_fixed<24,9> Dense_2_jetID_result_t;
typedef ap_fixed<9,3> weight19_t;
typedef ap_fixed<9,3> bias19_t;
typedef ap_uint<1> layer19_index;
typedef ap_ufixed<9,0,AP_RND_CONV,AP_SAT,0> layer21_t;
typedef ap_fixed<18,8> relu_1_pT_table_t;
typedef ap_ufixed<9,0,AP_RND_CONV,AP_SAT,0> layer22_t;
typedef ap_fixed<18,8> relu_2_jetID_table_t;
typedef ap_fixed<23,8> Dense_2_pT_result_t;
typedef ap_fixed<9,3> weight23_t;
typedef ap_fixed<9,3> bias23_t;
typedef ap_uint<1> layer23_index;
typedef ap_fixed<23,8> Dense_3_jetID_result_t;
typedef ap_fixed<9,3> weight25_t;
typedef ap_fixed<9,3> bias25_t;
typedef ap_uint<1> layer25_index;
typedef ap_ufixed<9,0,AP_RND_CONV,AP_SAT,0> layer27_t;
typedef ap_fixed<18,8> relu_2_pT_table_t;
typedef ap_ufixed<9,0,AP_RND_CONV,AP_SAT,0> layer28_t;
typedef ap_fixed<18,8> relu_3_jetID_table_t;
typedef ap_fixed<22,7> Dense_3_pT_result_t;
typedef ap_fixed<9,3> weight29_t;
typedef ap_fixed<9,3> bias29_t;
typedef ap_uint<1> layer29_index;
typedef ap_fixed<22,7> Dense_4_jetID_result_t;
typedef ap_fixed<9,3> weight31_t;
typedef ap_fixed<9,3> bias31_t;
typedef ap_uint<1> layer31_index;
typedef ap_ufixed<9,0,AP_RND_CONV,AP_SAT,0> layer33_t;
typedef ap_fixed<18,8> relu_3_pT_table_t;
typedef ap_ufixed<24,12,AP_RND,AP_SAT,0> layer34_t;
typedef ap_fixed<18,8> jet_id_output_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT,0> jet_id_output_exp_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT,0> jet_id_output_inv_table_t;
typedef ap_fixed<16,6,AP_RND,AP_SAT,0> layer35_t;
typedef ap_fixed<16,7> weight35_t;
typedef ap_fixed<16,7> bias35_t;
typedef ap_uint<1> layer35_index;
typedef ap_fixed<16,6> layer36_t;
typedef ap_fixed<18,8> pT_output_linear_table_t;


#endif
