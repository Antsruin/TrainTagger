#include <iostream>

#include "L1.h"
#include "parameters.h"


void L1(
    input_t model_input[N_INPUT_1_1*N_INPUT_2_1], input3_t adj_input[N_INPUT_1_3*N_INPUT_2_3],
    layer21_t layer21_out[N_LAYER_18], layer23_t layer23_out[N_LAYER_22]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=model_input complete dim=0
    #pragma HLS ARRAY_RESHAPE variable=adj_input complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer21_out complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer23_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=model_input,adj_input,layer21_out,layer23_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<bn_input_default_t, 20>(s2, "s2.txt");
        nnet::load_weights_from_txt<bn_input_default_t, 20>(b2, "b2.txt");
        nnet::load_weights_from_txt<gcn_dense_0_weight_t, 640>(w26, "w26.txt");
        nnet::load_weights_from_txt<gcn_dense_0_bias_t, 32>(b26, "b26.txt");
        nnet::load_weights_from_txt<gcn_dense_1_weight_t, 512>(w27, "w27.txt");
        nnet::load_weights_from_txt<gcn_dense_1_bias_t, 16>(b27, "b27.txt");
        nnet::load_weights_from_txt<weight13_t, 256>(w13, "w13.txt");
        nnet::load_weights_from_txt<bias13_t, 16>(b13, "b13.txt");
        nnet::load_weights_from_txt<weight16_t, 128>(w16, "w16.txt");
        nnet::load_weights_from_txt<bias16_t, 8>(b16, "b16.txt");
        nnet::load_weights_from_txt<weight18_t, 128>(w18, "w18.txt");
        nnet::load_weights_from_txt<bias18_t, 8>(b18, "b18.txt");
        nnet::load_weights_from_txt<weight22_t, 8>(w22, "w22.txt");
        nnet::load_weights_from_txt<bias22_t, 1>(b22, "b22.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    layer2_t layer2_out[N_INPUT_1_1*N_INPUT_2_1];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    nnet::normalize<input_t, layer2_t, config2>(model_input, layer2_out, s2, b2); // bn_input

    gcn_dense_0_result_t layer26_out[N_OUTPUTS_26*N_FILT_26];
    #pragma HLS ARRAY_PARTITION variable=layer26_out complete dim=0
    nnet::pointwise_conv_1d_cl<layer2_t, gcn_dense_0_result_t, config26>(layer2_out, layer26_out, w26, b26); // gcn_dense_0

    layer6_t layer6_out[N_INPUT_1_3*N_LAYER_2_4];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0
    nnet::dot2d<input3_t, gcn_dense_0_result_t, layer6_t, config6>(adj_input, layer26_out, layer6_out); // gcn_agg_0

    layer7_t layer7_out[N_INPUT_1_3*N_LAYER_2_4];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    nnet::relu<layer6_t, layer7_t, relu_config7>(layer6_out, layer7_out); // gcn_act_0

    gcn_dense_1_result_t layer27_out[N_OUTPUTS_27*N_FILT_27];
    #pragma HLS ARRAY_PARTITION variable=layer27_out complete dim=0
    nnet::pointwise_conv_1d_cl<layer7_t, gcn_dense_1_result_t, config27>(layer7_out, layer27_out, w27, b27); // gcn_dense_1

    layer10_t layer10_out[N_INPUT_1_3*N_LAYER_2_8];
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0
    nnet::dot2d<input3_t, gcn_dense_1_result_t, layer10_t, config10>(adj_input, layer27_out, layer10_out); // gcn_agg_1

    layer11_t layer11_out[N_INPUT_1_3*N_LAYER_2_8];
    #pragma HLS ARRAY_PARTITION variable=layer11_out complete dim=0
    nnet::relu<layer10_t, layer11_t, relu_config11>(layer10_out, layer11_out); // gcn_act_1

    layer12_t layer12_out[N_FILT_12];
    #pragma HLS ARRAY_PARTITION variable=layer12_out complete dim=0
    nnet::global_pooling1d_cl<layer11_t, layer12_t, config12>(layer11_out, layer12_out); // pool

    dense_0_result_t layer13_out[N_LAYER_13];
    #pragma HLS ARRAY_PARTITION variable=layer13_out complete dim=0
    nnet::dense<layer12_t, dense_0_result_t, config13>(layer12_out, layer13_out, w13, b13); // dense_0

    layer15_t layer15_out[N_LAYER_13];
    #pragma HLS ARRAY_PARTITION variable=layer15_out complete dim=0
    nnet::relu<dense_0_result_t, layer15_t, relu_config15>(layer13_out, layer15_out); // act_0

    jet_pt_reg_0_result_t layer16_out[N_LAYER_16];
    #pragma HLS ARRAY_PARTITION variable=layer16_out complete dim=0
    nnet::dense<layer15_t, jet_pt_reg_0_result_t, config16>(layer15_out, layer16_out, w16, b16); // jet_pt_reg_0

    jet_id_classification_result_t layer18_out[N_LAYER_18];
    #pragma HLS ARRAY_PARTITION variable=layer18_out complete dim=0
    nnet::dense<layer15_t, jet_id_classification_result_t, config18>(layer15_out, layer18_out, w18, b18); // jet_id_classification

    layer20_t layer20_out[N_LAYER_16];
    #pragma HLS ARRAY_PARTITION variable=layer20_out complete dim=0
    nnet::relu<jet_pt_reg_0_result_t, layer20_t, relu_config20>(layer16_out, layer20_out); // jet_pt_reg_act_0

    nnet::softmax<jet_id_classification_result_t, layer21_t, Softmax_config21>(layer18_out, layer21_out); // jet_id_output

    layer22_t layer22_out[N_LAYER_22];
    #pragma HLS ARRAY_PARTITION variable=layer22_out complete dim=0
    nnet::dense<layer20_t, layer22_t, config22>(layer20_out, layer22_out, w22, b22); // pT_output

    nnet::linear<layer22_t, layer23_t, linear_config23>(layer22_out, layer23_out); // pT_output_linear

}

