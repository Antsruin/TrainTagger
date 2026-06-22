#include <iostream>

#include "L1TGNNJetModel_V1.h"
#include "parameters.h"


void L1TGNNJetModel_V1(
    input_t model_input[N_INPUT_1_1*N_INPUT_2_1], input3_t adj_input[N_INPUT_1_3*N_INPUT_2_3],
    layer25_t layer25_out[N_LAYER_22], layer27_t layer27_out[N_LAYER_26]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=model_input complete dim=0
    #pragma HLS ARRAY_RESHAPE variable=adj_input complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer25_out complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer27_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=model_input,adj_input,layer25_out,layer27_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<bn_input_default_t, 20>(s2, "s2.txt");
        nnet::load_weights_from_txt<bn_input_default_t, 20>(b2, "b2.txt");
        nnet::load_weights_from_txt<gcn_dense_0_weight_t, 640>(w30, "w30.txt");
        nnet::load_weights_from_txt<gcn_dense_0_bias_t, 32>(b30, "b30.txt");
        nnet::load_weights_from_txt<gcn_dense_1_weight_t, 1024>(w31, "w31.txt");
        nnet::load_weights_from_txt<gcn_dense_1_bias_t, 32>(b31, "b31.txt");
        nnet::load_weights_from_txt<weight14_t, 1024>(w14, "w14.txt");
        nnet::load_weights_from_txt<bias14_t, 32>(b14, "b14.txt");
        nnet::load_weights_from_txt<weight17_t, 512>(w17, "w17.txt");
        nnet::load_weights_from_txt<bias17_t, 16>(b17, "b17.txt");
        nnet::load_weights_from_txt<weight20_t, 128>(w20, "w20.txt");
        nnet::load_weights_from_txt<bias20_t, 8>(b20, "b20.txt");
        nnet::load_weights_from_txt<weight22_t, 128>(w22, "w22.txt");
        nnet::load_weights_from_txt<bias22_t, 8>(b22, "b22.txt");
        nnet::load_weights_from_txt<weight26_t, 8>(w26, "w26.txt");
        nnet::load_weights_from_txt<bias26_t, 1>(b26, "b26.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    layer2_t layer2_out[N_INPUT_1_1*N_INPUT_2_1];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    nnet::normalize<input_t, layer2_t, config2>(model_input, layer2_out, s2, b2); // bn_input

    gcn_dense_0_result_t layer30_out[N_OUTPUTS_30*N_FILT_30];
    #pragma HLS ARRAY_PARTITION variable=layer30_out complete dim=0
    nnet::pointwise_conv_1d_cl<layer2_t, gcn_dense_0_result_t, config30>(layer2_out, layer30_out, w30, b30); // gcn_dense_0

    layer6_t layer6_out[N_INPUT_1_3*N_LAYER_2_4];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0
    nnet::dot2d<input3_t, gcn_dense_0_result_t, layer6_t, config6>(adj_input, layer30_out, layer6_out); // gcn_agg_0

    layer7_t layer7_out[N_INPUT_1_3*N_LAYER_2_4];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    nnet::relu<layer6_t, layer7_t, relu_config7>(layer6_out, layer7_out); // gcn_act_0

    gcn_dense_1_result_t layer31_out[N_OUTPUTS_31*N_FILT_31];
    #pragma HLS ARRAY_PARTITION variable=layer31_out complete dim=0
    nnet::pointwise_conv_1d_cl<layer7_t, gcn_dense_1_result_t, config31>(layer7_out, layer31_out, w31, b31); // gcn_dense_1

    layer10_t layer10_out[N_INPUT_1_3*N_LAYER_2_8];
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0
    nnet::dot2d<input3_t, gcn_dense_1_result_t, layer10_t, config10>(adj_input, layer31_out, layer10_out); // gcn_agg_1

    layer11_t layer11_out[N_INPUT_1_3*N_LAYER_2_8];
    #pragma HLS ARRAY_PARTITION variable=layer11_out complete dim=0
    nnet::relu<layer10_t, layer11_t, relu_config11>(layer10_out, layer11_out); // gcn_act_1

    layer12_t layer12_out[N_INPUT_1_3*N_LAYER_2_8];
    #pragma HLS ARRAY_PARTITION variable=layer12_out complete dim=0
    nnet::linear<layer11_t, layer12_t, linear_config12>(layer11_out, layer12_out); // act_pool

    layer13_t layer13_out[N_FILT_13];
    #pragma HLS ARRAY_PARTITION variable=layer13_out complete dim=0
    nnet::global_pooling1d_cl<layer12_t, layer13_t, config13>(layer12_out, layer13_out); // pool

    dense_0_result_t layer14_out[N_LAYER_14];
    #pragma HLS ARRAY_PARTITION variable=layer14_out complete dim=0
    nnet::dense<layer13_t, dense_0_result_t, config14>(layer13_out, layer14_out, w14, b14); // dense_0

    layer16_t layer16_out[N_LAYER_14];
    #pragma HLS ARRAY_PARTITION variable=layer16_out complete dim=0
    nnet::relu<dense_0_result_t, layer16_t, relu_config16>(layer14_out, layer16_out); // act_0

    dense_1_result_t layer17_out[N_LAYER_17];
    #pragma HLS ARRAY_PARTITION variable=layer17_out complete dim=0
    nnet::dense<layer16_t, dense_1_result_t, config17>(layer16_out, layer17_out, w17, b17); // dense_1

    layer19_t layer19_out[N_LAYER_17];
    #pragma HLS ARRAY_PARTITION variable=layer19_out complete dim=0
    nnet::relu<dense_1_result_t, layer19_t, relu_config19>(layer17_out, layer19_out); // act_1

    jet_pt_regression_dense_result_t layer20_out[N_LAYER_20];
    #pragma HLS ARRAY_PARTITION variable=layer20_out complete dim=0
    nnet::dense<layer19_t, jet_pt_regression_dense_result_t, config20>(layer19_out, layer20_out, w20, b20); // jet_pt_regression_dense

    jet_id_classification_result_t layer22_out[N_LAYER_22];
    #pragma HLS ARRAY_PARTITION variable=layer22_out complete dim=0
    nnet::dense<layer19_t, jet_id_classification_result_t, config22>(layer19_out, layer22_out, w22, b22); // jet_id_classification

    layer24_t layer24_out[N_LAYER_20];
    #pragma HLS ARRAY_PARTITION variable=layer24_out complete dim=0
    nnet::relu<jet_pt_regression_dense_result_t, layer24_t, relu_config24>(layer20_out, layer24_out); // jet_pt_regression_act

    nnet::softmax<jet_id_classification_result_t, layer25_t, Softmax_config25>(layer22_out, layer25_out); // jet_id_output

    layer26_t layer26_out[N_LAYER_26];
    #pragma HLS ARRAY_PARTITION variable=layer26_out complete dim=0
    nnet::dense<layer24_t, layer26_t, config26>(layer24_out, layer26_out, w26, b26); // pT_output

    nnet::linear<layer26_t, layer27_t, linear_config27>(layer26_out, layer27_out); // pT_output_linear

}

