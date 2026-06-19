#include <iostream>

#include "L1TSC4NGJetModel_test.h"
#include "parameters.h"


void L1TSC4NGJetModel_test(
    input_t model_input[N_INPUT_1_1*N_INPUT_2_1],
    layer34_t layer34_out[N_LAYER_31], layer36_t layer36_out[N_LAYER_35]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=model_input complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer34_out complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer36_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=model_input,layer34_out,layer36_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<norm_input_default_t, 20>(s2, "s2.txt");
        nnet::load_weights_from_txt<norm_input_default_t, 20>(b2, "b2.txt");
        nnet::load_weights_from_txt<weight37_t, 600>(w37, "w37.txt");
        nnet::load_weights_from_txt<bias37_t, 30>(b37, "b37.txt");
        nnet::load_weights_from_txt<weight38_t, 450>(w38, "w38.txt");
        nnet::load_weights_from_txt<bias38_t, 15>(b38, "b38.txt");
        nnet::load_weights_from_txt<weight39_t, 150>(w39, "w39.txt");
        nnet::load_weights_from_txt<bias39_t, 10>(b39, "b39.txt");
        nnet::load_weights_from_txt<weight14_t, 320>(w14, "w14.txt");
        nnet::load_weights_from_txt<bias14_t, 32>(b14, "b14.txt");
        nnet::load_weights_from_txt<weight17_t, 160>(w17, "w17.txt");
        nnet::load_weights_from_txt<bias17_t, 16>(b17, "b17.txt");
        nnet::load_weights_from_txt<weight19_t, 512>(w19, "w19.txt");
        nnet::load_weights_from_txt<bias19_t, 16>(b19, "b19.txt");
        nnet::load_weights_from_txt<weight23_t, 128>(w23, "w23.txt");
        nnet::load_weights_from_txt<bias23_t, 8>(b23, "b23.txt");
        nnet::load_weights_from_txt<weight25_t, 128>(w25, "w25.txt");
        nnet::load_weights_from_txt<bias25_t, 8>(b25, "b25.txt");
        nnet::load_weights_from_txt<weight29_t, 32>(w29, "w29.txt");
        nnet::load_weights_from_txt<bias29_t, 4>(b29, "b29.txt");
        nnet::load_weights_from_txt<weight31_t, 64>(w31, "w31.txt");
        nnet::load_weights_from_txt<bias31_t, 8>(b31, "b31.txt");
        nnet::load_weights_from_txt<weight35_t, 4>(w35, "w35.txt");
        nnet::load_weights_from_txt<bias35_t, 1>(b35, "b35.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    layer2_t layer2_out[N_INPUT_1_1*N_INPUT_2_1];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    nnet::normalize<input_t, layer2_t, config2>(model_input, layer2_out, s2, b2); // norm_input

    Conv1D_1_result_t layer37_out[N_OUTPUTS_37*N_FILT_37];
    #pragma HLS ARRAY_PARTITION variable=layer37_out complete dim=0
    nnet::pointwise_conv_1d_cl<layer2_t, Conv1D_1_result_t, config37>(layer2_out, layer37_out, w37, b37); // Conv1D_1

    layer5_t layer5_out[N_OUTPUTS_3*N_FILT_3];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    nnet::relu<Conv1D_1_result_t, layer5_t, relu_config5>(layer37_out, layer5_out); // relu_1

    Conv1D_2_result_t layer38_out[N_OUTPUTS_38*N_FILT_38];
    #pragma HLS ARRAY_PARTITION variable=layer38_out complete dim=0
    nnet::pointwise_conv_1d_cl<layer5_t, Conv1D_2_result_t, config38>(layer5_out, layer38_out, w38, b38); // Conv1D_2

    layer8_t layer8_out[N_OUTPUTS_6*N_FILT_6];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0
    nnet::relu<Conv1D_2_result_t, layer8_t, relu_config8>(layer38_out, layer8_out); // relu_2

    Conv1D_3_result_t layer39_out[N_OUTPUTS_39*N_FILT_39];
    #pragma HLS ARRAY_PARTITION variable=layer39_out complete dim=0
    nnet::pointwise_conv_1d_cl<layer8_t, Conv1D_3_result_t, config39>(layer8_out, layer39_out, w39, b39); // Conv1D_3

    layer11_t layer11_out[N_OUTPUTS_9*N_FILT_9];
    #pragma HLS ARRAY_PARTITION variable=layer11_out complete dim=0
    nnet::relu<Conv1D_3_result_t, layer11_t, relu_config11>(layer39_out, layer11_out); // relu_3

    layer12_t layer12_out[N_OUTPUTS_9*N_FILT_9];
    #pragma HLS ARRAY_PARTITION variable=layer12_out complete dim=0
    nnet::linear<layer11_t, layer12_t, linear_config12>(layer11_out, layer12_out); // act_pool

    layer13_t layer13_out[N_FILT_13];
    #pragma HLS ARRAY_PARTITION variable=layer13_out complete dim=0
    nnet::global_pooling1d_cl<layer12_t, layer13_t, config13>(layer12_out, layer13_out); // pool

    Dense_1_jetID_result_t layer14_out[N_LAYER_14];
    #pragma HLS ARRAY_PARTITION variable=layer14_out complete dim=0
    nnet::dense<layer13_t, Dense_1_jetID_result_t, config14>(layer13_out, layer14_out, w14, b14); // Dense_1_jetID

    layer16_t layer16_out[N_LAYER_14];
    #pragma HLS ARRAY_PARTITION variable=layer16_out complete dim=0
    nnet::relu<Dense_1_jetID_result_t, layer16_t, relu_config16>(layer14_out, layer16_out); // relu_1_jetID

    Dense_1_pT_result_t layer17_out[N_LAYER_17];
    #pragma HLS ARRAY_PARTITION variable=layer17_out complete dim=0
    nnet::dense<layer13_t, Dense_1_pT_result_t, config17>(layer13_out, layer17_out, w17, b17); // Dense_1_pT

    Dense_2_jetID_result_t layer19_out[N_LAYER_19];
    #pragma HLS ARRAY_PARTITION variable=layer19_out complete dim=0
    nnet::dense<layer16_t, Dense_2_jetID_result_t, config19>(layer16_out, layer19_out, w19, b19); // Dense_2_jetID

    layer21_t layer21_out[N_LAYER_17];
    #pragma HLS ARRAY_PARTITION variable=layer21_out complete dim=0
    nnet::relu<Dense_1_pT_result_t, layer21_t, relu_config21>(layer17_out, layer21_out); // relu_1_pT

    layer22_t layer22_out[N_LAYER_19];
    #pragma HLS ARRAY_PARTITION variable=layer22_out complete dim=0
    nnet::relu<Dense_2_jetID_result_t, layer22_t, relu_config22>(layer19_out, layer22_out); // relu_2_jetID

    Dense_2_pT_result_t layer23_out[N_LAYER_23];
    #pragma HLS ARRAY_PARTITION variable=layer23_out complete dim=0
    nnet::dense<layer21_t, Dense_2_pT_result_t, config23>(layer21_out, layer23_out, w23, b23); // Dense_2_pT

    Dense_3_jetID_result_t layer25_out[N_LAYER_25];
    #pragma HLS ARRAY_PARTITION variable=layer25_out complete dim=0
    nnet::dense<layer22_t, Dense_3_jetID_result_t, config25>(layer22_out, layer25_out, w25, b25); // Dense_3_jetID

    layer27_t layer27_out[N_LAYER_23];
    #pragma HLS ARRAY_PARTITION variable=layer27_out complete dim=0
    nnet::relu<Dense_2_pT_result_t, layer27_t, relu_config27>(layer23_out, layer27_out); // relu_2_pT

    layer28_t layer28_out[N_LAYER_25];
    #pragma HLS ARRAY_PARTITION variable=layer28_out complete dim=0
    nnet::relu<Dense_3_jetID_result_t, layer28_t, relu_config28>(layer25_out, layer28_out); // relu_3_jetID

    Dense_3_pT_result_t layer29_out[N_LAYER_29];
    #pragma HLS ARRAY_PARTITION variable=layer29_out complete dim=0
    nnet::dense<layer27_t, Dense_3_pT_result_t, config29>(layer27_out, layer29_out, w29, b29); // Dense_3_pT

    Dense_4_jetID_result_t layer31_out[N_LAYER_31];
    #pragma HLS ARRAY_PARTITION variable=layer31_out complete dim=0
    nnet::dense<layer28_t, Dense_4_jetID_result_t, config31>(layer28_out, layer31_out, w31, b31); // Dense_4_jetID

    layer33_t layer33_out[N_LAYER_29];
    #pragma HLS ARRAY_PARTITION variable=layer33_out complete dim=0
    nnet::relu<Dense_3_pT_result_t, layer33_t, relu_config33>(layer29_out, layer33_out); // relu_3_pT

    nnet::softmax<Dense_4_jetID_result_t, layer34_t, softmax_config34>(layer31_out, layer34_out); // jet_id_output

    layer35_t layer35_out[N_LAYER_35];
    #pragma HLS ARRAY_PARTITION variable=layer35_out complete dim=0
    nnet::dense<layer33_t, layer35_t, config35>(layer33_out, layer35_out, w35, b35); // pT_output

    nnet::linear<layer35_t, layer36_t, linear_config36>(layer35_out, layer36_out); // pT_output_linear

}

