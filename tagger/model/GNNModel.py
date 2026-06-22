import os
import json
import hls4ml
import numpy as np
import tensorflow as tf
import qkeras
from qkeras import BatchNormalization, QDense, QActivation,QBatchNormalization, quantized_bits, quantized_relu
from schema import Schema, And, Optional

from tagger.model.JetTagModel import JetModelFactory, JetTagModel
from tagger.model.QKerasModel import QKerasModel
from tagger.model.common import choose_aggregator, initialise_tensorflow


# ── Model Class ───────────────────────────────────────────────────────────────
@JetModelFactory.register('GNNModel')
class GNNModel(QKerasModel):

    # YAML schema validation
    schema = Schema({
        'model': str,
        "run_config" : JetTagModel.run_schema,
        'model_config': {
            'gcn_units':    [int],   # e.g. [32, 32]
            'dense_units':  [int],   # e.g. [16]
            'regression_units': [int],
            'regression_output_units': int,
            'kernel_initializer' : str,
            "aggregator" : And(str, lambda s: s in  ["mean", "max", "attention"]),
        },
        'quantization_config' : QKerasModel.quantization_schema,
        'training_config' : {"weight_method" : And(str, lambda s: s in  ["none", "ptref", "onlyclass"]),
                            "validation_split" : And(float, lambda s: s > 0.0),
                            "epochs" : And(int, lambda s: s >= 1),
                            "batch_size" : And(int, lambda s: s >= 1),
                            "learning_rate" : And(float, lambda s: s > 0.0),
                            "loss_weights" : And(list, lambda s: len(s) == 2),
                            "EarlyStopping_patience" : And(int, lambda s: s > 0),
                            "ReduceLROnPlateau_factor" : And(float, lambda s: 1.0 >= s >= 0.0),
                            "ReduceLROnPlateau_patience" : int,
                            "ReduceLROnPlateau_min_lr" : And(float, lambda s: s >= 0.0)},
        'firmware_config' : {"input_precision" : str,
                            "class_precision" : str,
                            "reg_precision": str,
                            "dense_precision": str,
                            "clock_period" : And(float, lambda s: 0.0 < s <= 10),
                            "fpga_part" : str,
                            "project_name" : str}
    })

    def build_model(self, input_shape, output_shape, adj_shape=None):

        initialise_tensorflow(self.run_config['num_threads'])
        gcn_units   = self.model_config['gcn_units']    # [32, 32]
        dense_units = self.model_config['dense_units']  # [16]
        regression_units = self.model_config['regression_units']  # [8]
        regression_output_units = self.model_config['regression_output_units']  # [1]
        bits        = self.quantization_config['quantizer_bits']
        integer     = self.quantization_config['quantizer_bits_int']
        n_classes   = output_shape[0]

        kq = quantized_bits(bits, integer, alpha=self.quantization_config['quantizer_alpha_val'],keep_negative=1)
        aq = quantized_relu(bits, 0)

        # ── Inputs ────────────────────────────────────────────────────────────
        X_in = tf.keras.Input(shape=input_shape, name='model_input', dtype=tf.float32)  # (B, 16, 20)
        A_in = tf.keras.Input(shape=adj_shape,   name='adj_input', dtype=tf.float32)    # (B, 16, 16)

        # ── QBatchNorm on input features ──────────────────────────────────────
        x = BatchNormalization(name='bn_input',dtype=tf.float32)(X_in)                # (B, 16, 20)

        # ── GCN layers: H' = act( A @ (H @ W) )  ────────────────────────────────
        for i, units in enumerate(gcn_units):
            # Step 1: per-node linear transform, weights shared across all 16 nodes
            # (B, 16, F_in) -> (B, 16, units)
            x = QDense(units, kernel_quantizer=kq, bias_quantizer=kq,
                    name=f'gcn_dense_{i}')(x)

            # Step 2: neighbourhood aggregation via adjacency matmul (Dot2D)
            # Dot(axes=[2,1]): contracts axis-2 of A (cols, N) with axis-1 of x (nodes, N)
            # = bmm(A, x) -> (B, 16, units)
            x = tf.keras.layers.Dot(axes=[2, 1], name=f'gcn_agg_{i}')([A_in, x])

            # Step 3: quantized activation
            x = QActivation(aq, name=f'gcn_act_{i}')(x)

        # ── GAP collapses node dimension → (B, units_last) ──────────────────────
        # x = QActivation(activation='quantized_bits(18,8)', name='act_pool')(x)
        agg = choose_aggregator(choice=self.model_config['aggregator'], name="pool")
        x = agg(x)
        
        # ── Shared Dense Trunk ────────────────────────────────────────────────
        for i, units in enumerate(dense_units):
            x = QDense(units, kernel_quantizer=kq, bias_quantizer=kq,
                    name=f'dense_{i}')(x)
            x = QActivation(aq, name=f'act_{i}')(x)

        # ── Head 1: Classification ───────────────────────────────────────────
        id_out = QDense(n_classes, kernel_quantizer=kq, bias_quantizer=kq,
                        name='jet_id_classification')(x)
        id_out = tf.keras.layers.Softmax(name='jet_id_output')(id_out)

        # ── Head 2: pT Regression ────────────────────────────────────────────
        for i, units in enumerate(regression_units):
            x = QDense(units, kernel_quantizer=kq, bias_quantizer=kq,
                            name=f'jet_pt_reg_{i}')(x)
            x = QActivation(aq, name=f'jet_pt_reg_act_{i}')(x)

        pt_out = QDense(
            regression_output_units,
            name='pT_output',
            kernel_quantizer=quantized_bits(
                self.quantization_config['pt_output_quantization'][0],
                self.quantization_config['pt_output_quantization'][1],
                alpha=self.quantization_config['quantizer_alpha_val'],
            ),
            bias_quantizer=quantized_bits(
                self.quantization_config['pt_output_quantization'][0],
                self.quantization_config['pt_output_quantization'][1],
                alpha=self.quantization_config['quantizer_alpha_val'],
            ),
            kernel_initializer=self.model_config['kernel_initializer'],
            )(x)

        self.jet_model = tf.keras.Model(
            inputs=[X_in, A_in],
            outputs=[id_out, pt_out],
            name='GNNModel'
        )
        self.jet_model.summary()

    def fit(self, X_train, y_train, pt_target_train, sample_weight,
            ADJ_train=None):
        """Override fit to pass both X and ADJ as inputs."""
        self.history = self.jet_model.fit(
            {'model_input': X_train, 'adj_input': ADJ_train},
            {
                self.loss_name + self.output_id_name: y_train,
                self.loss_name + self.output_pt_name: pt_target_train,
            },
            sample_weight=sample_weight,
            epochs=self.training_config['epochs'],
            batch_size=self.training_config['batch_size'],
            verbose=self.run_config['verbose'],
            validation_split=self.training_config['validation_split'],
            callbacks=self.callbacks,
            shuffle=True,
        )
    
    def firmware_convert(self, firmware_dir: str, build: bool = False):
        """Run hls4ml conversion for the GNN model (decomposed into native Keras ops)."""

        hls4ml_outdir = firmware_dir + '/' + self.firmware_config['project_name']
        os.system(f'rm -rf {hls4ml_outdir}')

        # Generate per-layer config
        config = hls4ml.utils.config_from_keras_model(self.jet_model, granularity='name')
        config['IOType'] = 'io_parallel'

        # Set precision for every layer explicitly
        for layer in self.jet_model.layers:
            layer_name  = layer.name
            layer_class = layer.__class__.__name__

            if layer_class in ['InputLayer']:
                config['LayerName'][layer_name]['Precision'] = \
                    self.firmware_config['input_precision']
                config['LayerName'][layer_name]['result'] = \
                    self.firmware_config['input_precision']

            elif layer_class in ['QBatchNormalization', 'BatchNormalization']:
                config['LayerName'][layer_name]['Precision'] = \
                    self.firmware_config['input_precision']
                config['LayerName'][layer_name]['result'] = \
                    self.firmware_config['input_precision']
                config['LayerName'][layer_name]['Trace'] = not build

            elif layer_class in ['QDense', 'Dense'] and layer_name not in ['pT_output']:
                config['LayerName'][layer_name]['Precision']['weight'] = \
                    self.firmware_config['dense_precision']
                config['LayerName'][layer_name]['Precision']['bias'] = \
                    self.firmware_config['dense_precision']
                config['LayerName'][layer_name]['Trace'] = not build

            elif layer_class == 'Dot':
                # Adjacency matmul — keep same precision as input
                config['LayerName'][layer_name]['Precision'] = \
                    self.firmware_config['input_precision']
                config['LayerName'][layer_name]['Trace'] = not build

            elif layer_class in ['Permute', 'Reshape', 'Flatten', 'Concatenate']:
                # Reshape ops — no weights, skip trace
                print(f'Skipping trace for: {layer_name}')

            else:
                config['LayerName'][layer_name]['Trace'] = not build

        # Output layer precisions
        config['LayerName']['jet_id_output']['Precision']['result'] = \
            self.firmware_config['class_precision']
        config['LayerName']['jet_id_output']['Implementation'] = 'latency'

        config['LayerName']['pT_output']['Precision']['result'] = \
            self.firmware_config['reg_precision']
        config['LayerName']['pT_output']['Precision']['bias'] = \
            self.firmware_config['reg_precision']
        config['LayerName']['pT_output']['Precision']['weight'] = \
            self.firmware_config['reg_precision']
        config['LayerName']['pT_output']['Implementation'] = 'latency'
        config["Model"]["TraceOutput"] = not build

        # Convert
        self.hls_jet_model = hls4ml.converters.convert_from_keras_model(
            self.jet_model,
            backend='Vitis',
            project_name=self.firmware_config['project_name'],
            clock_period=self.firmware_config['clock_period'],
            hls_config=config,
            output_dir=hls4ml_outdir,
            part=self.firmware_config['fpga_part'],
        )


        self.hls_jet_model.compile()

        print('Saving config as config.json ...')
        with open(hls4ml_outdir + '/config.json', 'w') as fp:
            json.dump(config, fp, indent=2)

        if build:
            self.hls_jet_model.build(csim=False, reset=True)

