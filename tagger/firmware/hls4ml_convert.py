from argparse import ArgumentParser

from tagger.model.common import fromFolder

if __name__ == "__main__":

    parser = ArgumentParser()
    # parser.add_argument('-m', '--model_path', default='output2/baseline', help='Input model path for conversion')
    parser.add_argument('-m', '--model_path', default='output2/gnn_tagger_v2', help='Input model path for conversion')
    parser.add_argument(
        '-o', '--outpath', default='firmware', help='Jet tagger synthesized output directory'
    )

    args = parser.parse_args()

    # Load the model
    model = fromFolder(args.model_path)
    model.firmware_convert(args.outpath, build=False)
