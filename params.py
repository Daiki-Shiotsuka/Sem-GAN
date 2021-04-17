import argparse
def create_parser():

    parser = argparse.ArgumentParser()

    # モデル用ハイパーパラメータ
    parser.add_argument('--image_height', type=int, default=256)
    parser.add_argument('--image_width', type=int, default=256)
    parser.add_argument('--a_channels', type=int, default=3, help="number of domain A's channels ")
    parser.add_argument('--b_channels', type=int, default=3, help="number of domain B's channels ")
    parser.add_argument('--d_conv_dim', type=int, default=64)
    parser.add_argument('--seg_channels', type=int, default=20, help="BDD has 20 classes")
    # トレーニング用ハイパーパラメータ
    parser.add_argument('--dataset_name', type=str, default='day2night', help='dataset name')
    parser.add_argument('--epochs', type=int, default=200, help='numebr of Epochs')
    parser.add_argument('--start_epoch', type=int, default=0, help='srart epoch')
    parser.add_argument('--decay_epoch', type=int, default=100, help='lr decayを実行し始めるEpoch数.')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='Dataloaderに使われるスレッド数.')
    parser.add_argument('--lr', type=float, default=0.0002, help='training rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='param of Adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='param of Adam')
    parser.add_argument('--n_cpu', type=int, default=8, help='batchを生成するときに使用するスレッド数.')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--lambda_cycle', type=int, default=10, help='lambda_cycle')

    #test
    parser.add_argument('--test_epoch', type=str, default='latest', help='checlpoint name used in test')
    parser.add_argument('--dataroot_dir', type=str, default='../data/')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_semgan')
    parser.add_argument('--sample_dir', type=str, default='samples_semgan')
    parser.add_argument('--sample_seg_dir', type=str, default='samples_segmentation')
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--log_step', type=int , default=20)
    parser.add_argument('--sample_every', type=int , default=200, help='サンプルをとる頻度、batch単位.')
    parser.add_argument('--checkpoint_every', type=int , default=1, help='Check pointをとる頻度、epoch単位.')
    return parser
