import argparse
from train import train_model
from test import test_model
from visualize import visualize_filters
from configs import config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--test', action='store_true', help='Run testing')
    parser.add_argument('--visualize', action='store_true', help='Visualize filters')
    parser.add_argument('--model', type=str, default='resnet', choices=['customcnn', 'resnet'], 
                        help='Model type: customcnn or resnet')
    parser.add_argument('--use_bn', action='store_true', help='Use batch normalization')
    
    # 新增优化策略参数
    parser.add_argument('--channels', type=str, default=None, 
                        help='Channel sizes for CustomCNN (comma separated), e.g., "64,128"')
    parser.add_argument('--resnet_channels', type=str, default=None,
                        help='Channel sizes for ResNet (comma separated), e.g., "64,128,256"')
    parser.add_argument('--loss', type=str, default=None, 
                        choices=['cross_entropy', 'mse', 'l1'],
                        help='Loss function type')
    parser.add_argument('--l1', type=float, default=None,
                        help='L1 regularization coefficient')
    parser.add_argument('--activation', type=str, default=None,
                        choices=['relu', 'leaky_relu', 'elu'],
                        help='Activation function')
    parser.add_argument('--optimizer', type=str, default=None,
                        choices=['sgd', 'adam', 'rmsprop'],
                        help='Optimizer type')
    
    args = parser.parse_args()
    
    # 更新配置 (只更新提供的参数，保持默认值不变)
    config.model_type = args.model
    config.use_bn = args.use_bn
    
    if args.channels is not None:
        config.customcnn_channels = list(map(int, args.channels.split(',')))
    
    if args.resnet_channels is not None:
        config.resnet_channels = list(map(int, args.resnet_channels.split(',')))
    
    if args.loss is not None:
        config.loss_function = args.loss
    
    if args.l1 is not None:
        config.l1_lambda = args.l1
    
    if args.activation is not None:
        config.activation = args.activation
    
    if args.optimizer is not None:
        config.optimizer = args.optimizer
    
    if args.train:
        train_model()
    if args.test:
        test_model()
    if args.visualize:
        visualize_filters()