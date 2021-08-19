import argparse
import pprint

import torch
from torch import optim
import torch.nn as nn

import torch_optimizer as custom_optim

from simple_nmt.data_loader import DataLoader
import simple_nmt.data_loader as data_loader

from simple_nmt.models.seq2seq import Seq2Seq
from simple_nmt.models.transformer import Transformer
from simple_nmt.models.rnnlm import LanguageModel

from simple_nmt.trainer import SingleTrainer
from simple_nmt.rl_trainer import MinimumRiskTrainingEngine
from simple_nmt.trainer import MaximumLikelihoodEstimationEngine


def define_argparser(is_continue=False):
    p = argparse.ArgumentParser()

    if is_continue:
        p.add_argument(
            '--load_fn',
            required=True,
            help='Model file name to continue.'
        )

    p.add_argument(
        '--model_fn',
        required=not is_continue,
        help='Model file name to save. Additional information would be annotated to the file name.'
    )
    p.add_argument(
        '--train',
        required=not is_continue,
        help='Training set file name except the extention. (ex: train.en --> train)'
    )
    p.add_argument(
        '--valid',
        required=not is_continue,
        help='Validation set file name except the extention. (ex: valid.en --> valid)'
    )
    p.add_argument(
        '--lang',
        required=not is_continue,
        help='Set of extention represents language pair. (ex: en + ko --> enko)'
    )
    p.add_argument(
        '--gpu_id',
        type=int,
        default=-1,
        help='GPU ID to train. Currently, GPU parallel is not supported. -1 for CPU. Default=%(default)s'
    )
    p.add_argument(
        '--off_autocast',
        action='store_true',
        help='Turn-off Automatic Mixed Precision (AMP), which speed-up training.',
    )

    p.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Mini batch size for gradient descent. Default=%(default)s'
    )
    p.add_argument(
        '--n_epochs',
        type=int,
        default=20,
        help='Number of epochs to train. Default=%(default)s'
    )
    p.add_argument(
        '--verbose',
        type=int,
        default=2,
        help='VERBOSE_SILENT, VERBOSE_EPOCH_WISE, VERBOSE_BATCH_WISE = 0, 1, 2. Default=%(default)s'
    )
    p.add_argument(
        '--init_epoch',
        required=is_continue,
        type=int,
        default=1,
        help='Set initial epoch number, which can be useful in continue training. Default=%(default)s'
    )

    p.add_argument(
        '--max_length',
        type=int,
        default=100,
        help='Maximum length of the training sequence. Default=%(default)s'
    )
    p.add_argument(
        '--dropout',
        type=float,
        default=.2,
        help='Dropout rate. Default=%(default)s'
    )
    p.add_argument(
        '--word_vec_size',
        type=int,
        default=512,
        help='Word embedding vector dimension. Default=%(default)s'
    )
    p.add_argument(
        '--hidden_size',
        type=int,
        default=768,
        help='Hidden size of LSTM. Default=%(default)s'
    )
    p.add_argument(
        '--n_layers',
        type=int,
        default=4,
        help='Number of layers in LSTM. Default=%(default)s'
    )
    p.add_argument(
        '--max_grad_norm',
        type=float,
        default=5.,
        help='Threshold for gradient clipping. Default=%(default)s'
    )
    p.add_argument(
        '--iteration_per_update',
        type=int,
        default=1,
        help='Number of feed-forward iterations for one parameter update. Default=%(default)s'
    )

    p.add_argument(
        '--lr',
        type=float,
        default=1.,
        help='Initial learning rate. Default=%(default)s',
    )

    p.add_argument(
        '--lr_step',
        type=int,
        default=1,
        help='Number of epochs for each learning rate decay. Default=%(default)s',
    )
    p.add_argument(
        '--lr_gamma',
        type=float,
        default=.5,
        help='Learning rate decay rate. Default=%(default)s',
    )
    p.add_argument(
        '--lr_decay_start',
        type=int,
        default=10,
        help='Learning rate decay start at. Default=%(default)s',
    )

    p.add_argument(
        '--use_adam',
        action='store_true',
        help='Use Adam as optimizer instead of SGD. Other lr arguments should be changed.',
    )
    p.add_argument(
        '--use_radam',
        action='store_true',
        help='Use rectified Adam as optimizer. Other lr arguments should be changed.',
    )

    p.add_argument(
        '--rl_lr',
        type=float,
        default=.01,
        help='Learning rate for reinforcement learning. Default=%(default)s'
    )
    p.add_argument(
        '--rl_n_samples',
        type=int,
        default=1,
        help='Number of samples to get baseline. Default=%(default)s'
    )
    p.add_argument(
        '--rl_n_epochs',
        type=int,
        default=10,
        help='Number of epochs for reinforcement learning. Default=%(default)s'
    )
    p.add_argument(
        '--rl_n_gram',
        type=int,
        default=6,
        help='Maximum number of tokens to calculate BLEU for reinforcement learning. Default=%(default)s'
    )
    p.add_argument(
        '--rl_reward',
        type=str,
        default='gleu',
        help='Method name to use as reward function for RL training. Default=%(default)s'
    )

    p.add_argument(
        '--use_transformer',
        action='store_true',
        help='Set model architecture as Transformer.',
    )
    p.add_argument(
        '--n_splits',
        type=int,
        default=8,
        help='Number of heads in multi-head attention in Transformer. Default=%(default)s',
    )

    config = p.parse_args()

    return config


def get_model(input_size, output_size, config):
    if config.use_transformer:
        model = Transformer(
            input_size,                     # Source vocabulary size
            config.hidden_size,             # Transformer doesn't need word_vec_size.
            output_size,                    # Target vocabulary size
            n_splits=config.n_splits,       # Number of head in Multi-head Attention.
            n_enc_blocks=config.n_layers,   # Number of encoder blocks
            n_dec_blocks=config.n_layers,   # Number of decoder blocks
            dropout_p=config.dropout,       # Dropout rate on each block
        )
    else:
        model = Seq2Seq(
            input_size,
            config.word_vec_size,           # Word embedding vector size
            config.hidden_size,             # LSTM's hidden vector size
            output_size,
            n_layers=config.n_layers,       # number of layers in LSTM
            dropout_p=config.dropout        # dropout-rate in LSTM
        )

    return model


def get_crit(output_size, pad_index):
    # Default weight for loss equals to 1, but we don't need to get loss for PAD token.
    # Thus, set a weight for PAD to zero.
    # output_size : is number : |V_t|
    loss_weight = torch.ones(output_size)
        # 패딩이 되어있는곳은 맞춰도 점수를 주고 싶지 않은것.
    '''      ___________
            |   |   |   |
            -------------
            |   |   |||||
            |-----------|
            |   |||||||||
            -------------    
        
    '''
    loss_weight[pad_index] = 0.
        # 패드인덱스를 받아와서 거기다가 0을 할당해서, 맞춰도 점수를 주지마.
    '''???????????????????????????????????????????????????????????'''
        # pad_index를 가져와야하는데, main함수에서 보면 data_loader의 PAD ; 1을 가져왔으. 왜?
    # Instead of using Cross-Entropy loss,
    # we can use Negative Log-Likelihood(NLL) loss with log-probability.
    crit = nn.NLLLoss(
        weight=loss_weight,
        reduction='sum'
    )

    return crit


def get_optimizer(model, config):
    if config.use_adam:
        if config.use_transformer:
            optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(.9, .98))
        else: # case of rnn based seq2seq.
            optimizer = optim.Adam(model.parameters(), lr=config.lr)
    elif config.use_radam:
        optimizer = custom_optim.RAdam(model.parameters(), lr=config.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=config.lr) # initial lr = 1, 그리고 그래디언트가 폭주할거 같으면 클립핑을 함(?)-> 요놈은 한번도 안해봄.

    return optimizer


def get_scheduler(optimizer, config):
    '''Adam 같은 경우는 쓸 필요가 없고, SGD같은 경우 써야한데'''
    '''
         LR |
          1 |-----------------
            |
            |                  --
            |                      -- ... 10번부터는 lr을 반씩 줄여가면서 진행
            |_____________________________________
               1  2  3  4 ... 9 10 11 12         epochs
    
    '''
    if config.lr_step > 0:
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
                                                    optimizer,
                                                    milestones=[i for i in range(
                                                                                max(0, config.lr_decay_start - 1), # 내가 원하는 시작 에폭부터
                                                                                (config.init_epoch - 1) + config.n_epochs, # 내가 원하는 끝나는 에폭까지
                                                                                config.lr_step # 1
                                                                                )],
                                                    gamma=config.lr_gamma, # 0.5씩 작아짐.
                                                    last_epoch=config.init_epoch - 1 if config.init_epoch > 1 else -1,
                                                )
    else:
        lr_scheduler = None

    return lr_scheduler


# 결국 이친구가 돌아갈텐데
def main(config, model_weight=None, opt_weight=None):

    def print_config(config):
        '''
        이쁘게 프린트 해주기
        https://wikidocs.net/105471
        '''
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(config))
    print_config(config)


    loader = DataLoader(
        config.train,                           # Train file name except extention, which is language.
        config.valid,                           # Validation file name except extension.
        (config.lang[:2], config.lang[-2:]),    # Source and target language. // 예) en, ko -> enko로 등록을 해야함.
        batch_size=config.batch_size,
        device=-1,                              # Lazy loading
        max_length=config.max_length,           # Loger sequence will be excluded.
        dsl=False,                              # Turn-off Dual-supervised Learning mode.
    )

    input_size, output_size = len(loader.src.vocab), len(loader.tgt.vocab)
        # loader는 loader.src.vocab, loader.train_iter.src[0], [1]등을 갖고 있음.
        # input언어의 vocab size, output언어의 vocab size
    model = get_model(input_size, output_size, config)
    crit = get_crit(output_size, data_loader.PAD) # 얘 이상해
    '''?????????????????????????????????????????????????????'''

    # continue_train.py에서 결정되는부분임.
    if model_weight is not None:    
        model.load_state_dict(model_weight)

    # Pass models to GPU device if it is necessary.
    if config.gpu_id >= 0:
        model.cuda(config.gpu_id)
        crit.cuda(config.gpu_id)

    optimizer = get_optimizer(model, config)

    if opt_weight is not None and (config.use_adam or config.use_radam):
        optimizer.load_state_dict(opt_weight)

    lr_scheduler = get_scheduler(optimizer, config)

    if config.verbose >= 2:
        print(model)
        print(crit)
        print(optimizer)

    # Start training. This function maybe equivalant to 'fit' function in Keras.
    mle_trainer = SingleTrainer(MaximumLikelihoodEstimationEngine, config)
    mle_trainer.train(
        model,
        crit,
        optimizer,
        train_loader=loader.train_iter,
        valid_loader=loader.valid_iter,
        src_vocab=loader.src.vocab,
        tgt_vocab=loader.tgt.vocab,
        n_epochs=config.n_epochs,
        lr_scheduler=lr_scheduler,
    )


    
    # 강화학습 부분
    if config.rl_n_epochs > 0:
        optimizer = optim.SGD(model.parameters(), lr=config.rl_lr)
        mrt_trainer = SingleTrainer(MinimumRiskTrainingEngine, config)

        mrt_trainer.train(
            model,
            None, # We don't need criterion for MRT.
            optimizer,
            train_loader=loader.train_iter,
            valid_loader=loader.valid_iter,
            src_vocab=loader.src.vocab,
            tgt_vocab=loader.tgt.vocab,
            n_epochs=config.rl_n_epochs,
        )


if __name__ == '__main__':
    # 이런식으로 정의하나보다.. 배웠다.
    config = define_argparser()
    main(config)
