import argparse
import sys
import codecs
from operator import itemgetter

import torch

from simple_nmt.data_loader import DataLoader
import simple_nmt.data_loader as data_loader
from simple_nmt.models.seq2seq import Seq2Seq
from simple_nmt.models.transformer import Transformer


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument(
        '--model_fn',
        required=True,
        help='Model file name to use'
    )
    p.add_argument(
        '--gpu_id',
        type=int,
        default=-1,
        help='GPU ID to use. -1 for CPU. Default=%(default)s'
    )

    p.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Mini batch size for parallel inference. Default=%(default)s'
    )
    p.add_argument(
        '--max_length',
        type=int,
        default=255,
        help='Maximum sequence length for inference. Default=%(default)s'
    )
    p.add_argument(
        '--n_best',
        type=int,
        default=1,
        help='Number of best inference result per sample. Default=%(default)s'
    )
    p.add_argument(
        '--beam_size', # beam search 안할 거면 1
        type=int,
        default=5,
        help='Beam size for beam search. Default=%(default)s'
    )
    p.add_argument(
        '--lang',
        type=str,
        default=None,
        help='Source language and target language. Example: enko'
    )
    p.add_argument(
        '--length_penalty',
        type=float,
        default=1.2,
        help='Length penalty parameter that higher value produce shorter results. Default=%(default)s',
    )

    config = p.parse_args()

    return config


def read_text(batch_size=128):
    # This method gets sentences from standard input and tokenize those.
    '''
    iterator처럼 행동하는 함수., mecab에 bpe까지 한것을 넣어줘야함.

    line은 문장들이 들어가 있는데, 문장과 문장 사이가 ' '로 구분되어 있어서 split(' ')을 통해 구분한다.

    '''
    lines = []

    sys.stdin = codecs.getreader("utf-8")(sys.stdin.detach())

    for line in sys.stdin:
        if line.strip() != '':
            lines += [line.strip().split(' ')]

        if len(lines) >= batch_size: # 배치사이즈만큼 차면 line을 내뱉고 // 라인을 비워줘.
            yield lines
            lines = []

    if len(lines) > 0:
        yield lines


def to_text(indice, vocab):
    # This method converts index to word to show the translation result.
    lines = []

    for i in range(len(indice)):
        line = []
        for j in range(len(indice[i])):
            index = indice[i][j]

            if index == data_loader.EOS:
                # line += ['<EOS>']
                break
            else:
                line += [vocab.itos[index]] # itos : index to string

        line = ' '.join(line)
        lines += [line]

    return lines


def is_dsl(train_config):
    # return 'dsl_lambda' in vars(train_config).keys()
    return not ('rl_n_epochs' in vars(train_config).keys())


def get_vocabs(train_config, config, saved_data):
    if is_dsl(train_config):
        assert config.lang is not None

        if config.lang == train_config.lang:
            is_reverse = False
        else:
            is_reverse = True

        if not is_reverse:
            # Load vocabularies from the model.
            src_vocab = saved_data['src_vocab']
            tgt_vocab = saved_data['tgt_vocab']
        else:
            src_vocab = saved_data['tgt_vocab']
            tgt_vocab = saved_data['src_vocab']

        return src_vocab, tgt_vocab, is_reverse
    else:
        # Load vocabularies from the model.
        src_vocab = saved_data['src_vocab']
        tgt_vocab = saved_data['tgt_vocab']

    return src_vocab, tgt_vocab, False


def get_model(input_size, output_size, train_config, is_reverse=False):
    # Declare sequence-to-sequence model.
    if 'use_transformer' in vars(train_config).keys() and train_config.use_transformer:
        model = Transformer(
            input_size,
            train_config.hidden_size,
            output_size,
            n_splits=train_config.n_splits,
            n_enc_blocks=train_config.n_layers,
            n_dec_blocks=train_config.n_layers,
            dropout_p=train_config.dropout,
        )
    else:
        model = Seq2Seq(
            input_size,
            train_config.word_vec_size,
            train_config.hidden_size,
            output_size,
            n_layers=train_config.n_layers,
            dropout_p=train_config.dropout,
        )

    if is_dsl(train_config):
        if not is_reverse:
            model.load_state_dict(saved_data['model'][0])
        else:
            model.load_state_dict(saved_data['model'][1])
    else:
        model.load_state_dict(saved_data['model'])  # Load weight parameters from the trained model.
    model.eval()  # We need to turn-on the evaluation mode, which turns off all drop-outs.
    # eavl 안하면 dropout같은거 실행되서 성능 진짜 낮아짐

    return model


if __name__ == '__main__':
    # 뭐야 이게...
    '''??????????????????????????????'''
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())


    config = define_argparser()

    # Load saved model. # saved from trainer.py
    saved_data = torch.load(
        config.model_fn,
        map_location='cpu', # map_location = 'cpu' if config.gpu_id < 0 else 'cuda:%d' % config.gpu_id
    )


    # Load configuration setting in training.
    train_config = saved_data['config']

    src_vocab, tgt_vocab, is_reverse = get_vocabs(train_config, config, saved_data)
    # is_reverse는 dual learning할때 필요함. // dsl아니니까 그냥 저장된 vocab 불러옴.


    # Initialize dataloader, but we don't need to read training & test corpus.
    # What we need is just load vocabularies from the previously trained model.
    loader = DataLoader()
    loader.load_vocab(src_vocab, tgt_vocab)
    '''
    데이터를 불러오지 않아도.. 보캡 빌드했던걸 불러올수 있다는거지.. 잘와닿지는 않는다.
    '''

    input_size, output_size = len(loader.src.vocab), len(loader.tgt.vocab)
    model = get_model(input_size, output_size, train_config, is_reverse)

    # Put models to device if it is necessary.
    if config.gpu_id >= 0:
        model.cuda(config.gpu_id)

    with torch.no_grad():
        # Get sentences from standard input.
        for lines in read_text(batch_size=config.batch_size): # 배치 사이즈만큼 읽어옴.
            # Since packed_sequence must be sorted by decreasing order of length, // 팩트 시퀀스를 쓰려면 각 문장별로 length를 알려줘야함.
            # sorting by length in mini-batch should be restored by original order. // 문장 길이순서대로 있어야 한다. 가장 긴게 맨위에 있어야함.
            # Therefore, we need to memorize the original index of the sentence.
            lengths         = [len(line) for line in lines]
            original_indice = [i for i in range(len(lines))] # 소팅을 하면 문자의 순서가 바뀔테니 그걸 가지고 있을 index

            sorted_tuples = sorted(
                zip(lines, lengths, original_indice),
                key=itemgetter(1),
                reverse=True,
            )
                # 결과물은
            '''
            # 결과물은

            (line, lengths, original_indice) * lines의 길이만큼 튜플들이 리스트안에 묶여있음.
            '''
            sorted_lines    = [sorted_tuples[i][0] for i in range(len(sorted_tuples))]
            lengths         = [sorted_tuples[i][1] for i in range(len(sorted_tuples))]
            original_indice = [sorted_tuples[i][2] for i in range(len(sorted_tuples))]

            # Converts string to list of index.
            x = loader.src.numericalize( # Field에 있는 numericalize메서드 : word2idx
                loader.src.pad(sorted_lines), 
                    # Field에 pad라는 메서드가 있나봐. 모지란 부분을 pad로 채워줌.
                    # sorted_lines는 아직 raw tokens이야. 사람이 읽을 수 있는 토큰들이야.
                device='cuda:%d' % config.gpu_id if config.gpu_id >= 0 else 'cpu'
            )
                # |x| = (batch_size, length) # one hot vector [42, 332, 15, 13, 1, 1, 1, 1, 1, ...]
                # 저걸 수행하면 

            if config.beam_size == 1:
                y_hats, indice = model.search(x)
                # |y_hats| = (b, L, output_size)
                # |indice| = (b, L)

                output = to_text(indice, loader.tgt.vocab)
                sorted_tuples = sorted(zip(output, original_indice), key=itemgetter(1)) # output은 번역이 된 결과물인데, 이건 들어올때랑 순서가 달라진 결과물이라서, 다시 되돌리는 과정이 필요하다.
                output = [sorted_tuples[i][0] for i in range(len(sorted_tuples))]

                sys.stdout.write('\n'.join(output) + '\n')
            else:
                # Take mini-batch parallelized beam search.
                batch_indice, _ = model.batch_beam_search(
                    x,
                    beam_size=config.beam_size,
                    max_length=config.max_length,
                    n_best=config.n_best,
                    length_penalty=config.length_penalty,
                )

                # Restore the original_indice.
                output = []
                for i in range(len(batch_indice)):
                    output += [to_text(batch_indice[i], loader.tgt.vocab)]
                sorted_tuples = sorted(zip(output, original_indice), key=itemgetter(1))
                output = [sorted_tuples[i][0] for i in range(len(sorted_tuples))]

                for i in range(len(output)):
                    sys.stdout.write('\n'.join(output[i]) + '\n')
