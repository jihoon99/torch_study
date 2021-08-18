import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import simple_nmt.data_loader as data_loader
from simple_nmt.search import SingleBeamSearchBoard


class Attention(nn.Module):

    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.linear = nn.Linear(hidden_size, hidden_size, bias=False) # 맨처음에 projection needed for 가중치 refer to encoder part
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, h_src, h_t_tgt, mask=None):
        # |h_src| = (batch_size, length, hidden_size) - 인코더의 모든 히든 스테잇
        # |h_t_tgt| = (batch_size, 1, hidden_size) - 디코더의 히든 스테잇
        # |mask| = (batch_size, length) - src의 마스킹할 정보

        query = self.linear(h_t_tgt)                     # [B,1,H] * [B,H,H] = [B,1,H]
        # |query| = (batch_size, 1, hidden_size)

        weight = torch.bmm(query, h_src.transpose(1, 2)) # [B,1,H] * [B, H, L] => [B, 1, L] // bmm : batch multiplication
        # |weight| = (batch_size, 1, length)
        if mask is not None:
            # Set each weight as -inf, if the mask value equals to 1.
            # Since the softmax operation makes -inf to 0, 
            # masked weights would be set to 0 after softmax operation.
            # Thus, if the sample is shorter than other samples in mini-batch,
            # the weight for empty time-step would be set to 0.
            weight.masked_fill_(mask.unsqueeze(1), -float('inf')) # mask가 있는 부분에 -float('inf')를 넣어줘
        weight = self.softmax(weight)

        context_vector = torch.bmm(weight, h_src)        # [B,1,L]*[B,L,H] -> [B,1,H]
        # |context_vector| = (batch_size, 1, hidden_size)
        # 해석으 해보면, 샘플 데이터에서, 디코더의 시점에서, 어텐션을 적용한 컨텐스트 벡터

        return context_vector


class Encoder(nn.Module):

    def __init__(self, word_vec_size, hidden_size, n_layers=4, dropout_p=.2):
        super(Encoder, self).__init__()

        # Be aware of value of 'batch_first' parameter.
        # Also, its hidden_size is half of original hidden_size,
        # because it is bidirectional.
        self.rnn = nn.LSTM(
            word_vec_size, # input shape
            int(hidden_size / 2), # bidirectional 할 것이기 때문에, 나누기 2를 했다. -> 만약 소수점이 되버리면?
            num_layers=n_layers, # stacking LSTM
            dropout=dropout_p,
            bidirectional=True,
            batch_first=True, # batch의 쉐입이 첫번째가 아니라서 앞으로 오게 강제함
        )

    def forward(self, emb):
        # |emb| = (batch_size, length, word_vec_size)

        if isinstance(emb, tuple): # 임베딩 타입이 튜플이니? 
            x, lengths = emb
            x = pack(x, lengths.tolist(), batch_first=True) # https://simonjisu.github.io/nlp/2018/07/05/packedsequence.html
            # input : input은 T*B*(*) /T는 가장긴 시퀀스/B는 배치사이즈,/(*)은 dim
            # length : list of sequence lengths of each batch element


            # Below is how pack_padded_sequence works.
            # As you can see,
            # PackedSequence object has information about mini-batch-wise information,
            # not time-step-wise information.
            # 
            # a = [torch.tensor([1,2,3]), 
            #      torch.tensor([3,4])]

            # b = torch.nn.utils.rnn.pad_sequence(a, batch_first=True)
            # >>>>
            # tensor([[ 1,  2,  3],
            #         [ 3,  4,  0]])
            # torch.nn.utils.rnn.pack_padded_sequence(b, batch_first=True, lengths=[3,2]
            # >>>>PackedSequence(data=tensor([ 1,  3,  2,  4,  3]), batch_sizes=tensor([ 2,  2,  1]))
        
        else:
            x = emb

        y, h = self.rnn(x)
        # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        # y: containing the output features (h_t) from the last layer of the LSTM, for each t // 모든 t시점에서 나온 hidden
        # h: (containing the final hidden state for each element in the batch // containing the final cell state for each element in the batch.)
        # |y| = (batch_size, length, hidden_size) : hidden_size * 2(정방향) / 2(역방향)
        # |h[0]| = (num_layers * 2, batch_size, hidden_size / 2)
                # num_layer * num_direction
                # 바이다이렉셔널이라 num_layers * 2임 // ?배치사이즈 // ?(hidden_size / 2)

        if isinstance(emb, tuple):
            y, _ = unpack(y, batch_first=True) # 위에 packedsequence가 들어가있으면 풀어줘야 하기 때문에 씀.
        
        # y : [b, n, h]
        # h : [l*2, b, h/2], [l*2, b, h/2]
        return y, h


'''         y1            y2
            |             |
        |-------|     |-------|
        |  RNN  | ->  |  RNN  | -> h            y1,y2...는 y에 나옴 // h : final hidden만 할당이 됨.
        |_______|     |_______|
            |             |
            x1            x2
'''



class Decoder(nn.Module):

    def __init__(self, word_vec_size, hidden_size, n_layers=4, dropout_p=.2):
        super(Decoder, self).__init__()

        # Be aware of value of 'batch_first' parameter and 'bidirectional' parameter.
        self.rnn = nn.LSTM(
            word_vec_size + hidden_size, # input feeding? 을 해줄거기 때문에(concat) 차원이 늘어난다.
            hidden_size,
            num_layers=n_layers,
            dropout=dropout_p,
            bidirectional=False,
            batch_first=True,
        )

    def forward(self, emb_t, h_t_1_tilde, h_t_1):
        '''
        추론할때나, input feeding을 해줄것이기 때문에, 한스텝씩 들어올거야.
        h_t_1_tilde : 저번에 예측한 hidden의 정보값. before softmax
        h_t_1 : h_{t-1} = [h_{t-1}, c_{t-1}]   tuple임. // 전 스텝의 hidden값. //  [n layer, b, h]라는데(?)
        
        # |emb_t| = (b, 1, word_vec_size)
        # |h_t_1_tilde| = (b, 1, h)
        # |h_t_1| = [(n_l, b, h),(n_l, b, h)] : t-1 시점 전의 모든 히든들..같음 not sure
        '''
        batch_size = emb_t.size(0) # [batch]
        hidden_size = h_t_1[0].size(-1) # [hidden]

        if h_t_1_tilde is None:
            # If this is the first time-step, 이제 막 디코더가 시작한것임.
            h_t_1_tilde = emb_t.new(batch_size, 1, hidden_size).zero_() # .new -> 텐서는 디바이스와, 타입이 같아야 arithmetic이 가능한데,.. 그러면 두번을 설정해 줘야함. 귀찮자나..
                                                                                    # 가장 간단하게 하는 방법이. 저 텐서와 같은 디바이스, 타입인놈을 만들어줘. 하는게 new이다.
                                                                        # .zero_() -> inplace 연산이다.

        # Input feeding trick.
        x = torch.cat([emb_t, h_t_1_tilde], dim=-1) # [b, 1, w + h]

        # Unlike encoder, decoder must take an input for sequentially.
        y, h = self.rnn(x, h_t_1) # h_t_1 : [(n_l, b, h), (n_l, b, h)] : 이전 시점의 hidden, context tensors, it is 0 when it's not provided.
            # y : [b, 1, h] // h: [l, b, h],[l,b,h]
            # |decoder_output| = (b, 1, h)
            # |decoder_hidden| = (n, b, h), (n,b,h)
        return y, h


class Generator(nn.Module):

    def __init__(self, hidden_size, output_size):
        super(Generator, self).__init__()

        self.output = nn.Linear(hidden_size, output_size) # output_size : word vec size
        self.softmax = nn.LogSoftmax(dim=-1) # logSoftmax를 함. (왜?)

    def forward(self, x):
        # |x| = (batch_size, length, hidden_size) : 학습할때는 length길이 만큼 한번에 들어감. 왜냐하면 teacher forcing이니까.

        y = self.softmax(self.output(x)) # linear에 한번 통과한다. 그러면 사이즈가 word sz로 바뀜.
        # |y| = (batch_size, length, output_size)

        # Return log-probability instead of just probability. : 미니배치, 각 샘플별, 각 단어별, 로그 확률값이 리턴이됨.
        return y


class Seq2Seq(nn.Module):

    def __init__(
        self,
        input_size,
        word_vec_size,
        hidden_size,
        output_size,
        n_layers=4,
        dropout_p=.2
    ):

        '''
        input_size : input언어의 vocab size
        word_vec_size : embed size
        hidden_size : hidden sz
        output_size : target언어의 vocab size
        '''

        self.input_size = input_size
        self.word_vec_size = word_vec_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        super(Seq2Seq, self).__init__()


        # 임베드 정의
        self.emb_src = nn.Embedding(input_size, word_vec_size)
        self.emb_dec = nn.Embedding(output_size, word_vec_size)

        # 
        self.encoder = Encoder(
            word_vec_size, hidden_size,
            n_layers=n_layers, dropout_p=dropout_p,
        )
        self.decoder = Decoder(
            word_vec_size, hidden_size,
            n_layers=n_layers, dropout_p=dropout_p,
        )
        self.attn = Attention(hidden_size)

        # attn에서 나온 context vec와 // decoder의 output하고 -> h_tilde
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.tanh = nn.Tanh() # 위 concat에 씌어줄 activation fn
        self.generator = Generator(hidden_size, output_size)

    def generate_mask(self, x, length):
        '''
        x : [bs, n]
        length : [bs,] such as [4,3,1]
        '''
        mask = []

        max_length = max(length)
        for l in length:
            if max_length - l > 0:
                # If the length is shorter than maximum length among samples, 
                # set last few values to be 1s to remove attention weight.
                mask += [torch.cat([x.new_ones(1, l).zero_(),
                                    x.new_ones(1, (max_length - l))
                                    ], dim=-1)]
            else:
                # If the length of the sample equals to maximum length among samples, 
                # set every value in mask to be 0.
                mask += [x.new_ones(1, l).zero_()]

        mask = torch.cat(mask, dim=0).bool() # [[4,4], [4,4], [4,4]] -> [3, 4]짜리 텐서로 flatten

        '''
            length 에) 아래와 같은 텐서가 있을때 

            --- --- --- ---
            |  |   |   |  |  [4,
            ___ ___ ___ ___
            |  |   |   ||||   3,
            --- --- --- ---
            |   ||| ||| |||   1] 라는 x_length모양이 있을것임.
            --- --- --- ---

            --- --- --- ---
            | 0|  0|  0| 0|  
            ___ ___ ___ ___
            | 0|  0|  0| 1|  
            --- --- --- ---
            | 0| 1| | 1| 1|   
            --- --- --- ---
            으로 나오게 한다.
        '''
        return mask



    def merge_encoder_hiddens(self, encoder_hiddens):

        '''
        for loop을 하여 속도가 안좋음.
        '''
        new_hiddens = []
        new_cells = []

        hiddens, cells = encoder_hiddens
            # encoder_hiddens는 hiddens와 cell_state두개를 갖고 있음.
            # hiddens : [2*layers, batch, hidden/2]

        # i-th and (i+1)-th layer is opposite direction.
        # Also, each direction of layer is half hidden size.
        # Therefore, we concatenate both directions to 1 hidden size layer.
        for i in range(0, hiddens.size(0), 2): # 0~2*layers만큼 for문을 돌림.
            new_hiddens += [torch.cat([hiddens[i], hiddens[i + 1]], dim=-1)] # 0,1 // 2,3 // 이런식으로 묶어서 넣어줌.
                # new_hiddens : [bs, hs/2*2] -> [bs, hs]
            new_cells += [torch.cat([cells[i], cells[i + 1]], dim=-1)]

        new_hiddens, new_cells = torch.stack(new_hiddens), torch.stack(new_cells)
                # new_hiddens : [layers, bs, hs]
                # new_cells : [layers, bs, hs]
        return (new_hiddens, new_cells)


    def fast_merge_encoder_hiddens(self, encoder_hiddens):
        '''
        parallel하게 해보자
        encoder : [l*2, b, h/2], [l*2, b, h/2]
        '''
        # Merge bidirectional to uni-directional
        # (layers*2, bs, hs/2) -> (layers, bs, hs).
        # Thus, the converting operation will not working with just 'view' method.
        h_0_tgt, c_0_tgt = encoder_hiddens # 두개 모두 [2layer, b, h/2]
        batch_size = h_0_tgt.size(1)

        # contiguous : 메모리상에 잘 붙어있게 선언하는것.
        # transpose까지 하면 : [b, 2layer, h/2]
        # view : [b, -1, hs] --> [b, layer, h]
        # transpose : [layer, b, h]
        h_0_tgt = h_0_tgt.transpose(0, 1).contiguous().view(batch_size,
                                                            -1,
                                                            self.hidden_size
                                                            ).transpose(0, 1).contiguous()
        c_0_tgt = c_0_tgt.transpose(0, 1).contiguous().view(batch_size,
                                                            -1,
                                                            self.hidden_size
                                                            ).transpose(0, 1).contiguous()
        # You can use 'merge_encoder_hiddens' method, instead of using above 3 lines.
        # 'merge_encoder_hiddens' method works with non-parallel way.
        # h_0_tgt = self.merge_encoder_hiddens(h_0_tgt)

        # |h_src| = (batch_size, length, hidden_size)
        # |h_0_tgt| = (n_layers, batch_size, hidden_size)
        # [l, b, h], [l, b, h]
        return h_0_tgt, c_0_tgt

    def forward(self, src, tgt):

        '''
        학습할때는 teacher forcing을 할 것임.

        src : input sentence = [bs, n, V_i]
        tgt : target sentence = [bs, m, V_t]
        '''
        # output = [bs, m, V_t]

        batch_size = tgt.size(0)

        mask = None
        x_length = None
        if isinstance(src, tuple):
            x, x_length = src
            '''
            x_length에서 마스크 정보가 주어지면 generate_mask를 하라고 했음.
            '''
            # Based on the length information, gererate mask to prevent that
            # shorter sample has wasted attention.
            mask = self.generate_mask(x, x_length) 
            # //x : [bs, n] // x_length : [bs,] // mask : [bs, n]
            # |mask| = (batch_size, length)
            '''
            length 에) 아래와 같은 텐서가 있을때 

            --- --- --- ---
            |  |   |   |  |  [4,
            ___ ___ ___ ___
            |  |   |   ||||   3,
            --- --- --- ---
            |   ||| ||| |||   1] 라는 x_length모양이 있을것임.
            --- --- --- ---
            
            즉 [4,3,1]이 들어가 있음. 여기서 배치사이즈는 3임.
            '''

        else:
            x = src

        if isinstance(tgt, tuple):
            tgt = tgt[0]


        # Get word embedding vectors for every time-step of input sentence.
        emb_src = self.emb_src(x) # |emb_src| = (b, n, emb)

        # The last hidden state of the encoder would be a initial hidden state of decoder.
        h_src, h_0_tgt = self.encoder((emb_src, x_length)) # packed_padded_sequence로 처리를 함.
            # |h_src| = (b, n, h) : 인코더의 모든 t시점에서의 히든스테이트
            # |h_0_tgt| = [l*2, b, h/2], [l*2, b, h/2] : 인코더에서 레이어마다 나온 마지막 히든스테이트(컨텍스트)
                # -> 여기서 이친구를 decoder의 init hidden으로 넣어줘야 하는데,feature가 h/2임. 이걸 h로 변환해줘야함.

        h_0_tgt = self.fast_merge_encoder_hiddens(h_0_tgt)
            # merge_encoder_hidden부터 살펴보자
            # [l, b, h], [l, b, h]

        # teacher forcing이기 때문에 정답을 한꺼번에 만들어.
        emb_tgt = self.emb_dec(tgt)
            # |emb_tgt| = (b, l, emb)
        h_tilde = [] # 여기도 한방에 들어갈거야.

        h_t_tilde = None # 첫번째 타임스텝에서는 전에 있던 h_t_tilde는 없다.
        decoder_hidden = h_0_tgt # ([layer, bs, hs], [layer, bs, hs])

        # Run decoder until the end of the time-step.
        for t in range(tgt.size(1)): # length of sentence
            # Teacher Forcing: take each input from training set,
            # not from the last time-step's output.
            # Because of Teacher Forcing,
            # training procedure and inference procedure becomes different.
            # Of course, because of sequential running in decoder,
            # this causes severe bottle-neck.
            emb_t = emb_tgt[:, t, :].unsqueeze(1) # 한 단어씩 번갈아가면서 들어간다. // unsqueeze : 특정 차원에 차원을 추가한다.
                # 인덱싱할 경우 [b, l, emb] -> [b,emb]되버릴 수 있다. 따라서 명시적으로 그냥 선언하자.
            # |emb_t| = (batch_size, 1, word_vec_size)
            # |h_t_tilde| = (batch_size, 1, hidden_size)

            decoder_output, decoder_hidden = self.decoder(emb_t, # 현시점의 단어.
                                                          h_t_tilde, # 지난 타임 스텝의 틸다
                                                          decoder_hidden # [l, b, h], [l, b, h]
                                                          )
            # |decoder_output| = (b, 1, h)
            # |decoder_hidden| = (n, b, h), (n,b,h)


            context_vector = self.attn(h_src, decoder_output, mask)
            # |context_vector| = (batch_size, 1, hidden_size)

            h_t_tilde = self.tanh(self.concat(torch.cat([decoder_output,
                                                         context_vector
                                                         ], dim=-1)))
            # |h_t_tilde| = (batch_size, 1, hidden_size)
            # self.concat -> 2h, h

            h_tilde += [h_t_tilde]

        h_tilde = torch.cat(h_tilde, dim=1)
            # h_tilde = (b, 1, h)
            # concat on dim 1 => (b, m, h)
            # |h_tilde| = (b, length, h)

        y_hat = self.generator(h_tilde)
        # |y_hat| = (b, length, output_size:vocab_size)

        return y_hat



    def search(self, src, is_greedy=True, max_length=255):
        '''
        추론을 위한 method

        is_greedy : softmax에서 가장 높은 확률값을 갖는 친구를 return
            - false일 경우 distribution sampling
        '''
        if isinstance(src, tuple):
            # zero pad부분 masking
            x, x_length = src
            mask = self.generate_mask(x, x_length)
        else:
            x, x_length = src, None
            mask = None
        batch_size = x.size(0)

        # Same procedure as teacher forcing.
        emb_src = self.emb_src(x) # [b, n, emb]
        h_src, h_0_tgt = self.encoder((emb_src, x_length)) # (b,n,h), ([l*2, b, h/2], [l*2, b, h/2])
        decoder_hidden = self.fast_merge_encoder_hiddens(h_0_tgt) # [l, b, h], [l, b, h]


        # --------------- 여기서부터 달라져----------------------------
        # decoding첫 파트 BOS넣어주기 : [b, 1]
        y = x.new(batch_size, 1).zero_() + data_loader.BOS 
            # data_loader의 상단에 보면 BOS오브젝트 있음.
            # x와 같은 타입, 디바이스를 [B, 1]을 0으로 채워서 만들고 거기다가 BOS인 2를 넣는다.
            # 즉 [B,1] 2가 들어간 텐서가 만들어짐.
        '''
               [[2]
                [2]
                 .
                 .
                [2]]
        '''

        is_decoding = x.new_ones(batch_size, 1).bool() # bunch of 1
            # 배치마다 디코딩이 끝나는 부분이 다를것임.(?)
            # 아직 디코딩 중이면, 1, 디코딩 끝낫으면 0
            # EOS가 나오면 끝나는것,,
        h_t_tilde, y_hats, indice = None, [], []
        
        # Repeat a loop while sum of 'is_decoding' flag is bigger than 0,
        # or current time-step is smaller than maximum length.
        while is_decoding.sum() > 0 and len(indice) < max_length:
            # Unlike training procedure,
            # take the last time-step's output during the inference.
            emb_t = self.emb_dec(y) # 맨처음 y는 BOS(2)임.
            # |emb_t| = (batch_size, 1, word_vec_size)

            decoder_output, decoder_hidden = self.decoder(emb_t, # [B, 1, W]
                                                          h_t_tilde, # None
                                                          decoder_hidden) # [l,b,h],[l,b,h]
                # decoder_output : [b, 1, h] 
                # decoder_hidden : [n,b,h], [n,b,h]
            '''
            decoder_output
                 |
                ____
               |    | -> decoder_hidden
                ----
            
            '''
            context_vector = self.attn(h_src, decoder_output, mask)
                # (b, 1, h)  # softmax(Q*W*K)
            h_t_tilde = self.tanh(self.concat(torch.cat([decoder_output,
                                                         context_vector
                                                         ], dim=-1)))
            y_hat = self.generator(h_t_tilde)
                # |y_hat| = (b, 1, output_size) 단어 분포가 나와.
                # 각 샘플별, 현제 스탭에 관한, 단어 로그 확률 분포가 나옴.
            y_hats += [y_hat]

            if is_greedy:
                y = y_hat.argmax(dim=-1)
                # |y| = (batch_size, 1)
                # 만약 EOS면 3이 될거야
            else:
                # Take a random sampling based on the multinoulli distribution.
                y = torch.multinomial(y_hat.exp().view(batch_size, -1), 1) # exponential이 왜필요할까?
                # |y| = (batch_size, 1)

            # Put PAD if the sample is done.
            y = y.masked_fill_(~is_decoding, data_loader.PAD)
                # ~is_decoding 에서 ~은 -1임.
                # 1. 맨처음 step을 기준으로 말하자면 : is_decoding은 True로 채워진 [b, 1] matric이다.
                # 2. -1을 전부 취해주면 False로 채워진 [b,1]임.
                # 3. ~is_decoding이 True인 부분에 PAD(1)을 y자리에 채운다.
                # 4. 즉 한번 EOS(밑밑줄), PAD라고 불리우면 계속 PAD라고 예측하게 선언하는것.
                
            # Update is_decoding if there is EOS token.

            is_decoding = is_decoding * torch.ne(y, data_loader.EOS)
                # |is_decoding| = (batch_size, 1)
                # EOS가 y에서 나왔으면 is_decoding쪽을 0으로 만들어 버림.
                # torch.ne(x, y) : x != y일 경우 True인 Tensor를 반환. 즉 EOS가 아닌 부분은 True로 놓고 is_decoding이랑 곱하기를 해서 살려두는 것임.
            '''
                               [1]    *    [T]
                               [1]    *    [T]
                               [1]    *    [F]

            '''
            indice += [y]

        y_hats = torch.cat(y_hats, dim=1)
        indice = torch.cat(indice, dim=1)
        # |y_hats| = (batch_size, length, output_size)
        # |indice| = (batch_size, length) # sample별 문장별 원핫인코딩.

        return y_hats, indice



    #@profile
    def batch_beam_search(
        self,
        src,
        beam_size=5,
        max_length=255,
        n_best=1,
        length_penalty=.2
    ):
        mask, x_length = None, None

        if isinstance(src, tuple):
            x, x_length = src
            mask = self.generate_mask(x, x_length)
            # |mask| = (batch_size, length)
        else:
            x = src
        batch_size = x.size(0)

        emb_src = self.emb_src(x)
        h_src, h_0_tgt = self.encoder((emb_src, x_length))
        # |h_src| = (batch_size, length, hidden_size)
        h_0_tgt = self.fast_merge_encoder_hiddens(h_0_tgt)

        # initialize 'SingleBeamSearchBoard' as many as batch_size
        boards = [SingleBeamSearchBoard(
            h_src.device,
            {
                'hidden_state': {
                    'init_status': h_0_tgt[0][:, i, :].unsqueeze(1),
                    'batch_dim_index': 1,
                }, # |hidden_state| = (n_layers, batch_size, hidden_size)
                'cell_state': {
                    'init_status': h_0_tgt[1][:, i, :].unsqueeze(1),
                    'batch_dim_index': 1,
                }, # |cell_state| = (n_layers, batch_size, hidden_size)
                'h_t_1_tilde': {
                    'init_status': None,
                    'batch_dim_index': 0,
                }, # |h_t_1_tilde| = (batch_size, 1, hidden_size)
            },
            beam_size=beam_size,
            max_length=max_length,
        ) for i in range(batch_size)]
        is_done = [board.is_done() for board in boards]

        length = 0
        # Run loop while sum of 'is_done' is smaller than batch_size, 
        # or length is still smaller than max_length.
        while sum(is_done) < batch_size and length <= max_length:
            # current_batch_size = sum(is_done) * beam_size

            # Initialize fabricated variables.
            # As far as batch-beam-search is running, 
            # temporary batch-size for fabricated mini-batch is 
            # 'beam_size'-times bigger than original batch_size.
            fab_input, fab_hidden, fab_cell, fab_h_t_tilde = [], [], [], []
            fab_h_src, fab_mask = [], []
            
            # Build fabricated mini-batch in non-parallel way.
            # This may cause a bottle-neck.
            for i, board in enumerate(boards):
                # Batchify if the inference for the sample is still not finished.
                if board.is_done() == 0:
                    y_hat_i, prev_status = board.get_batch()
                    hidden_i    = prev_status['hidden_state']
                    cell_i      = prev_status['cell_state']
                    h_t_tilde_i = prev_status['h_t_1_tilde']

                    fab_input  += [y_hat_i]
                    fab_hidden += [hidden_i]
                    fab_cell   += [cell_i]
                    fab_h_src  += [h_src[i, :, :]] * beam_size
                    fab_mask   += [mask[i, :]] * beam_size
                    if h_t_tilde_i is not None:
                        fab_h_t_tilde += [h_t_tilde_i]
                    else:
                        fab_h_t_tilde = None

            # Now, concatenate list of tensors.
            fab_input  = torch.cat(fab_input,  dim=0)
            fab_hidden = torch.cat(fab_hidden, dim=1)
            fab_cell   = torch.cat(fab_cell,   dim=1)
            fab_h_src  = torch.stack(fab_h_src)
            fab_mask   = torch.stack(fab_mask)
            if fab_h_t_tilde is not None:
                fab_h_t_tilde = torch.cat(fab_h_t_tilde, dim=0)
            # |fab_input|     = (current_batch_size, 1)
            # |fab_hidden|    = (n_layers, current_batch_size, hidden_size)
            # |fab_cell|      = (n_layers, current_batch_size, hidden_size)
            # |fab_h_src|     = (current_batch_size, length, hidden_size)
            # |fab_mask|      = (current_batch_size, length)
            # |fab_h_t_tilde| = (current_batch_size, 1, hidden_size)

            emb_t = self.emb_dec(fab_input)
            # |emb_t| = (current_batch_size, 1, word_vec_size)

            fab_decoder_output, (fab_hidden, fab_cell) = self.decoder(emb_t,
                                                                      fab_h_t_tilde,
                                                                      (fab_hidden, fab_cell))
            # |fab_decoder_output| = (current_batch_size, 1, hidden_size)
            context_vector = self.attn(fab_h_src, fab_decoder_output, fab_mask)
            # |context_vector| = (current_batch_size, 1, hidden_size)
            fab_h_t_tilde = self.tanh(self.concat(torch.cat([fab_decoder_output,
                                                             context_vector
                                                             ], dim=-1)))
            # |fab_h_t_tilde| = (current_batch_size, 1, hidden_size)
            y_hat = self.generator(fab_h_t_tilde)
            # |y_hat| = (current_batch_size, 1, output_size)

            # separate the result for each sample.
            # fab_hidden[:, begin:end, :] = (n_layers, beam_size, hidden_size)
            # fab_cell[:, begin:end, :]   = (n_layers, beam_size, hidden_size)
            # fab_h_t_tilde[begin:end]    = (beam_size, 1, hidden_size)
            cnt = 0
            for board in boards:
                if board.is_done() == 0:
                    # Decide a range of each sample.
                    begin = cnt * beam_size
                    end = begin + beam_size

                    # pick k-best results for each sample.
                    board.collect_result(
                        y_hat[begin:end],
                        {
                            'hidden_state': fab_hidden[:, begin:end, :],
                            'cell_state'  : fab_cell[:, begin:end, :],
                            'h_t_1_tilde' : fab_h_t_tilde[begin:end],
                        },
                    )
                    cnt += 1

            is_done = [board.is_done() for board in boards]
            length += 1

        # pick n-best hypothesis.
        batch_sentences, batch_probs = [], []

        # Collect the results.
        for i, board in enumerate(boards):
            sentences, probs = board.get_n_best(n_best, length_penalty=length_penalty)

            batch_sentences += [sentences]
            batch_probs     += [probs]

        return batch_sentences, batch_probs
