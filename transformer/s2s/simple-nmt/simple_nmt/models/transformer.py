import torch
import torch.nn as nn

import simple_nmt.data_loader as data_loader
from simple_nmt.search import SingleBeamSearchBoard


class Attention(nn.Module):

    def __init__(self):
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)
            # K,V의 n에 대해서 softmax를 하는것임. 즉 참조할 부분(encoder)을 확률적으로 표현

    def forward(self, Q, K, V, mask=None, dk=64):

        # |Q| = (batch_size, m, hidden_size)
            # |Q| = (n_head * bs, m, hs/n_heads)
        # |K| = |V| = (batch_size, n, hidden_size)
        # |mask| = (batch_size, m, n)
            # |mask| = (n_head * bs, m, n)

        w = torch.bmm(Q, K.transpose(1, 2))
            # [bs, m, hs] * [bs, hs, n]
        # |w| = (batch_size, m, n)
            # s2s생각해보면, decoder에서 |w| = (bs, 1, hs)를 갖고 어텐션을 순차적으로 했었어,
            # 반면, attn에서는 한방에 한다. 모든 m에 대하여 한방에 하니까 속도는 빠르되, 메모리가 많이 필요함.
        if mask is not None:
            assert w.size() == mask.size()
            w.masked_fill_(mask, -float('inf'))

        w = self.softmax(w / (dk**.5)) # 스케일을 해주면 학습할때 좀더 안정적임.
            # 확률 분포 자체가 좀더 flat해짐.
        c = torch.bmm(w, V)
            # [bs, m, n] * [bs, n, hs]
        # |c| = (batch_size, m, hidden_size) -> 밖에서 기준으로는 |c| = (n_splits * bs, m, hs/splits)

        return c


class MultiHead(nn.Module):

    def __init__(self, hidden_size, n_splits):
        super().__init__()

        self.hidden_size = hidden_size
        self.n_splits = n_splits

        # Note that we don't have to declare each linear layer, separately.
        self.Q_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.K_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)

        self.attn = Attention()

    def forward(self, Q, K, V, mask=None):
        '''
        디코더의 self attn 같은 경우에는 미래를 보지 못하게 하는 mask가 필요함.
        인코더에는 <PAD>마스크를 함.
        '''
        # |Q|    = (batch_size, m, hidden_size)
        # |K|    = (batch_size, n, hidden_size)
        # |V|    = |K|
        # |mask| = (batch_size, m, n)

        QWs = self.Q_linear(Q).split(self.hidden_size // self.n_splits, dim=-1)
            # 결과물 : list - number of heads
            # QWs[0]이런식으로 꺼내 쓰면 된다.
        KWs = self.K_linear(K).split(self.hidden_size // self.n_splits, dim=-1)
        VWs = self.V_linear(V).split(self.hidden_size // self.n_splits, dim=-1)
        # |QW_i| = (batch_size, m, hidden_size / n_splits)
        # |KW_i| = |VW_i| = (batch_size, n, hidden_size / n_splits)

        # By concatenating splited linear transformed results,
        # we can remove sequential operations,
        # like mini-batch parallel operations.
            # 피피티에서 설명했듯, 헤드를 배치로 바꿈.
        QWs = torch.cat(QWs, dim=0)
        KWs = torch.cat(KWs, dim=0)
        VWs = torch.cat(VWs, dim=0)
        # |QWs| = (batch_size * n_splits, m, hidden_size / n_splits)
        # |KWs| = |VWs| = (batch_size * n_splits, n, hidden_size / n_splits)

        if mask is not None:
            mask = torch.cat([mask for _ in range(self.n_splits)], dim=0)
            # |mask| = (batch_size, m, n) -> (batch_size * n_splits, m, n)
            # mask는 반복되는거라서,, 그냥 연장하나보다.

        c = self.attn(
            QWs, KWs, VWs,
            mask=mask,
            dk=self.hidden_size // self.n_splits,
        )
        # |c| = (batch_size * n_splits, m, hidden_size / n_splits)

        # We need to restore temporal mini-batchfied multi-head attention results.
        c = c.split(Q.size(0), dim=0)
            # Q.size = batch_size
            # c.split하면  [bs * n_splits, m, hs/n_splits] -> [bs, m, hs/n_splits]
        # |c_i| = (batch_size, m, hidden_size / n_splits)
        c = self.linear(torch.cat(c, dim=-1))
        # |c| = (batch_size, m, hidden_size)

        return c


class EncoderBlock(nn.Module):
    '''
        Encoder block은 MySequential을 통해서 쌓을거야.
        원본 논문 : Residual하고, LayerNorm한다.
        하지만 우리는 : LayerNorm -> Multihead -> Residual 할것이다.

        순서 : x -> layerNorm -> Multihead -> dropout -> Residual -> layerNorm -> FFN -> Residual

    '''
    def __init__(
        self,
        hidden_size,
        n_splits,
        dropout_p=.1,
        use_leaky_relu=False,
    ):
        super().__init__()

        self.attn = MultiHead(hidden_size, n_splits)
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn_dropout = nn.Dropout(dropout_p)

        # FFN을 하는데 -> 늘렷다가 줄이는 FFN
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.LeakyReLU() if use_leaky_relu else nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.fc_norm = nn.LayerNorm(hidden_size)
        self.fc_dropout = nn.Dropout(dropout_p)

    def forward(self, x, mask):
        # |x|    = (batch_size, n, hidden_size)
        # |mask| = (batch_size, n, n)

        # Post-LN:
        # z = self.attn_norm(x + self.attn_dropout(self.attn(Q=x,
        #                                                    K=x,
        #                                                    V=x,
        #                                                    mask=mask)))
        # z = self.fc_norm(z + self.fc_dropout(self.fc(z)))

        # Pre-LN:
        z = self.attn_norm(x)
        z = x + self.attn_dropout(self.attn(Q=z,
                                            K=z,
                                            V=z,
                                            mask=mask))
        z = z + self.fc_dropout(self.fc(self.fc_norm(z)))
        # |z| = (batch_size, n, hidden_size)

        return z, mask


class DecoderBlock(nn.Module):
    '''
        decoderblock은 attn이 두개의 종류를 갖는다.

        1. self attn
        2. encoder와 하는 일반적인 attn

    
    '''
    def __init__(
        self,
        hidden_size,
        n_splits,
        dropout_p=.1,
        use_leaky_relu=False,
    ):
        super().__init__()

        self.masked_attn = MultiHead(hidden_size, n_splits) # self attn
        self.masked_attn_norm = nn.LayerNorm(hidden_size)
        self.masked_attn_dropout = nn.Dropout(dropout_p)

        self.attn = MultiHead(hidden_size, n_splits) # normal attn
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn_dropout = nn.Dropout(dropout_p)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.LeakyReLU() if use_leaky_relu else nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.fc_norm = nn.LayerNorm(hidden_size)
        self.fc_dropout = nn.Dropout(dropout_p)

    def forward(self, x, key_and_value, mask, prev, future_mask):
        # |key_and_value| = (batch_size, n, hidden_size) : 인코더의 아웃풋.
        # |mask|          = (batch_size, m, n) : sorce <PAD> masking

        # In case of inference, we don't have to repeat same feed-forward operations.
        # Thus, we save previous feed-forward results.
        if prev is None: # Training mode : Teacher Forcing
            # |x|           = (batch_size, m, hidden_size)
            # |prev|        = None
            # |future_mask| = (batch_size, m, m)
            '''
                     --- --- --- ---
                    |   |||||||||||||
                     --- --- --- ---
                    |   |   |||||||||
                     --- --- --- ---
                    |   |   |   |||||
                     --- --- --- ---
                    |   |   |   |   |
                     --- --- --- ---

            c = torch.bmm(w, V)
                # [bs, m, n] * [bs, n, hs]

            마스킹 된 부분은 softmax후 0이 되니까 mask안된 부분만 반영이 됨.
            
            '''
            # |z|           = (batch_size, m, hidden_size)

            # Post-LN:
            # z = self.masked_attn_norm(x + self.masked_attn_dropout(
            #     self.masked_attn(x, x, x, mask=future_mask)
            # ))

            # Pre-LN:
            z = self.masked_attn_norm(x)
            z = x + self.masked_attn_dropout(
                self.masked_attn(z, z, z, mask=future_mask)
            )
        else: # Inference mode : AutoRegressive한 작업임.
                # 추론문일 때는 한 타임 스탭씩 진행해야 한다.
                # prev는 beam_search할때 언급했는데,, 모든 타임 스텝의 hidden을 갖고 있어야한다고 했음. 이때 모든 타임 스탭의 히든이 prev임.
                # 즉 prev가 있으면 추론모드임.
            # |x|           = (batch_size, 1, hidden_size)
            # |prev|        = (batch_size, t - 1, hidden_size) : t 이전 input의 모든 attn 결과값을 갖고 있음.
            # |future_mask| = None
            # |z|           = (batch_size, 1, hidden_size)

            # Post-LN:
            # z = self.masked_attn_norm(x + self.masked_attn_dropout(
            #     self.masked_attn(x, prev, prev, mask=None)
            # ))

            # Pre-LN:
            normed_prev = self.masked_attn_norm(prev)
            z = self.masked_attn_norm(x)
            z = x + self.masked_attn_dropout(
                self.masked_attn(z, normed_prev, normed_prev, mask=None) # mask를 None으로 한 이유는, 어차피 input에 미래가 없기 때문이다.
            )

        # Post-LN:
        # z = self.attn_norm(z + self.attn_dropout(self.attn(Q=z,
        #                                                    K=key_and_value,
        #                                                    V=key_and_value,
        #                                                    mask=mask)))

        # Pre-LN:
        normed_key_and_value = self.attn_norm(key_and_value)
        z = z + self.attn_dropout(self.attn(Q=self.attn_norm(z),
                                            K=normed_key_and_value,
                                            V=normed_key_and_value,
                                            mask=mask))
        # |z| = (batch_size, m, hidden_size) 왜 z에서 m이지?


        # Post-LN:
        # z = self.fc_norm(z + self.fc_dropout(self.fc(z)))

        # Pre-LN:
        z = z + self.fc_dropout(self.fc(self.fc_norm(z)))
        # |z| = (batch_size, m, hidden_size)

        return z, key_and_value, mask, prev, future_mask
            # 이 입력을 위의 블록에서 고대로 받아서 똑같은 행위를 함. : 출력이랑 입력이랑 같음.


class MySequential(nn.Sequential):
    '''
    nn.Sequential만 상속 받아서 forward만 갈아 엎음.

    *x같은 경우 x,key_and_value,mask,prev,future_mask같은 tuple이 들어갈거야.
    '''

        # *args : 파라미터를 몇개를 받을지 모르는 경우 사용, args는 튜플 형태로 전달.
    def forward(self, *x):
        # nn.Sequential class does not provide multiple input arguments and returns.
        # Thus, we need to define new class to solve this issue.
        # Note that each block has same function interface.

        for module in self._modules.values():
            x = module(*x)

        return x


class Transformer(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        n_splits,
        n_enc_blocks=6,
        n_dec_blocks=6,
        dropout_p=.1,
        use_leaky_relu=False,
        max_length=512,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_splits = n_splits
        self.n_enc_blocks = n_enc_blocks
        self.n_dec_blocks = n_dec_blocks
        self.dropout_p = dropout_p
        self.max_length = max_length

        super().__init__()

        self.emb_enc = nn.Embedding(input_size, hidden_size)
        self.emb_dec = nn.Embedding(output_size, hidden_size)
        self.emb_dropout = nn.Dropout(dropout_p)

        # positioning Encoding : 미리 계산해놓음.. 엄청 큰 메트릭을 만들고, 필요한 만큼 꺼내 쓰면됨.
        self.pos_enc = self._generate_pos_enc(hidden_size, max_length)


        # nn.Sequential을 안쓰는 이유는,, source를 보면, forward(self, input)임. 이때 input으로 단일만 받고 tuple을 받지 못한다.
        self.encoder = MySequential(
            *[EncoderBlock(
                hidden_size,
                n_splits,
                dropout_p,
                use_leaky_relu,
              ) for _ in range(n_enc_blocks)],
        )
        self.decoder = MySequential(
            *[DecoderBlock(
                hidden_size,
                n_splits,
                dropout_p,
                use_leaky_relu,
              ) for _ in range(n_dec_blocks)],
        )
        self.generator = nn.Sequential(
            nn.LayerNorm(hidden_size), # Only for Pre-LN Transformer. -> decoder에서 나오자마자 LN이 필요함.
            nn.Linear(hidden_size, output_size),
            nn.LogSoftmax(dim=-1),
                # https://junstar92.tistory.com/118 : log_softmax
        )

    @torch.no_grad()
    def _generate_pos_enc(self, hidden_size, max_length):
        '''
        positioning encoding을 만들어 주는 함수.

        예) pos = 3(word) // dim_idx = 2 = 2*i
            
                        pos
            sin(  --------------- )
                    10^4(2*i/d)

        '''
        enc = torch.FloatTensor(max_length, hidden_size).zero_()
        # |enc| = (max_length, hidden_size)

        pos = torch.arange(0, max_length).unsqueeze(-1).float()
        dim = torch.arange(0, hidden_size // 2).unsqueeze(0).float()
        # |pos| = (max_length, 1)
        # |dim| = (1, hidden_size // 2)

        enc[:, 0::2] = torch.sin(pos / 1e+4**dim.div(float(hidden_size)))
        enc[:, 1::2] = torch.cos(pos / 1e+4**dim.div(float(hidden_size)))

        return enc




    def _position_encoding(self, x, init_pos=0):
        # |x| = (batch_size, n, hidden_size)
        # |self.pos_enc| = (max_length, hidden_size)
        assert x.size(-1) == self.pos_enc.size(-1)
        assert x.size(1) + init_pos <= self.max_length

        pos_enc = self.pos_enc[init_pos:init_pos + x.size(1)].unsqueeze(0)
        # |pos_enc| = (1, n, hidden_size)
        x = x + pos_enc.to(x.device)

        return x



    @torch.no_grad()
    def _generate_mask(self, x, length):
        '''
        
        에) length : [4,3,2]
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
                # If length of sample equals to maximum length among samples,
                # set every value in mask to be 0.
                mask += [x.new_ones(1, l).zero_()]

        mask = torch.cat(mask, dim=0).bool()
        # |mask| = (batch_size, max_length)

        return mask



    def forward(self, x, y):
        '''
        우리는 teacher forcing이니까,, x,y둘다 들어와야함.
            x는 packed_sequence가 들어옴.
        출력은 logSoftmax.
        '''
        # |x[0]| = (batch_size, n)
        # |y|    = (batch_size, m)

        # Mask to prevent having attention weight on padding position.
        with torch.no_grad():
            mask = self._generate_mask(x[0], x[1]) # x[0] : one hot encoding, x[1] : sequence_length
            # |mask| = (batch_size, n)
            x = x[0]

            mask_enc = mask.unsqueeze(1).expand(*x.size(), mask.size(-1))
            mask_dec = mask.unsqueeze(1).expand(*y.size(), mask.size(-1))
            # |mask_enc| = (batch_size, n, n)
            # |mask_dec| = (batch_size, m, n)

        z = self.emb_dropout(self._position_encoding(self.emb_enc(x)))
        z, _ = self.encoder(z, mask_enc)
        # |z| = (batch_size, n, hidden_size)

        # Generate future mask
        with torch.no_grad():
            future_mask = torch.triu(x.new_ones((y.size(1), y.size(1))), diagonal=1).bool()
            # |future_mask| = (m, m)
            future_mask = future_mask.unsqueeze(0).expand(y.size(0), *future_mask.size())
            # |fwd_mask| = (batch_size, m, m)

        h = self.emb_dropout(self._position_encoding(self.emb_dec(y)))
        h, _, _, _, _ = self.decoder(h, z, mask_dec, None, future_mask)
        # |h| = (batch_size, m, hidden_size)

        y_hat = self.generator(h)
        # |y_hat| = (batch_size, m, output_size)

        return y_hat

    def search(self, x, is_greedy=True, max_length=255):
        # |x[0]| = (batch_size, n)
        batch_size = x[0].size(0)

        mask = self._generate_mask(x[0], x[1])
        # |mask| = (batch_size, n)
        x = x[0]

        mask_enc = mask.unsqueeze(1).expand(mask.size(0), x.size(1), mask.size(-1))
        mask_dec = mask.unsqueeze(1)
        # |mask_enc| = (batch_size, n, n)
        # |mask_dec| = (batch_size, 1, n)

        z = self.emb_dropout(self._position_encoding(self.emb_enc(x)))
        z, _ = self.encoder(z, mask_enc)
        # |z| = (batch_size, n, hidden_size)

        # Fill a vector, which has 'batch_size' dimension, with BOS value.
        y_t_1 = x.new(batch_size, 1).zero_() + data_loader.BOS
        # |y_t_1| = (batch_size, 1)
        is_decoding = x.new_ones(batch_size, 1).bool()

        prevs = [None for _ in range(len(self.decoder._modules) + 1)]
        y_hats, indice = [], []
        # Repeat a loop while sum of 'is_decoding' flag is bigger than 0,
        # or current time-step is smaller than maximum length.
        while is_decoding.sum() > 0 and len(indice) < max_length:
            # Unlike training procedure,
            # take the last time-step's output during the inference.
            h_t = self.emb_dropout(
                self._position_encoding(self.emb_dec(y_t_1), init_pos=len(indice))
            )
            # |h_t| = (batch_size, 1, hidden_size))
            if prevs[0] is None:
                prevs[0] = h_t
            else:
                prevs[0] = torch.cat([prevs[0], h_t], dim=1)

            for layer_index, block in enumerate(self.decoder._modules.values()):
                prev = prevs[layer_index]
                # |prev| = (batch_size, len(y_hats), hidden_size)

                h_t, _, _, _, _ = block(h_t, z, mask_dec, prev, None)
                # |h_t| = (batch_size, 1, hidden_size)

                if prevs[layer_index + 1] is None:
                    prevs[layer_index + 1] = h_t
                else:
                    prevs[layer_index + 1] = torch.cat([prevs[layer_index + 1], h_t], dim=1)
                # |prev| = (batch_size, len(y_hats) + 1, hidden_size)

            y_hat_t = self.generator(h_t)
            # |y_hat_t| = (batch_size, 1, output_size)

            y_hats += [y_hat_t]
            if is_greedy: # Argmax
                y_t_1 = torch.topk(y_hat_t, 1, dim=-1)[1].squeeze(-1)
            else: # Random sampling                
                y_t_1 = torch.multinomial(y_hat_t.exp().view(x.size(0), -1), 1)
            # Put PAD if the sample is done.
            y_t_1 = y_t_1.masked_fill_(
                ~is_decoding,
                data_loader.PAD,
            )

            # Update is_decoding flag.
            is_decoding = is_decoding * torch.ne(y_t_1, data_loader.EOS)
            # |y_t_1| = (batch_size, 1)
            # |is_decoding| = (batch_size, 1)
            indice += [y_t_1]

        y_hats = torch.cat(y_hats, dim=1)
        indice = torch.cat(indice, dim=-1)
        # |y_hats| = (batch_size, m, output_size)
        # |indice| = (batch_size, m)

        return y_hats, indice

    #@profile
    def batch_beam_search(
        self,
        x,
        beam_size=5,
        max_length=255,
        n_best=1,
        length_penalty=.2,
    ):
        # |x[0]| = (batch_size, n)
        batch_size = x[0].size(0)
        n_dec_layers = len(self.decoder._modules)

        mask = self._generate_mask(x[0], x[1])
        # |mask| = (batch_size, n)
        x = x[0]

        mask_enc = mask.unsqueeze(1).expand(mask.size(0), x.size(1), mask.size(-1))
        mask_dec = mask.unsqueeze(1)
        # |mask_enc| = (batch_size, n, n)
        # |mask_dec| = (batch_size, 1, n)

        z = self.emb_dropout(self._position_encoding(self.emb_enc(x)))
        z, _ = self.encoder(z, mask_enc)
        # |z| = (batch_size, n, hidden_size)

        prev_status_config = {}
        for layer_index in range(n_dec_layers + 1):
            prev_status_config['prev_state_%d' % layer_index] = {
                'init_status': None,
                'batch_dim_index': 0,
            }
        # Example of prev_status_config:
        # prev_status_config = {
        #     'prev_state_0': {
        #         'init_status': None,
        #         'batch_dim_index': 0,
        #     },
        #     'prev_state_1': {
        #         'init_status': None,
        #         'batch_dim_index': 0,
        #     },
        #
        #     ...
        #
        #     'prev_state_${n_layers}': {
        #         'init_status': None,
        #         'batch_dim_index': 0,
        #     }
        # }

        boards = [
            SingleBeamSearchBoard(
                z.device,
                prev_status_config,
                beam_size=beam_size,
                max_length=max_length,
            ) for _ in range(batch_size)
        ]
        done_cnt = [board.is_done() for board in boards]

        length = 0
        while sum(done_cnt) < batch_size and length <= max_length:
            fab_input, fab_z, fab_mask = [], [], []
            fab_prevs = [[] for _ in range(n_dec_layers + 1)]

            for i, board in enumerate(boards): # i == sample_index in minibatch
                if board.is_done() == 0:
                    y_hat_i, prev_status = board.get_batch()

                    fab_input += [y_hat_i                 ]
                    fab_z     += [z[i].unsqueeze(0)       ] * beam_size
                    fab_mask  += [mask_dec[i].unsqueeze(0)] * beam_size

                    for layer_index in range(n_dec_layers + 1):
                        prev_i = prev_status['prev_state_%d' % layer_index]
                        if prev_i is not None:
                            fab_prevs[layer_index] += [prev_i]
                        else:
                            fab_prevs[layer_index] = None

            fab_input = torch.cat(fab_input, dim=0)
            fab_z     = torch.cat(fab_z,     dim=0)
            fab_mask  = torch.cat(fab_mask,  dim=0)
            for i, fab_prev in enumerate(fab_prevs): # i == layer_index
                if fab_prev is not None:
                    fab_prevs[i] = torch.cat(fab_prev, dim=0)
            # |fab_input|    = (current_batch_size, 1,)
            # |fab_z|        = (current_batch_size, n, hidden_size)
            # |fab_mask|     = (current_batch_size, 1, n)
            # |fab_prevs[i]| = (current_batch_size, length, hidden_size)
            # len(fab_prevs) = n_dec_layers + 1

            # Unlike training procedure,
            # take the last time-step's output during the inference.
            h_t = self.emb_dropout(
                self._position_encoding(self.emb_dec(fab_input), init_pos=length)
            )
            # |h_t| = (current_batch_size, 1, hidden_size)
            if fab_prevs[0] is None:
                fab_prevs[0] = h_t
            else:
                fab_prevs[0] = torch.cat([fab_prevs[0], h_t], dim=1)

            for layer_index, block in enumerate(self.decoder._modules.values()):
                prev = fab_prevs[layer_index]
                # |prev| = (current_batch_size, m, hidden_size)

                h_t, _, _, _, _ = block(h_t, fab_z, fab_mask, prev, None)
                # |h_t| = (current_batch_size, 1, hidden_size)

                if fab_prevs[layer_index + 1] is None:
                    fab_prevs[layer_index + 1] = h_t
                else:
                    fab_prevs[layer_index + 1] = torch.cat(
                        [fab_prevs[layer_index + 1], h_t],
                        dim=1,
                    ) # Append new hidden state for each layer.

            y_hat_t = self.generator(h_t)
            # |y_hat_t| = (batch_size, 1, output_size)

            # |fab_prevs[i][begin:end]| = (beam_size, length, hidden_size)
            cnt = 0
            for board in boards:
                if board.is_done() == 0:
                    begin = cnt * beam_size
                    end = begin + beam_size

                    prev_status = {}
                    for layer_index in range(n_dec_layers + 1):
                        prev_status['prev_state_%d' % layer_index] = fab_prevs[layer_index][begin:end]

                    board.collect_result(y_hat_t[begin:end], prev_status)

                    cnt += 1

            done_cnt = [board.is_done() for board in boards]
            length += 1

        batch_sentences, batch_probs = [], []

        for i, board in enumerate(boards):
            sentences, probs = board.get_n_best(n_best, length_penalty=length_penalty)

            batch_sentences += [sentences]
            batch_probs     += [probs]

        return batch_sentences, batch_probs
