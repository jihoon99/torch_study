from operator import itemgetter

import torch
import torch.nn as nn

import simple_nmt.data_loader as data_loader

LENGTH_PENALTY = .2
MIN_LENGTH = 5


class SingleBeamSearchBoard():
    '''
    From Board
        - input : x_t
        - last hidden : h_t_1
        - last cell : c_t_1
        - last hidden_tilde : h_tilde_t_1
    
    '''
    def __init__(
        self,
        device,
        prev_status_config, # Fake minibatch를 만들기 위해선, input, hidden, cell, hidden_tilde가 필요함. 이게 prev_status_config에 들어감.// type은 dict의 dict
        beam_size=5,
        max_length=255,
    ):
        self.beam_size = beam_size
        self.max_length = max_length

        # To put data to same device.
        self.device = device
        # Inferred word index for each time-step. For now, initialized with initial time-step. 첫번째니까 빔 사이즈 만큼 모두 BOS가 들어가야함.
        self.word_indice = [torch.LongTensor(beam_size).zero_().to(self.device) + data_loader.BOS]
            # [[0,0,0,0,0],
            #  [1,0,0,0,0],
            #  ...]
        # Beam index for selected word index, at each time-step. 빔 사이즈 만큼 -1을 채워넣은 텐서
        self.beam_indice = [torch.LongTensor(beam_size).zero_().to(self.device) - 1]
        # Cumulative log-probability for each beam. 
        self.cumulative_probs = [torch.FloatTensor([.0] + [-float('inf')] * (beam_size - 1)).to(self.device)]
            # 처음 cumulative_probs에서 [0, -inf, -inf, -inf, -inf]로 하고싶음. 왜냐면 BOS는 확정적인거라 1의 확률을 갖는데 log(BOS) = 0임.
            # 가지가 분할하기 전에는 한개의 확률만 갖으므로 나머지는 -inf로 채움.
            # 첫번째 빔에서(0)만 5개의 후보가 뽑힐거야
        # 1 if it is done else 0
        self.masks = [torch.BoolTensor(beam_size).zero_().to(self.device)]
            # 0이면 진행, 1(EOS)면 끝.

        # We don't need to remember every time-step of hidden states:
        #       prev_hidden, prev_cell, prev_h_t_tilde
        # What we need is remember just last one.

        #-------------------- 맨처음 세팅해줘야 하는것 -----------------------
        self.prev_status = {}
        self.batch_dims = {}
        for prev_status_name, each_config in prev_status_config.items():
            init_status = each_config['init_status'] # 바로 전의 state을 가져오고
            batch_dim_index = each_config['batch_dim_index'] # 배치 인덱스를 가져와
            if init_status is not None:
                self.prev_status[prev_status_name] = torch.cat([init_status] * beam_size,
                                                               dim=batch_dim_index)
                    # hidden, cell :  [L, B*beam_size, H] 여기서 B는 1임.
            else:
                self.prev_status[prev_status_name] = None
                    # h_tilde : [B*beam, 1, hidden]
            self.batch_dims[prev_status_name] = batch_dim_index
                # {hidden_state : 1, ...}
        self.current_time_step = 0
        self.done_cnt = 0






    def get_length_penalty(
        self,
        length,
        alpha=LENGTH_PENALTY,
        min_length=MIN_LENGTH,
    ):
        # Calculate length-penalty,
        # because shorter sentence usually have bigger probability.
        # In fact, we represent this as log-probability, which is negative value.
        # Thus, we need to multiply bigger penalty for shorter one.
        p = ((min_length + 1) / (min_length + length))**alpha
        # 6/(5+1)
        return p



    def is_done(self):
        '''
        빔사이즈보다, done_cnt가 크거나 같으면 1을 리턴, 아니면 0을 리턴.   
        done_cnt는 collect_result에서 업데이트 될것.     
        '''

        # Return 1, if we had EOS more than 'beam_size'-times.
        if self.done_cnt >= self.beam_size:
            return 1
        return 0



    def get_batch(self):
        '''
        returning [beam_size,1] : last step output(V_t)
                [baem_size, L, H] : prev_state_i  = 이거 튜플임.
        '''
        y_hat = self.word_indice[-1].unsqueeze(-1)
            # word_indice : 5(beamSize) 이전 타임 스탭의 출력물을 가져옴. -> unsqueeze(-1)
        # |y_hat| = (beam_size, 1)
        # if model != transformer:
        #     |hidden| = |cell| = (n_layers, beam_size, hidden_size)
        #     |h_t_tilde| = (beam_size, 1, hidden_size) or None
        # else:
        #     |prev_state_i| = (beam_size, length, hidden_size),
        #     where i is an index of layer.
        return y_hat, self.prev_status




    #@profile 
    def collect_result(self, y_hat, prev_status):
        # |y_hat| = (beam_size, 1, output_size)
        # prev_status is a dict, which has following keys:
        # if model != transformer:
        #     |hidden| = |cell| = (n_layers, beam_size, hidden_size)
        #     |h_t_tilde| = (beam_size, 1, hidden_size)
        # else:
        #     |prev_state_i| = (beam_size, length, hidden_size),
        #     where i is an index of layer.
        output_size = y_hat.size(-1)

        self.current_time_step += 1

        #---------------- Calculate cumulative log-probability. ----------------------
        # First, fill -inf value to last cumulative probability, if the beam is already finished.
        # Second, expand -inf filled cumulative probability to fit to 'y_hat'.
        # (beam_size) --> (beam_size, 1, 1) --> (beam_size, 1, output_size)
        # Third, add expanded cumulative probability to 'y_hat'
        cumulative_prob = self.cumulative_probs[-1].masked_fill_(self.masks[-1], -float('inf'))
            # cumulative_probs들이 5개씩 탁탁탁 쌓일텐데 그중 마지막걸 가져와서
            # 마지막 마스킹 정보를 갖고와서, True이면 마스크를 하고 with -inf로, 아니면 마스킹을 하지 않는다.
        cumulative_prob = y_hat + cumulative_prob.view(-1, 1, 1).expand(self.beam_size, 1, output_size) # broadcasting되기 때문에 expand안해도됨.
        # |cumulative_prob| = (beam_size, 1, output_size)

        # Now, we have new top log-probability and its index.
        # We picked top index as many as 'beam_size'.
        # Be aware that we picked top-k from whole batch through 'view(-1)'.

        # Following lines are using torch.topk, which is slower than torch.sort.
        # top_log_prob, top_indice = torch.topk(
        #     cumulative_prob.view(-1), # (beam_size * output_size,)
        #     self.beam_size,
        #     dim=-1,
        # )

        # Following lines are using torch.sort, instead of using torch.topk.
        top_log_prob, top_indice = cumulative_prob.view(-1).sort(descending=True)
            # torch.sort를 사용하면 : values, indice두개를 내뱉는다.
        top_log_prob, top_indice = top_log_prob[:self.beam_size], top_indice[:self.beam_size]
            # 상위 5개만 갖고온다.

        # |top_log_prob| = (beam_size,)
        # |top_indice| = (beam_size,)

        # Because we picked from whole batch, original word index should be calculated again.
        self.word_indice += [top_indice.fmod(output_size)]
            # fmod : element-wise나머지 구하기. // devided by output_size
            # outputsize로 나누면 원래 vocab_index가 리턴이 되겠네
        # Also, we can get an index of beam, which has top-k log-probability search result.
        self.beam_indice += [top_indice.div(float(output_size)).long()]
            # 41030 -> 4번째 빔에서, 1030번째 단어. 여기서 구하고자 하는것은 몇번째 빔인지 구하고 싶음.

        # Add results to history boards.
        self.cumulative_probs += [top_log_prob]
        self.masks += [torch.eq(self.word_indice[-1], data_loader.EOS)] # Set finish mask if we got EOS.
            # torch equal (word_indice[-1], data_loader.EOS) -> 1 if it is, else 0
        # Calculate a number of finished beams.
        self.done_cnt += self.masks[-1].float().sum() # EOS가 몇개 있엇는지 확인

        # In beam search procedure, we only need to memorize latest status.
        # For seq2seq, it would be lastest hidden and cell state, and h_t_tilde.
        # The problem is hidden(or cell) state and h_t_tilde has different dimension order.
        # In other words, a dimension for batch index is different.
        # Therefore self.batch_dims stores the dimension index for batch index.
        # For transformer, lastest status is each layer's decoder output from the biginning.
        # Unlike seq2seq, transformer has to memorize every previous output for attention operation.
        for prev_status_name, prev_status in prev_status.items():
            self.prev_status[prev_status_name] = torch.index_select(
                prev_status,
                dim=self.batch_dims[prev_status_name], # 어떤 차원을 뽑아올지
                index=self.beam_indice[-1] # 정해진 dim에서 몇번째 데이터를 뽑아올지.
            ).contiguous()







    def get_n_best(self, n=1, length_penalty=.2):
        '''
        output : 최고의 확률을 갖는 5개를 선별하고, 
        sentences와 probs를 return한다.

        sentences : [[2021, 3, 1394, ...],
                    [3019, ,20, 391, ...],
                    ...
                    [1010, 50, 0203, ...]]

        probs : [0.3, 0.5, 0.1, 0.3, 0.4]

        '''

        sentences, probs, founds = [], [], []

        for t in range(len(self.word_indice)):  # for each time-step,,,
            # word_indice : [[0,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],...]
            for b in range(self.beam_size):  # for each beam, // 5번
                if self.masks[t][b] == 1:  # if we had EOS on this time-step and beam, EOS를 찾으면,
                    # Take a record of penaltified log-proability.
                    probs += [self.cumulative_probs[t][b] * self.get_length_penalty(t, alpha=length_penalty)]
                    founds += [(t, b)] # 어디서 EOS를 찾았는지 -> 그 확률이 어떻게 되나 나중에 역추적할라고
                    # 재수가 없으면 EOS없이 max len으로 끝날수 있음.

        # Also, collect log-probability from last time-step, for the case of EOS is not shown.
        for b in range(self.beam_size):
            if self.cumulative_probs[-1][b] != -float('inf'): # If this beam does not have EOS,
                if not (len(self.cumulative_probs) - 1, b) in founds:
                    probs += [self.cumulative_probs[-1][b] * self.get_length_penalty(len(self.cumulative_probs),
                                                                                     alpha=length_penalty)]
                    founds += [(t, b)]

        # Sort and take n-best.
        sorted_founds_with_probs = sorted(
            zip(founds, probs),
            key=itemgetter(1),
            reverse=True,
        )[:n]
        probs = []

        for (end_index, b), prob in sorted_founds_with_probs:
            sentence = []

            # Trace from the end.
            for t in range(end_index, 0, -1):
                sentence = [self.word_indice[t][b]] + sentence
                b = self.beam_indice[t][b] # 빔따라서 거꾸로 올라감.

            sentences += [sentence]
            probs += [prob]

        return sentences, probs
