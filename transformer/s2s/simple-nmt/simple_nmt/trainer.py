import numpy as np

import torch
from torch import optim
import torch.nn.utils as torch_utils
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler


# Core of the library, contains an engine for training and evaluating, most of the classic machine learning metrics 
# and a variety of handlers to ease the pain of training and validation of neural networks.

from ignite.engine import Engine # Runs a given process_function over each batch of a dataset, emitting events as it goes.
from ignite.engine import Events # Events that are fired by the Engine during execution.
from ignite.metrics import RunningAverage # Compute running average of a metric or the output of process function.
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from simple_nmt.utils import get_grad_norm, get_parameter_norm


VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2

# Gradient init -> FeedForward -> Loss -> back prop -> Gradient descent -> 현제상태 출력 // 을 만들거야. 
# Train, Valid engine두개를 선언하고
# 그 두개를 물고 있는 엔진을 하나 또 설정해
class MaximumLikelihoodEstimationEngine(Engine):

    def __init__(self, func, model, crit, optimizer, lr_scheduler, config):
        self.model = model
        self.crit = crit # criterion (loss)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.config = config

        super().__init__(func)
        # func을 Engine class에 보내 실행하기.. 근데 그럼 어떤 효과가 잇는거지?
        '''https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module'''
        ''' 이해 안되면 이거 해보기.
        class adding():
            def __init__(self, a,b):
                print(a+b)
                return a+b
            
        class bdding(adding):
            def __init__(self):
                z = 1
                super(bdding, self).__init__(1,3)

        bdding()
        '''    
        

        self.best_loss = np.inf
        self.scaler = GradScaler() 


    # static : https://wikidocs.net/21054
    @staticmethod
    #@profile
    def train(engine, mini_batch):
        '''
            engine : 여기에 Engine을 받나본데? // train engine, valid engine
            mini_batch : train_loader가 return하는 객체
                pytorch.DataLoader? 이런 비슷한거 있음.
        '''
        # You have to reset the gradients of all model parameters
        # before to take another step in gradient descent.

        '''
            engine.state.iteration ; Number of iterations the engine has completed. Initialized as 0 and the first iteration is 1.
                https://pytorch.org/ignite/v0.4.6/concepts.html#state
            engine.iteration_per_update : update당 Iteration 숫자... 먼지 모르겟음.
        '''
        # accumulate gradient update
        engine.model.train() # 파라미터 학습 할 것임.
        if engine.state.iteration % engine.config.iteration_per_update == 1 \
            or engine.config.iteration_per_update == 1: # 만약 설정을 config.iteration_per_update 을 1로 했을 경우, 매 iter마다 업데이트, 만약 설정을 10으로 하면 11마다 zero grad함.
            if engine.state.iteration > 1:
                engine.optimizer.zero_grad()

        device = next(engine.model.parameters()).device # 모델이 어느 gpu에 있는지 따오는거
        # 현재 모델이 어느 gpu에 있는지 구해서, 미니배치안에 잇는 텐서들을 해당 디바이스에 보내준다.

        mini_batch.src = (mini_batch.src[0].to(device), mini_batch.src[1]) # encoder tuple, length // 근데 tuple gpu로 옮기는데 length는 안옮기네
        mini_batch.tgt = (mini_batch.tgt[0].to(device), mini_batch.tgt[1]) # decoder x, y
            # tuple은 tensor와 length가 있음.

            # Raw target variable has both BOS and EOS token. 
            # The output of sequence-to-sequence does not have BOS token. 
            # Thus, remove BOS token for reference.
        x, y = mini_batch.src, mini_batch.tgt[0][:, 1:]
            # |x| = (batch_size, length) : x에서는 tensor, length를 가져가고
            # |y| = (batch_size, length) : y(decoder)에서는 length가 빠지고, tensor들만 들어감. 
            # encoder에는 isinstance라는게 들어가서 tuple인지 확인했었어.
            # y의 원래 텐서 모습은 {BOS, y_1, y_2, EOS}이건데, Teacher forcing답안용이기때문에 {y_1, y_2, eos} // 모델에 학습 데이터 들어갈땐 {BOS, y_1, y_2} 들어갈거야.

        with autocast(not engine.config.off_autocast):
                # Take feed-forward
                # Similar as before, the input of decoder does not have EOS token.
                # Thus, remove EOS token for decoder input.
            y_hat = engine.model(x, mini_batch.tgt[0][:, :-1])  
                # 모델에 학습 데이터 들어갈땐 {BOS, y_1, y_2} 들어갈거야.
                # |y_hat| = (b, l, |V|) : 각 미니배치별, length별, 단어별 로그 확률 값이 들어가 있음.

            loss = engine.crit(
                y_hat.contiguous().view(-1, y_hat.size(-1)), # flatten
                y.contiguous().view(-1) # flatten
            )

            backward_target = loss.div(y.size(0)).div(engine.config.iteration_per_update) 
                # NLL을 보면 -1/n *sigma(sgima)인데, Loss정의시 1/n을 안했음. 그래서 직접 나눠줘야함.
                # 그래서 미니 배치 사이즈로 나눠주고, 이터레이션 퍼 업데이트로 나눠줌.

        # gpu가 있을때 autocast가 수행
        if engine.config.gpu_id >= 0 and not engine.config.off_autocast:
            engine.scaler.scale(backward_target).backward()
        else:
            backward_target.backward()

        word_count = int(mini_batch.tgt[1].sum())
            # 단어의 갯수를 알아야 로스를 정확하게 구할 수 있음.
            # 1 : 에는 각 batch별 length가 들어가 있을 것이기 때문에
        p_norm = float(get_parameter_norm(engine.model.parameters())) # parameter norm
        g_norm = float(get_grad_norm(engine.model.parameters())) # gradient norm : 안정적일수록 작아짐.
        '''
                @torch.no_grad()
                def get_grad_norm(parameters, norm_type=2):
                    parameters = list(filter(lambda p: p.grad is not None, parameters)) # 파라미터의수

                    total_norm = 0

                    try:
                        for p in parameters:
                            total_norm += (p.grad.data**norm_type).sum() # ||L2||_norm
                        total_norm = total_norm ** (1. / norm_type) # 
                    except Exception as e:
                        print(e)

                    return total_norm


                @torch.no_grad()
                def get_parameter_norm(parameters, norm_type=2):
                    total_norm = 0

                    try:
                        for p in parameters:
                            total_norm += (p.data**norm_type).sum()
                        total_norm = total_norm ** (1. / norm_type)
                    except Exception as e:
                        print(e)

                    return total_norm
        
        '''


        if engine.state.iteration % engine.config.iteration_per_update == 0 and \
            engine.state.iteration > 0:
                # In order to avoid gradient exploding, we apply gradient clipping.
            torch_utils.clip_grad_norm_(
                engine.model.parameters(),
                engine.config.max_grad_norm,
            )

            # Take a step of gradient descent.
            # GPU가 있을 경우에, scale을 함.
            if engine.config.gpu_id >= 0 and not engine.config.off_autocast:
                # Use scaler instead of engine.optimizer.step() if using GPU.
                engine.scaler.step(engine.optimizer)
                engine.scaler.update()
            else:
                engine.optimizer.step()

            # 만약 lr 스케쥴을 한다그러면..
            # if engine.config.use_noam_decay and engine.lr_scheduler is not None:
            #     engine.lr_scheduler.step()


        loss = float(loss / word_count)
            # 단어단 로스를 구할 수 있음.
        ppl = np.exp(loss)

        return {
            'loss': loss, # 나중에 화면에 출력할 것.
            'ppl': ppl,
            '|param|': p_norm if not np.isnan(p_norm) and not np.isinf(p_norm) else 0., # 학습 초기에 None, inf뜨는 경우가 있었다. 그러면 학습이 안되므로 0을 리턴하도록한다.
            '|g_param|': g_norm if not np.isnan(g_norm) and not np.isinf(g_norm) else 0.,
        }



    @staticmethod
    def validate(engine, mini_batch):
        engine.model.eval()

        with torch.no_grad():
            device = next(engine.model.parameters()).device
            mini_batch.src = (mini_batch.src[0].to(device), mini_batch.src[1])
            mini_batch.tgt = (mini_batch.tgt[0].to(device), mini_batch.tgt[1])

            x, y = mini_batch.src, mini_batch.tgt[0][:, 1:]
            # |x| = (batch_size, length)
            # |y| = (batch_size, length)

            with autocast(not engine.config.off_autocast):
                y_hat = engine.model(x, mini_batch.tgt[0][:, :-1])
                # |y_hat| = (batch_size, n_classes)
                loss = engine.crit(
                    y_hat.contiguous().view(-1, y_hat.size(-1)),
                    y.contiguous().view(-1),
                )
        
        word_count = int(mini_batch.tgt[1].sum())
        loss = float(loss / word_count)
        ppl = np.exp(loss)

        return {
            'loss': loss,
            'ppl': ppl,
        }




    @staticmethod
    def attach(
        train_engine, validation_engine,
        training_metric_names = ['loss', 'ppl', '|param|', '|g_param|'],
        validation_metric_names = ['loss', 'ppl'],
        verbose=VERBOSE_BATCH_WISE, # 0,1,2 중 하나.
    ):
        # Attaching would be repaeted for serveral metrics.
        # Thus, we can reduce the repeated codes by using this function.
        def attach_running_average(engine, metric_name):
            RunningAverage(output_transform=lambda x: x[metric_name]).attach(
                engine,
                metric_name,
            )

        for metric_name in training_metric_names:
            attach_running_average(train_engine, metric_name)

        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120) # progressbar만들어서 성과나오고.
            pbar.attach(train_engine, training_metric_names)

        if verbose >= VERBOSE_EPOCH_WISE:

            # 에폭이 끝날때마다 train_engine에서 print안에 있는 친구들을 출력하라고 함.
            @train_engine.on(Events.EPOCH_COMPLETED)
            def print_train_logs(engine):
                avg_p_norm = engine.state.metrics['|param|']
                avg_g_norm = engine.state.metrics['|g_param|']
                avg_loss = engine.state.metrics['loss']

                print('Epoch {} - |param|={:.2e} |g_param|={:.2e} loss={:.4e} ppl={:.2f}'.format(
                    engine.state.epoch,
                    avg_p_norm,
                    avg_g_norm,
                    avg_loss,
                    np.exp(avg_loss),
                ))

        # valid에 대해서도 동일하게 진행
        for metric_name in validation_metric_names:
            attach_running_average(validation_engine, metric_name)

        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(validation_engine, validation_metric_names)

        if verbose >= VERBOSE_EPOCH_WISE:
            @validation_engine.on(Events.EPOCH_COMPLETED)
            def print_valid_logs(engine):
                avg_loss = engine.state.metrics['loss']

                print('Validation - loss={:.4e} ppl={:.2f} best_loss={:.4e} best_ppl={:.2f}'.format(
                    avg_loss,
                    np.exp(avg_loss),
                    engine.best_loss,
                    np.exp(engine.best_loss),
                ))

    @staticmethod
    def resume_training(engine, resume_epoch):
        engine.state.iteration = (resume_epoch - 1) * len(engine.state.dataloader)
        engine.state.epoch = (resume_epoch - 1)

    @staticmethod
    def check_best(engine):
        loss = float(engine.state.metrics['loss'])
        if loss <= engine.best_loss:
            engine.best_loss = loss

    @staticmethod
    def save_model(engine, train_engine, config, src_vocab, tgt_vocab):
        avg_train_loss = train_engine.state.metrics['loss']
        avg_valid_loss = engine.state.metrics['loss']

        # Set a filename for model of last epoch.
        # We need to put every information to filename, as much as possible.
        model_fn = config.model_fn.split('.')
        
        model_fn = model_fn[:-1] + ['%02d' % train_engine.state.epoch,
                                    '%.2f-%.2f' % (avg_train_loss,
                                                   np.exp(avg_train_loss)
                                                   ),
                                    '%.2f-%.2f' % (avg_valid_loss,
                                                   np.exp(avg_valid_loss)
                                                   )
                                    ] + [model_fn[-1]]

        model_fn = '.'.join(model_fn)

        # Unlike other tasks, we need to save current model, not best model.
        torch.save(
            {
                'model': engine.model.state_dict(),
                'opt': train_engine.optimizer.state_dict(),
                'config': config,
                'src_vocab': src_vocab,
                'tgt_vocab': tgt_vocab,
            }, model_fn
        )
    

class SingleTrainer():

    def __init__(self, target_engine_class, config):
        self.target_engine_class = target_engine_class
        self.config = config

    def train(
        self,
        model, crit, optimizer,
        train_loader, valid_loader,
        src_vocab, tgt_vocab,
        n_epochs,
        lr_scheduler=None
    ):
        # Declare train and validation engine with necessary objects.
        train_engine = self.target_engine_class(
            self.target_engine_class.train,
            model,
            crit,
            optimizer,
            lr_scheduler,
            self.config
        )
        validation_engine = self.target_engine_class(
            self.target_engine_class.validate,
            model,
            crit,
            optimizer=None,
            lr_scheduler=None,
            config=self.config
        )

        # Do necessary attach procedure to train & validation engine.
        # Progress bar and metric would be attached.
        self.target_engine_class.attach(
            train_engine,
            validation_engine,
            verbose=self.config.verbose
        )

        # After every train epoch, run 1 validation epoch.
        # Also, apply LR scheduler if it is necessary.
        def run_validation(engine, validation_engine, valid_loader):
            validation_engine.run(valid_loader, max_epochs=1)

            if engine.lr_scheduler is not None:
                engine.lr_scheduler.step()

        # Attach above call-back function.
        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            run_validation,
            validation_engine,
            valid_loader
        )
        # Attach other call-back function for initiation of the training.
        train_engine.add_event_handler(
            Events.STARTED,
            self.target_engine_class.resume_training,
            self.config.init_epoch,
        )

        # Attach validation loss check procedure for every end of validation epoch.
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED, self.target_engine_class.check_best
        )
        # Attach model save procedure for every end of validation epoch.
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            self.target_engine_class.save_model,
            train_engine,
            self.config,
            src_vocab,
            tgt_vocab,
        )

        # Start training.
        train_engine.run(train_loader, max_epochs=n_epochs)

        return model
