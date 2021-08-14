import sys
import os.path

import torch

from train import define_argparser
from train import main


def overwrite_config(config, prev_config):
    # This method provides a compatibility for new or missing arguments.
    '''
    예를들어서 예전에 SGD만 있을때 모델이야. 근데 Adam이 나왓어, 그래서 그걸 적용해 보고 싶어.
    train.py에 Adam을 만든거지,, 
    '''
    for prev_key in vars(prev_config).keys():
        if not prev_key in vars(config).keys(): # vars : class의 dict속성 반환
            # 예전 config에서 이번 config에 없다면 워닝.
            # No such argument in current config. Ignore that value.
            print('WARNING!!! Argument "--%s" is not found in current argument parser.\tIgnore saved value:' % prev_key,
                  vars(prev_config)[prev_key])

    for key in vars(config).keys(): # 새로운 config
        if not key in vars(prev_config).keys(): # 새로운 컨피그가 생겼을때
            # No such argument in saved file. Use current value.
            print('WARNING!!! Argument "--%s" is not found in saved model.\tUse current value:' % key,
                  vars(config)[key])
        elif vars(config)[key] != vars(prev_config)[key]: # 과거와 현재의 컨피그가 다른 config사용했을때.
            if '--%s' % key in sys.argv:
                # User changed argument value at this execution.
                print('WARNING!!! You changed value for argument "--%s".\tUse current value:' % key,
                      vars(config)[key])
            else: # 여기는 잘 모르겟다.
                # User didn't changed at this execution, but current config and saved config is different.
                # This may caused by user's intension at last execution.
                # Load old value, and replace current value.
                vars(config)[key] = vars(prev_config)[key]

    return config


def continue_main(config, main):
    # If the model exists, load model and configuration to continue the training.
    if os.path.isfile(config.load_fn):
        saved_data = torch.load(config.load_fn, map_location='cpu')
        # saved_data = torch.load(config.load_fn, map_location='cpu' if config.gpu_id < 0 else f'cuda:{config.gpu_id}') # torch가 웃긴게, 예전에 학습햇던 모델이 1번 gpu에 했다면, 1번으로 자동으로 올릴려함. 그것을 방지하기 위함.
        # saved_data는 trainer.py에 있는 torch.save부분의 데이터를 불러옴.

        prev_config = saved_data['config']
        config = overwrite_config(config, prev_config)

        model_weight = saved_data['model']
        opt_weight = saved_data['opt']
        
        # from train.py
        main(config, model_weight=model_weight, opt_weight=opt_weight)
    else:
        print('Cannot find file %s' % config.load_fn)


if __name__ == '__main__':
    config = define_argparser(is_continue=True)
    continue_main(config, main)
