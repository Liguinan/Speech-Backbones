import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import run_spiral

if __name__ == '__main__':
    pretrained_model_type = 'base'  # or large
    dataset = 'train-other-500'  # or train-clean-100 or train-clean-360 or train-other-500 or dev-clean or dev-other or test-clean or test-other

    if pretrained_model_type == 'base':
        print("feature extraction on base model ...")
        run_args = {
            'model_save_dir': 'output/base/{}'.format(dataset),
            'config_path': 'examples/asr/conf/spiral',
            'config_name': 'spiral_base_pretrain_ls960',
            'data_dir': '/data/data/asr/librispeech',
            'test_manifest': 'manifest_json/librivox-{}.json'.format(dataset),
            'num_gpus': 1,
            'structured_config': 'true',
            'model_type': 'spiral',
            'run_mode': 'test',
            'extract_feature': 'true',
            'init_chkpt_dir': 'SPIRAL_model/base/xxxx/',
            'init_chkpt_file': 'st2vec-last.ckpt',
        }
        run_args_list = ['--{}={}'.format(k, v) for k, v in run_args.items()]
        print('run args: {}'.format(run_args_list))
        run_spiral.main(run_args_list)

    if pretrained_model_type == 'large':
        print("feature extraction on large model ...")
        run_args = {
            'model_save_dir': 'output/large/{}'.format(dataset),
            'config_path': 'examples/asr/conf/spiral',
            'config_name': 'spiral_large_pretrain_librilight',
            'data_dir': '/data/data/asr/librispeech',
            'test_manifest': 'manifest_json/librivox-{}.json'.format(dataset),
            'num_gpus': 1,
            'structured_config': 'true',
            'model_type': 'spiral',
            'run_mode': 'test',
            'extract_feature': 'true',
            'init_chkpt_dir': 'SPIRAL_model/large/xxxx/',
            'init_chkpt_file': 'st2vec-last.ckpt',
        }
        run_args_list = ['--{}={}'.format(k, v) for k, v in run_args.items()]
        print('run args: {}'.format(run_args_list))
        run_spiral.main(run_args_list)
        
