from cnp.trainer import RegressionTrainer
from cnp.helpers import Helper
from cnp.cnp import RegressionCNP
import torch


def run_experiment(config_file,
                   experiment_name,
                   google_colab,
                   results_dir='experiments/results'
                   ):
    # checking whether CUDA is available
    train_on_gpu = torch.cuda.is_available()
    if (train_on_gpu):
        print('Training on GPU!')
        config_file['experiment_params']['train_on_gpu'] = True
    else:
        print('No GPU available, training on CPU')
        config_file['experiment_params']['train_on_gpu'] = False

    CondNeuralProcess = RegressionCNP(**config_file['cnp_params'])

    trainer = RegressionTrainer(
        cnp=CondNeuralProcess,
        data_kwargs=config_file['data_kwargs'],
        **config_file['experiment_params'])

    state_dict_encoder, state_dict_decoder, state_dict_aggregator, \
        train_loss, vali_loss = trainer.run_training(
            plot_mode=config_file['train_kwargs']['plot_mode'],
            plot_progress=config_file['train_kwargs']['plot_progress'],
            batch_size_train=config_file['train_kwargs']['batch_size_train'],
            batch_size_vali=config_file['train_kwargs']['batch_size_vali'],
            print_after=config_file['train_kwargs']['print_after'])

    total_mse, task_mses = trainer.run_test(
        encoder_state_dict=state_dict_encoder,
        decoder_state_dict=state_dict_decoder,
        aggregator_state_dict=state_dict_aggregator,
        batch_size_test=config_file['train_kwargs']['batch_size_test'],
        plot_mode=config_file['train_kwargs']['plot_mode'])

    print(f"The mean squared error for this experiment is {total_mse}")

    values = [state_dict_encoder, state_dict_decoder, state_dict_aggregator,
              train_loss, vali_loss, task_mses, config_file]
    names = ['encoder', 'decoder', 'aggregator', 'train_loss', 'vali_loss',
             'task_mses', 'config_file']
    to_save = [(name, value) for name, value in zip(names, values)]

    if google_colab == 'yes':
        info = Helper.get_colab_sytstem_info()
    else:
        info = None

    Helper.save_results(results_dir, experiment_name, to_save, info)
