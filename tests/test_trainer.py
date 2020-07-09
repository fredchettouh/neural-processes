from cnp.cnp import RegressionCNP
from cnp.trainer import RegressionTrainer
import json
import torch


def test_early_stopping():
    with open('tests/fixtures/test_config_1d.json') as file:
        config = json.load(file)

    regressor = RegressionCNP(**config['cnp_params'])
    trainer = RegressionTrainer(
        config['data_kwargs'],
        cnp=regressor,
        lr=0.001,
        n_epochs=10,
        train_on_gpu=False,
        seed=None
    )
    vali_losses_1 = [0.5, 0.1, 0.09, 0.08, 0.09, 0.11]
    vali_losses_2 = [0.5, 0.1, 0.09, 0.08, 0.06, 0.11]
    vali_losses_3 = [0.5, 0.1, 0.09, 0.08, 0.09, 0.07]

    stop_training_1 = trainer.check_early_stopping(vali_losses_1, limit=2)
    stop_training_2 = trainer.check_early_stopping(vali_losses_2, limit=2)
    stop_training_3 = trainer.check_early_stopping(vali_losses_3, limit=2)

    assert stop_training_1
    assert not stop_training_2
    assert not stop_training_3


def test_check_early_stopping_rolling_mean():
    with open('tests/fixtures/test_config_1d.json') as file:
        config = json.load(file)

    regressor = RegressionCNP(**config['cnp_params'])
    trainer = RegressionTrainer(
        config['data_kwargs'],
        cnp=regressor,
        lr=0.001,
        n_epochs=10,
        train_on_gpu=False,
        seed=None
    )
    vali_losses_1 = [0.5, 0.1, 0.09, 0.02, 0.3, 0.04, 0.09,
                     0.05, 0.1, 0.2, 0.1, 0.2]
    vali_losses_2 = [0.5, 0.1, 0.09, 0.08, 0.06, 0.11]
    vali_losses_3 = [0.5, 0.1, 0.09, 0.08, 0.09, 0.07]
    vali_losses_4 = [0.5, 0.1, 0.09]

    stop_training_1 = trainer.check_early_stopping_rolling_mean(
        vali_losses_1)
    stop_training_2 = trainer.check_early_stopping_rolling_mean(
        vali_losses_2)
    stop_training_3 = trainer.check_early_stopping_rolling_mean(
        vali_losses_3)
    stop_training_4 = trainer.check_early_stopping_rolling_mean(
        vali_losses_4)

    assert not stop_training_1
    assert not stop_training_2
    assert not stop_training_3
    assert not stop_training_4


def test_run_training_simple_regression():
    with open('tests/fixtures/test_config_1d.json') as file:
        config = json.load(file)

    regressor = RegressionCNP(**config['cnp_params'])
    trainer = RegressionTrainer(
        config['data_kwargs'],
        cnp=regressor,
        lr=0.001,
        n_epochs=10,
        train_on_gpu=False,
        seed=None
    )
    plot_mode = None

    for i in range(5):
        print_after = [val for val in range(1, 11, 2)][i]
        batch_size_train = [val for val in range(1, 64, 10)][i]
        batch_size_vali = [val for val in range(1, 6, 1)][i]
        _, _, _, mean_epoch_loss, mean_vali_loss = trainer.run_training(
            print_after=print_after,
            batch_size_train=batch_size_train,
            batch_size_vali=batch_size_vali,
            plot_mode=plot_mode,
            plot_progress=False)

        assert (len(mean_epoch_loss) == 10)


def test_run_valiation_simple_regression():
    with open('tests/fixtures/test_config_1d.json') as file:
        config = json.load(file)

    regressor = RegressionCNP(**config['cnp_params'])
    trainer = RegressionTrainer(
        config['data_kwargs'],
        cnp=regressor,
        lr=0.001,
        n_epochs=10,
        train_on_gpu=False,
        seed=None
    )
    plot_mode = None
    batch_size_vali = 5

    valiloader = trainer._datagenerator.generate_loader_on_fly(
        batch_size_vali, trainer.data_kwargs, purpose='vali')

    trainer._validation_run(
        1, plot_mode=plot_mode, valiloader=valiloader)


def test_run_training_2d_pixel_regression():
    with open('tests/fixtures/test_config_greyscale.json') as file:
        config = json.load(file)

    regressor = RegressionCNP(**config['cnp_params'])
    trainer = RegressionTrainer(
        config['data_kwargs'],
        cnp=regressor,
        lr=0.001,
        n_epochs=10,
        train_on_gpu=False,
        seed=None
    )
    plot_mode = None

    for i in range(5):
        print_after = [val for val in range(1, 11, 2)][i]
        batch_size_train = [val for val in range(1, 64, 10)][i]
        batch_size_vali = [val for val in range(1, 6, 1)][i]
        _, _, _, mean_epoch_loss, mean_vali_loss = trainer.run_training(
            print_after=print_after,
            batch_size_train=batch_size_train,
            batch_size_vali=batch_size_vali,
            plot_mode=plot_mode,
            plot_progress=False
        )

        assert (len(mean_epoch_loss) == 10)


def test_run_valiation_2d_pixel_regression():
    with open('tests/fixtures/test_config_greyscale.json') as file:
        config = json.load(file)

    regressor = RegressionCNP(**config['cnp_params'])
    trainer = RegressionTrainer(
        config['data_kwargs'],
        cnp=regressor,
        lr=0.001,
        n_epochs=10,
        train_on_gpu=False,
        seed=None
    )
    plot_mode = None
    batch_size_vali = 5

    valiloader = trainer._datagenerator.generate_loader_on_fly(
        batch_size_vali, trainer.data_kwargs, purpose='vali')

    trainer._validation_run(1, plot_mode=plot_mode, valiloader=valiloader)


def test_run_test():
    with open('tests/fixtures/test_config_1d.json') as file:
        config = json.load(file)

    encoder_dict = torch.load('tests/fixtures/encoder')
    decoder_dict = torch.load('tests/fixtures/decoder')

    regressor = RegressionCNP(**config['cnp_params'])
    trainer = RegressionTrainer(
        config['data_kwargs'],
        cnp=regressor,
        lr=0.001,
        n_epochs=10,
        train_on_gpu=False,
        seed=None
    )

    trainer.run_test(encoder_state_dict=encoder_dict,
                     decoder_state_dict=decoder_dict,
                     aggregator_state_dict=None,
                     batch_size_test=config['train_kwargs'][
                         'batch_size_test'],
                     plot_mode=None
                     )
