from cnp.cnp import RegressionCNP
from cnp.trainer import RegressionTrainer
import json
import torch


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