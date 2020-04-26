from experiment import Experiment
from ax.service import managed_loop
from functools import partial


class HyperParam:

    def __init__(self,
                 test_parameters,
                 total_trials,
                 objective_name='MSE',
                 random_seed=4):
        """

        Parameters
        ----------
        total_trials : int
            scalar that indicates the number of

        test_parameters : list

            list of dictionaries in Ax format that indicates what values to be tested.
            Values that are not explicitly defined will default, i.e. number of epochs

        objective_name : str
            Name of the objective to be minimized

        random_seed : int
        """
        self.test_parameters = test_parameters
        self.total_trials = total_trials
        self.objective_name = objective_name
        self._random_seed = random_seed


    def train_evaluate(self, trainloader, valiloader, parameterization):
        trainer = Experiment(**parameterization)
        weights = trainer.run_training(trainloader)
        evaluation = trainer.run_test(weights, valiloader)
        return evaluation

    def run_experiment(self, trainloader, valiloader):
        partial_func = partial(self.train_evaluate, trainloader, valiloader)
        return managed_loop.optimize(
            parameters=self.test_parameters,
            evaluation_function=partial_func,
            objective_name='MSE',
            random_seed=self._random_seed,
            total_trials=self.total_trials
        )
