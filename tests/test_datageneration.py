from cnp.datageneration import PolynomialRegression, DataGenerator


def test__create_shuffled_linspace():
    xdim = [10, 1]
    xmin = -2
    xmax = 2
    steps = 400
    for dim in xdim:
        generator = DataGenerator(xdim=dim,
                                  range_x=(xmin, xmax),
                                  steps=steps)

        test_data = generator._create_shuffled_linspace()

        num_obs, num_dim = test_data.size()

        assert(num_obs == steps)
        assert(num_dim == dim)


def test_PolynomialRegression():

    num_instances = 64
    steps = 400
    xdim = 4
    mu_gen = 0
    sigma_gen = 2
    mu_noise = 0
    sigma_noise = 0.01

    generator = PolynomialRegression(
        xdim=xdim,
        steps=steps,
        range_x=(None, None)
    )

    x_values, func_falues = generator.generate_curves(mu_gen=mu_gen,
                                                      sigma_gen=sigma_gen,
                                                      mu_noise=mu_noise,
                                                      sigma_noise=sigma_noise,
                                                      num_instances_train=num_instances,
                                                      purpose='train',
                                                      seed=None)

    num_functions, num_points, num_dims = x_values.size()

    assert (num_instances == num_functions)
    assert (steps == num_points)
    assert (num_dims == xdim)

    kwargs = dict(
        mu_gen=mu_gen,
        sigma_gen=sigma_gen,
        mu_noise=mu_noise,
        sigma_noise=sigma_noise,
        num_instances_train=num_instances,
        seed=None)
    loader = generator.generate_loader_on_fly(num_instances, kwargs, 'train')
    num_functions, num_points, num_dims = loader.dataset.tensors[0].size()
    assert (num_instances == num_functions)
    assert (steps == num_points)
    assert (num_dims == xdim)
