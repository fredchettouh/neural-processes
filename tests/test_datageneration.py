from cnp.datageneration import PolynomialRegression, \
    GaussianProcess, TwoDImageRegression


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

    x_values, func_falues = generator.generate_curves(
        mu_gen=mu_gen, sigma_gen=sigma_gen, mu_noise=mu_noise,
        sigma_noise=sigma_noise, num_instances_train=num_instances,
        purpose='train', seed=None)

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


def test_create_shuffled_linspace():
    xdim = [10, 1]
    xmin = -2
    xmax = 2
    steps = 400
    for dim in xdim:
        generator = GaussianProcess(
            xdim=dim, range_x=(xmin, xmax), steps=steps)

        test_data = generator._create_shuffled_linspace()

        num_obs, num_dim = test_data.size()

        assert (num_obs == steps)
        assert (num_dim == dim)


def test_GaussianProcess():
    xdim = [2, 1, 10]
    xmin = -2
    xmax = 2
    num_instances = [64, 20, 10]
    steps = [200, 400, 500]
    purposes = ['train', 'vali', 'test']

    for d, s, p, nm in zip(xdim, steps, purposes, num_instances):

        gp = GaussianProcess(xdim=d, range_x=(xmin, xmax), steps=s)

        x_values, func_x = gp.generate_curves(
            noise=1e-4, length_scale=0.4, gamma=1,
            num_instances_train=64, num_instances_vali=20,
            num_instances_test=10, purpose=p)

        batch_size, steps, dimx = x_values.size()
        assert (steps == s and dimx == d and batch_size == nm)


def test_TwoDImageRegression():
    mnist = TwoDImageRegression(
        width=28,
        height=28,
        scale_mean=0.5,
        scale_std=0.5,
        link='~/.pytorch/MNIST_data/',
        share_train_data=0.8)

    assert(len(mnist._trainset) == 48000)
    assert(len(mnist._valiset) == 12000)

    x_values, func_x = mnist.generate_curves(
            num_instances_train=64,
            num_instances_vali=None,
            num_instances_test=None,
            purpose='train')

    batch_size, num_pixels, dim = x_values.size()
    batch_size_y, num_pixels_y, dim_y = x_values.size()

    assert(batch_size == 64 and num_pixels == mnist._width * mnist._height)
    assert(batch_size_y == 64 and num_pixels_y == mnist._width * mnist._height)
