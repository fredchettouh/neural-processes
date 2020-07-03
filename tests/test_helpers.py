from cnp.helpers import Helper
import torch
from datetime import datetime
import os


def test_scale_shift_uniform():
    a = -1
    b = 1
    size = (1, 1)

    instance = Helper.scale_shift_uniform(a, b, *size)

    assert (instance.item() > a)
    assert (instance.item() < b)


def test_sort_arrays():
    ten_1 = torch.tensor([5, 10, 2])
    ten_2 = torch.tensor([9, 50, 100])
    ten_3 = torch.tensor([50, 30, 150])

    sorted_tensors = Helper.sort_arrays(ten_1, ten_2, ten_3)

    assert (sorted_tensors[0][0] == 2)
    assert (sorted_tensors[1][0] == 100)
    assert (sorted_tensors[2][2] == 30)


def test_create_loader():
    x_values = torch.normal(0, 1, (64, 400, 10))
    y_values = torch.normal(0, 1, (64, 400, 1))

    loader = Helper.create_loader(x_values, y_values, 64)
    assert (loader.batch_size == 64)


def test_list_np_to_tensor():
    tensor_list = [torch.normal(0, 1, (400, 10)) for i in range(64)]
    stacked_tensors = Helper.list_np_to_tensor(tensor_list)
    assert (stacked_tensors.shape[0] == 64)
    assert (stacked_tensors.shape[1] == 400)
    assert (stacked_tensors.shape[2] == 10)


def test_save_results():
    # directory, experiment_name, args

    directory = 'tests/fixtures'
    experiment_name = 'temp'
    config_temp = ('res', {'test_key': 1})
    list_temp = ('error', [1, 2, 3, 4])

    current_date = datetime.today()
    day = str(current_date.day).zfill(2)
    month = str(current_date.month).zfill(2)
    year = str(current_date.year)
    hour = str(current_date.hour).zfill(2)
    minute = str(current_date.minute).zfill(2)
    date_time = f"{year}_{month}_{day}_{hour}_{minute}"
    try:
        Helper.save_results(
            directory, experiment_name, [list_temp, config_temp]
        )
        assert (f"temp_{date_time}" in os.listdir('tests/fixtures'))
        assert 'error.txt' in os.listdir(f'tests/fixtures/temp_{date_time}') \
               and 'res.json' in os.listdir(f'tests/fixtures/temp_{date_time}')
    except Exception as e:
        print(e)

    finally:

        file_names = os.listdir(f'tests/fixtures/temp_{date_time}')
        [os.remove(f'tests/fixtures/temp_{date_time}/{file}')
         for file in file_names]
        os.rmdir(f'tests/fixtures/temp_{date_time}')
