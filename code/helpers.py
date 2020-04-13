import torch
class Helper():

	def __init__(self):
		pass
		
	@staticmethod
	def scale_shift_uniform(a=0,b=1,*size):
		return torch.rand(size=(size))*(a-b)+b

	@staticmethod
	def list_np_to_sensor(list_of_arrays, stack=True):
		if stack:
			return torch.stack([array for array in list_of_arrays])
		else: [array for array in list_of_arrays]
