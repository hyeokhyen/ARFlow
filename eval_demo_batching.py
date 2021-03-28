import os
import numpy as np

# dir_ref = '/home/ubuntu/dataset/freeweights/Concentration_Curl/gxYRRLluNWg/0.000_10.000/optical_flow'
dir_ref = './demo_ref'
list_ref = os.listdir(dir_ref)
list_ref = [item for item in list_ref if item[-4:] == '.npy']
list_ref.sort()

num_batch = 12
dir_result = f'./demo_b_{num_batch}'
list_result = os.listdir(dir_result)
list_result = [item for item in list_result if item[-4:] == '.npy']
list_result.sort()

num_batch = 15
dir_result_2 = f'./demo_b_{num_batch}'

list_diff = []
list_diff_ref = []
for ref_frame in list_ref:
	try:
		# file_result = dir_result + f'/result_{int(ref_frame[:-4])}.npy'
		file_result = dir_result + f'/{ref_frame}'
		result_mat = np.load(file_result)
		# print (result_mat)
		# print (result_mat.shape)
		print ('load from ...', file_result)
		
		# file_result_2 = dir_result_2 + f'/result_{int(ref_frame[:-4])}.npy'
		file_result_2 = dir_result_2 + f'/{ref_frame}'
		result_mat_2 = np.load(file_result_2)
		# print (result_mat_2)
		# print (result_mat_2.shape)
		print ('load from ...', file_result_2)

		diff = np.sum(np.abs(result_mat_2-result_mat))
		print (diff)
		# assert np.array_equal(result_mat_2, result_mat)
		list_diff.append(diff)

		file_ref = dir_ref + f'/{ref_frame}'
		ref_mat = np.load(file_ref)
		# print (ref_mat)
		# print (ref_mat.shape)
		print ('load from ...', file_ref)

		diff_ref = np.sum(np.abs(ref_mat-result_mat))
		print (diff_ref)
		# assert np.array_equal(ref_mat, result_mat)
		list_diff_ref.append(diff_ref)
	except:
		break
	
	print ('--------------------')

print (list_diff)
print (list_diff_ref)
print (np.amin(list_diff), np.argmin(list_diff), np.amax(list_diff), np.argmax(list_diff))
print (np.amin(list_diff_ref), np.argmin(list_diff_ref), np.amax(list_diff_ref), np.argmax(list_diff_ref))
