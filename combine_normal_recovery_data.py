'''
Combine preprocessed data for normal and recovery driving
'''
import numpy as np
import pickle

print 'Combining data for normal and recovery driving into one pickle file...'

FILE_N='driving_data_mix01_recov_bdg_ent.p'
FILE_R='driving_data_recovery_final_section.p'
FILE_C='driving_data_mix01_recov_bdg_ent_fin_sec.p'


with open(FILE_N, 'rb') as f:
	normal = pickle.load(f)

with open(FILE_R, 'rb') as f:
	recov = pickle.load(f)


images = np.concatenate((normal['images'], recov['images']))
labels = np.concatenate((normal['labels'], recov['labels']))

driving_data = {'images': images, 'labels': labels}
with open(FILE_C, mode='wb') as f:
	pickle.dump(driving_data, f)

 
# "Sanity" check 
if len(normal['labels'])+len(recov['labels'])==len(driving_data['labels']):
   print 'DONE!'
