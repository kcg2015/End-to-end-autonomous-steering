'''
This script processes the Udacity normal driving data. It parses the driving log file
(.csv format) and organize the image names and the 
labels (steering angles, camera position, flipped image of not) 
and saves into a .p file to be used in model.py
'''

import numpy as np
import pickle
import matplotlib.pyplot as plt

OFFSET_LEFT=0.15
OFFSET_RIGHT=0.15   
PCT=0.1 # the percentage of the original image to keep
   
INPUT_FILE_NAME = 'driving_log.csv'
OUTPUT_FILE_NAME = 'driving_data_mix01_v1.p'

             
with open(INPUT_FILE_NAME, mode='r') as f:
	   lines = f.readlines() 
images,labels,angles = [],[],[]

for line in lines[29:]:
    fields = line.split(',')
    s=np.random.uniform(0,1)
    if np.random.uniform(0,1)<PCT:
        # Center image
        images.append(fields[0])
        # For the label tuple, the first element is steering angle;
        # the second element is for camera position: 0->center, 1->left, 
        # and 2->right; the third element is a inidcator: 0->no flipping;
        # 1->flipping
        labels.append((float(fields[3]), 0., 0.))
        angles.append((float(fields[3])))
        
        # Left image
        images.append(fields[1][1:])  # delete the leading space
        labels.append((float(fields[3]), 1., 0.))
        angles.append(min(1,float(fields[3])+OFFSET_LEFT))
    
        # Right image
        images.append(fields[2][1:])  # delete the leading space
        labels.append((float(fields[3]), 2., 0.))
        angles.append(max(-1,float(fields[3])-OFFSET_RIGHT))
       
    # Center & flipped image
    images.append(fields[0])
    labels.append((float(fields[3]), 0., 1.))
    angles.append(-1.0*(float(fields[3])))
    
       # Left & flipped image
    images.append(fields[1][1:])  # delete the leading space
    labels.append((float(fields[3]), 1., 1.))
    angles.append(-1.0*min(1,float(fields[3])+OFFSET_LEFT))
    
       # Right & flipped image
    images.append(fields[2][1:])  # delete the leading space
    labels.append((float(fields[3]), 2., 1.))
    angles.append(-1.0*max(-1,float(fields[3])-OFFSET_RIGHT))



images,labels,angles=np.array(images),np.array(labels),np.array(angles)  
data = {'images': np.array(images), 'labels': np.array(labels)}

    # Save to pickle file
with open(OUTPUT_FILE_NAME, mode='wb') as f:
	    pickle.dump(data, f)

# Plot the distributions of the angles as a "sanity" check       
print len(angles)
plt.hist(angles,19)
plt.show() 
