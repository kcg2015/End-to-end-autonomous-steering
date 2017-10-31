'''
Create and train a neural network using comma.ai's model
https://github.com/commaai/research/blob/master/train_steering_model.py
'''
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.cross_validation import train_test_split
import time
import numpy as np
from PIL import Image
import pickle
import tensorflow as tf
tf.python.control_flow_ops = tf  


# Model/Training parameters
NUM_EPOCH = 15
BATCH_SIZE = 16
H, W, CH = 160/4, 320/4, 3
LR = 1e-4
L2_REG_SCALE = 1e-4
ANGLE_OFFSET =0.15



def gen(data, labels, batch_size):

     start = 0
     end = start + batch_size
     n = data.shape[0]
     
     while True:

          # Read image data into memory as-needed
          image_batch  = data[start:end]
          label_batch = labels[start:end]
          images_tmp ,labels_tmp= [],[]
          for i, label in enumerate(label_batch):       
                 if (label[1] == 0.) and (label[2] == 0.):  # center
        		          new_label = label[0]
                 elif (label[1] == 0.) and (label[2] == 1.):  # center & flipped
        		          new_label = -1.0 * label[0]  
                 elif (label[1] == 1.) and (label[2] == 0.):  # left
                       new_label = min(1., label[0] + ANGLE_OFFSET)
                 elif (label[1] == 1.) and (label[2] == 1.):  # left & flipped
                       new_label = -1.0 * min(1., label[0] + ANGLE_OFFSET)    
                 elif (label[1] == 2.) and (label[2] == 0.): # right
                       new_label = max(-1., label[0] - ANGLE_OFFSET)
                 else:      # right and the flipped
	                  new_label = -1.0*max(-1., label[0] - ANGLE_OFFSET)
                 
                 labels_tmp.append(new_label)  
                 image = Image.open(image_batch[i]).convert('RGB')
                 # An image is flipped on the fly base on the label
                 if label[2]==1.:
                     image =image.transpose(Image.FLIP_LEFT_RIGHT)  
                 
                 #Crop part of the image (horizon and hood of the car)
                 image = image.crop((0,50, 320,140))
                 image =image.resize((W, H), Image.ANTIALIAS) 
                 image = np.asarray(image, dtype='float32')
                 images_tmp.append(image)
          images_tmp=np.array(images_tmp, dtype='float32')    
          labels_tmp=np.array(labels_tmp)
          X_batch = images_tmp
          y_batch = labels_tmp
          start += batch_size
          end += batch_size
          if start >=n:
                 start = 0
                 end = batch_size
          yield (X_batch, y_batch)       
           
             
def build_model():
	"""
	Use comma.ai's model as a baseline
	https://github.com/commaai/research/blob/master/train_steering_model.py
	"""
	ch, row, col = CH, H, W  # camera format

	model = Sequential()
	model.add(Lambda(lambda x: x/127.5 - 1.,
		input_shape=(row, col, ch),
		output_shape=(row, col, ch)))
	model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode='same', W_regularizer=l2(L2_REG_SCALE)))
	model.add(ELU())
	model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode='same', W_regularizer=l2(L2_REG_SCALE)))
	model.add(ELU())
	model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode='same', W_regularizer=l2(L2_REG_SCALE)))
	model.add(Flatten())
	model.add(Dropout(.2))
	model.add(ELU())
	model.add(Dense(512, W_regularizer=l2(L2_REG_SCALE)))
	model.add(Dropout(.5))
	model.add(ELU())
	model.add(Dense(1, W_regularizer=l2(L2_REG_SCALE)))

	model.compile(optimizer=Adam(lr=LR), loss='mean_squared_error')

	return model


if __name__=='__main__':

    # Load driving data
    with open('driving_data_mix01_recov_bdg_ent_fin_sec.p', mode='rb') as f:
    		driving_data = pickle.load(f)
    
    data, labels = driving_data['images'], driving_data['labels']
    X_train, X_val, y_train, y_val = train_test_split(data, labels, 
                                                      test_size=0.08, 
                                                      random_state=888)
    
    
    # Establish the model
    model = build_model()
    model.summary()
    
    train_gen = gen(X_train, y_train, BATCH_SIZE)
    val_gen = gen(X_val, y_val, BATCH_SIZE)
    
    # Set up the check points
    checkpoint_path="weights/test0126_mix01_bridge_entr_fin-{epoch:02d}.h5"
    checkpoint = ModelCheckpoint(checkpoint_path, verbose=1, 
                                 save_best_only=False, 
                                 save_weights_only=True, mode='auto')
    
    train_start_time = time.time()
    # Train model
    h = model.fit_generator(generator=train_gen, 
                            samples_per_epoch=X_train.shape[0],
                            nb_epoch=NUM_EPOCH, validation_data=val_gen, 
                            nb_val_samples=X_val.shape[0],callbacks=[checkpoint])
    history = h.history
    
    total_time = time.time() - train_start_time
    print('Total training time: %.2f sec (%.2f min)' % (total_time, total_time/60))
    
    
    #Save model architecture to model.json, model weights to model.h5
    json_string = model.to_json()
    with open('model_mix01_bridge_entr_fin.json', 'w') as f:
    		f.write(json_string)
    model.save_weights('model_mix01_bridge_entr_fin.h5')
    	# Save training history
    with open('train_mix01_bridge_entr_fin.p', 'wb') as f:
    		pickle.dump(history, f)
