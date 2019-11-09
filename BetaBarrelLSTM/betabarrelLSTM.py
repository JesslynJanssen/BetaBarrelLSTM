
import numpy
import keras
import keras.models 
import pandas as pd
import random
import sys

def BetaBarrel(calc):
	

    #Desired beta barrel sequence length
	sequenceLength = 150  

	#Import text
	sequences = pd.read_csv("/Users/jesslynjanssen/Desktop/BetaBarrelLSTM/FASTA.csv" , sep = ';')
	column = sequences['Sequence']
	text = '\n'.join(column)
	characters = sorted(list(set(text)))
	chars_indices = dict((c , i) for i , c in enumerate(characters))
	indices_chars = dict((i , c) for i , c in enumerate(characters))

	#Generate sentences and next characters
	maxlength = 70
	step = 1
	sentences = []
	next_chars = []
	for i in range(0 , len(text) - maxlength , step):
		sent = text[i : i + maxlength]
		character = text[i + maxlength]
		sentences.append(sent)
		next_chars.append(character)

	#Vectorise - (sentances , sentance length , characters)
	X = numpy.zeros((len(sentences) , maxlength , len(characters)) , numpy.int)
	Y = numpy.zeros((len(sentences) , len(characters)) , numpy.int)

	#One-hot encoding
	for i , sentence in enumerate(sentences):
		for t , character in enumerate(sentence):
			X[i , t , chars_indices[character]] = 1
		Y[i , chars_indices[next_chars[i]]] = 1

	#Setup neural network
	model = keras.models.Sequential()
	model.add(keras.layers.LSTM(128 , input_shape = (maxlength , len(characters))))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.core.Dense(200 , activation = 'relu'))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.Dense(len(characters) , activation = 'softmax'))

	#Compile model
	model.compile(keras.optimizers.Adam(lr = 0.01) , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

	if calc == 'train':
		#TensorBoard log (tensorboard --logdir=./logs)
		tensorboard = keras.callbacks.TensorBoard(log_dir = './' , histogram_freq = 1 , write_grads = True)
		#Early stopping
		stopping = keras.callbacks.EarlyStopping(monitor = 'val_loss' , patience = 50)
		#Train model
		model.summary()
		model.fit(X , Y , batch_size = 8192 , epochs = 1000 , validation_split = 0.2 , verbose = 2 , callbacks = [tensorboard , stopping])
		#Save Model
		#model.save('FASTA.h5')
		model.save('FASTA.h5')

	elif calc == 'generate':
		#Load Model
		model.load_weights('FASTA.h5')
		#Generate
		print('--------------------')
		start_index = random.randint(0 , len(text) - maxlength - 1)
		sentence = text[start_index : start_index + maxlength]
		print('Starting sequence:' , sentence)
		for iter in range(sequenceLength):
			x_pred = numpy.zeros((1 , maxlength , len(characters)))
			for t , character in enumerate(sentence):
				x_pred[0 , t , chars_indices[character]] = 1.0
			preds = model.predict(x_pred , verbose = 0)[0]
			preds = preds#[-1]
			temperature = 1.0
			preds = numpy.asarray(preds).astype('float64')
			preds[preds == 0.0] = 0.0000001
			preds = numpy.log(preds) / temperature
			exp_preds = numpy.exp(preds)
			preds = exp_preds / numpy.sum(exp_preds)
			probas = numpy.random.multinomial(1 , preds , 1)
			next_index = numpy.argmax(probas)
			next_char = indices_chars[next_index]
			sentence = sentence[1 : ] + next_char
			sys.stdout.write(next_char)
			sys.stdout.flush()




if __name__ == '__main__':
	calc = sys.argv[1]
	sets = sys.argv[2]


	if sets == 'BettaBarrel' or sets == 'betabarrel':
		BetaBarrel(calc)

