'''Train the neural networks, and save the parameter for later use'''

import os
import nn_model
import nn_collect
import nn_params

training, validating, testing = nn_collect.load_mnist()
net = nn_model.NeuralNetwork(nn_params.LAYER_SIZES, learning_rate=1.0, mini_batch_size=16, epochs=20)

def train():
    net.fit(training, validating)
    net.save(nn_params.MODEL_FILE)  

print os.path.join(os.curdir, 'models', nn_params.MODEL_FILE)
if os.path.exists(os.path.join(os.curdir, 'models', nn_params.MODEL_FILE)):
    response = raw_input('pre-trained model exist, retrain? [y/N] ').strip().lower()
    if response.find('y') != -1:
        train()
    else:
        net.load(nn_params.MODEL_FILE)
else:
    train()

counter = 0
correct = 0
for test_case in testing:
    data = test_case[0]
    label = test_case[1]
    result = net.predict(data)
    if label == result:
        correct += 1
    counter += 1

print 'Correct rate:', float(correct)/ counter * 100


