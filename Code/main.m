% import data as Num Matrix
inputs = trn;
outputs = trnAns;
inputLayer = length(inputs(:,1));
outputLayer = length(outputs(:,1));
% hiddenLayer = [300 30]
hiddenLayer = [30 30];
firstHiddenNeuronsCount = hiddenLayer(1);
nodeLayers = [inputLayer, hiddenLayer, outputLayer];
numEpochs = 10;%number of epochs
batchSize = 10;%mini-batch size
eta = 0.5;%learning rate
lmbda = 5.0;%regularization parameter
mu_momentum = 0.3;%momentum parameter
scheduleLearningRateFactor = 1.01;%factor of scheduled learning rate

% add dropout layer after the 1st hidden layer 
dropout = 0.1;

% train # of network for getting ensemble
numNetwork = 1;
baggingSize = 0.999999999999; % the percentage of randomly drawn subset, 0.9 means we use 90% of the data in each neural network

% For cost functions, "cross" = cross-entropy, "quad" = quadratic, "log" = log-likelihood.
cost_function = 'log';

% For transfer functions: sigmoid, tanh, softmax, ReLU
%activationF_hidden = @sigmoid;
%activationF_derivative_hidden = @sigmoid_derivative;
%activationF_output = @sigmoid;
%activationF_derivative_output = @sigmoid_derivative;

%activationF_hidden = @tanh_rescale;
%activationF_derivative_hidden = @tanh_derivative;
%activationF_output = @tanh_rescale;
%activationF_derivative_output = @tanh_derivative;

% do not use ReLu as output neuron activationF when Cost function is "cross"
%activationF_hidden = @ReLU;
%activationF_derivative_hidden = @ReLU_derivative;
%activationF_output = @softmax;
%activationF_derivative_output = @softmax_derivative;

activationF_hidden = @sigmoid;
activationF_derivative_hidden = @sigmoid_derivative;
activationF_output = @softmax;
activationF_derivative_output = @softmax_derivative;

% Data Seperation
% split = [0.8,0.1,0.1];
% [trainInputs, trainOutputs, valInputs, valOutputs, testInputs, testOutputs] = splitData(inputs, inputs, outputs, split);
split = [0.9,0.1];
[trainInputs, trainOutputs, testInputs, testOutputs] = splitData(inputs, inputs, outputs, split);

% train network
[outWeightsEnsemble, outBiasesEnsemble] = trainNN(baggingSize, numNetwork, nodeLayers, trainInputs, trainOutputs, ...
        numEpochs, batchSize, cost_function, eta, lmbda, mu_momentum, scheduleLearningRateFactor, ...
        activationF_hidden, activationF_derivative_hidden, activationF_output, activationF_derivative_output, ...
        firstHiddenNeuronsCount, dropout, ...
        testInputs, testOutputs);

