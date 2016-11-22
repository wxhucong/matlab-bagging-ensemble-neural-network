%% iris Network - Epoch=40, hidden nodes=20, batchSize=10, eta=0.1, trans=sigmoid, cost=cross, mom=0.3, reg=5.0
% import data as Num Matrix
data = iris(:,:);
inputs = (data(:,1:end-3))';
outputs = (data(:,end-2:end))';
inputLayer = length(inputs(:,1));
outputLayer = length(outputs(:,1));
hiddenLayer = [20];
nodeLayers = [inputLayer, hiddenLayer, outputLayer];
numEpochs = 40;
batchSize = 10;
eta = 0.1;
lmbda = 5.0;
mu_momentum = 0.3;
scheduleLearningRateFactor = 1.0;

% For cost functions, "cross" = cross-entropy, "quad" = quadratic, "log" = log-likelihood.
cost_function = 'cross';

% For transfer functions: sigmoid, tanh, softmax, ReLU
activationF_hidden = @sigmoid;
activationF_derivative_hidden = @sigmoid_derivative;
activationF_output = @sigmoid;
activationF_derivative_output = @sigmoid_derivative;

% Initialize all weights and biases
[weights, biases] = NetworkBuilder(nodeLayers);
% Data Seperation
split = [0.8,0.1,0.1];
[trainInputs, trainOutputs, valInputs, valOutputs, testInputs, testOutputs] = splitData(data, inputs, outputs, split);
% SGD
fileID = fopen('iris Network - Epoch=40, hidden nodes=20, batchSize=10, eta=0.1, trans=sigmoid, cost=cross, mom=0.3, reg=5.0.txt', 'w');
fprintf(fileID, 'iris Network : Epoch=40, hidden nodes=20, batchSize=10, eta=0.1, trans=sigmoid, cost=cross, mom=0.3, reg=5.0\n');
fprintf(fileID, '        |                   TRAIN                   ||                VALIDATION                ||                   TEST                   \n');
[outWeights, outBiases] = SGD(fileID, trainInputs, trainOutputs, weights, biases, ...
    numEpochs, batchSize, cost_function, eta, lmbda, mu_momentum, scheduleLearningRateFactor, ...
    activationF_hidden, activationF_derivative_hidden, activationF_output, activationF_derivative_output, ...
    valInputs, valOutputs, testInputs, testOutputs);
fclose(fileID);

%% iris Network - Epoch=40, hidden nodes=20, batchSize=10, eta=0.1, trans=relu, cost=cross, mom=0.3, reg=5.0
% import data as Num Matrix
data = iris(:,:);
inputs = (data(:,1:end-3))';
outputs = (data(:,end-2:end))';
inputLayer = length(inputs(:,1));
outputLayer = length(outputs(:,1));
hiddenLayer = [20];
nodeLayers = [inputLayer, hiddenLayer, outputLayer];
numEpochs = 40;
batchSize = 10;
eta = 0.1;
lmbda = 5.0;
mu_momentum = 0.3;
scheduleLearningRateFactor = 1.0;

% For cost functions, "cross" = cross-entropy, "quad" = quadratic, "log" = log-likelihood.
cost_function = 'cross';

% For transfer functions: sigmoid, tanh, softmax, ReLU
activationF_hidden = @ReLU;
activationF_derivative_hidden = @ReLU_derivative;
activationF_output = @softmax;
activationF_derivative_output = @softmax_derivative;

% Initialize all weights and biases
[weights, biases] = NetworkBuilder(nodeLayers);
% Data Seperation
split = [0.8,0.1,0.1];
[trainInputs, trainOutputs, valInputs, valOutputs, testInputs, testOutputs] = splitData(data, inputs, outputs, split);
% SGD
fileID = fopen('iris Network - Epoch=40, hidden nodes=20, batchSize=10, eta=0.1, trans=relu, cost=cross, mom=0.3, reg=5.0.txt', 'w');
fprintf(fileID, 'iris Network : Epoch=40, hidden nodes=20, batchSize=10, eta=0.1, trans=relu, cost=cross, mom=0.3, reg=5.0\n');
fprintf(fileID, '        |                   TRAIN                   ||                VALIDATION                ||                   TEST                   \n');
[outWeights, outBiases] = SGD(fileID, trainInputs, trainOutputs, weights, biases, ...
    numEpochs, batchSize, cost_function, eta, lmbda, mu_momentum, scheduleLearningRateFactor, ...
    activationF_hidden, activationF_derivative_hidden, activationF_output, activationF_derivative_output, ...
    valInputs, valOutputs, testInputs, testOutputs);
fclose(fileID);

%% iris Network - Epoch=40, hidden nodes=20, batchSize=10, eta=0.1, trans=relu, cost=cross, mom=0.0, reg=5.0
% import data as Num Matrix
data = iris(:,:);
inputs = (data(:,1:end-3))';
outputs = (data(:,end-2:end))';
inputLayer = length(inputs(:,1));
outputLayer = length(outputs(:,1));
hiddenLayer = [20];
nodeLayers = [inputLayer, hiddenLayer, outputLayer];
numEpochs = 40;
batchSize = 10;
eta = 0.1;
lmbda = 5.0;
mu_momentum = 0.0;
scheduleLearningRateFactor = 1.0;

% For cost functions, "cross" = cross-entropy, "quad" = quadratic, "log" = log-likelihood.
cost_function = 'cross';

% For transfer functions: sigmoid, tanh, softmax, ReLU
activationF_hidden = @ReLU;
activationF_derivative_hidden = @ReLU_derivative;
activationF_output = @softmax;
activationF_derivative_output = @softmax_derivative;

% Initialize all weights and biases
[weights, biases] = NetworkBuilder(nodeLayers);
% Data Seperation
split = [0.8,0.1,0.1];
[trainInputs, trainOutputs, valInputs, valOutputs, testInputs, testOutputs] = splitData(data, inputs, outputs, split);
% SGD
fileID = fopen('iris Network - Epoch=40, hidden nodes=20, batchSize=10, eta=0.1, trans=relu, cost=cross, mom=0.0, reg=5.0.txt', 'w');
fprintf(fileID, 'iris Network : Epoch=40, hidden nodes=20, batchSize=10, eta=0.1, trans=relu, cost=cross, mom=0.0, reg=5.0\n');
fprintf(fileID, '        |                   TRAIN                   ||                VALIDATION                ||                   TEST                   \n');
[outWeights, outBiases] = SGD(fileID, trainInputs, trainOutputs, weights, biases, ...
    numEpochs, batchSize, cost_function, eta, lmbda, mu_momentum, scheduleLearningRateFactor, ...
    activationF_hidden, activationF_derivative_hidden, activationF_output, activationF_derivative_output, ...
    valInputs, valOutputs, testInputs, testOutputs);
fclose(fileID);


%% MNIST Network - Epoch=30, hidden nodes=30, batchSize=10, eta=3.0, trans=sigmoid, cost=quad, mom=0.3, reg=5.0
inputs = trn;
outputs = trnAns;
inputLayer = length(inputs(:,1));
outputLayer = length(outputs(:,1));
hiddenLayer = [30];
nodeLayers = [inputLayer, hiddenLayer, outputLayer];
numEpochs = 30;
batchSize = 10;
eta = 3.0;
lmbda = 5.0;
mu_momentum = 0.3;
scheduleLearningRateFactor = 1.0;
% For cost functions, "cross" = cross-entropy, "quad" = quadratic, "log" = log-likelihood.
cost_function = 'quad';

% For transfer functions: sigmoid, tanh, softmax, ReLU
activationF_hidden = @sigmoid;
activationF_derivative_hidden = @sigmoid_derivative;
activationF_output = @sigmoid;
activationF_derivative_output = @sigmoid_derivative;

% Initialize all weights and biases
[weights, biases] = NetworkBuilder(nodeLayers);
% Data Seperation
split = [0.8,0.1,0.1];
[trainInputs, trainOutputs, valInputs, valOutputs, testInputs, testOutputs] = splitData(inputs, inputs, outputs, split);
% SGD
fileID = fopen('MNIST Network - Epoch=30, hidden nodes=30, batchSize=10, eta=3.0, trans=sigmoid, cost=quad, mom=0.3, reg=5.0.txt', 'w');
fprintf(fileID, 'MNIST Network : Epoch=30, hidden nodes=30, batchSize=10, eta=3.0, trans=sigmoid, cost=quad, mom=0.3, reg=5.0\n');
fprintf(fileID, '        |                   TRAIN                   ||                VALIDATION                ||                   TEST                   \n');
[outWeights, outBiases] = SGD(fileID, trainInputs, trainOutputs, weights, biases, ...
    numEpochs, batchSize, cost_function, eta, lmbda, mu_momentum, scheduleLearningRateFactor, ...
    activationF_hidden, activationF_derivative_hidden, activationF_output, activationF_derivative_output, ...
    valInputs, valOutputs, testInputs, testOutputs);
fclose(fileID);

%% MNIST Network - Epoch=30, hidden nodes=30, batchSize=10, eta=3.0, trans=softmax, cost=log, mom=0.3, reg=0.0
inputs = trn;
outputs = trnAns;
inputLayer = length(inputs(:,1));
outputLayer = length(outputs(:,1));
hiddenLayer = [30];
nodeLayers = [inputLayer, hiddenLayer, outputLayer];
numEpochs = 30;
batchSize = 10;
eta = 3.0;
lmbda = 0.0;
mu_momentum = 0.3;
scheduleLearningRateFactor = 1.0;
% For cost functions, "cross" = cross-entropy, "quad" = quadratic, "log" = log-likelihood.
cost_function = 'log';

% For transfer functions: sigmoid, tanh, softmax, ReLU
activationF_hidden = @sigmoid;
activationF_derivative_hidden = @sigmoid_derivative;
activationF_output = @softmax;
activationF_derivative_output = @softmax_derivative;

% Initialize all weights and biases
[weights, biases] = NetworkBuilder(nodeLayers);
% Data Seperation
split = [0.8,0.1,0.1];
[trainInputs, trainOutputs, valInputs, valOutputs, testInputs, testOutputs] = splitData(inputs, inputs, outputs, split);
% SGD
fileID = fopen('MNIST Network - Epoch=30, hidden nodes=30, batchSize=10, eta=3.0, trans=softmax, cost=log, mom=0.3, reg=0.0.txt', 'w');
fprintf(fileID, 'MNIST Network : Epoch=30, hidden nodes=30, batchSize=10, eta=3.0, trans=softmax, cost=log, mom=0.3, reg=0.0\n');
fprintf(fileID, '        |                   TRAIN                   ||                VALIDATION                ||                   TEST                   \n');
[outWeights, outBiases] = SGD(fileID, trainInputs, trainOutputs, weights, biases, ...
    numEpochs, batchSize, cost_function, eta, lmbda, mu_momentum, scheduleLearningRateFactor, ...
    activationF_hidden, activationF_derivative_hidden, activationF_output, activationF_derivative_output, ...
    valInputs, valOutputs, testInputs, testOutputs);
fclose(fileID);


%% MNIST Network - Epoch=30, hidden nodes=30, batchSize=10, eta=1.0, trans=softmax, cost=log, mom=0.3, reg=5.0
inputs = trn;
outputs = trnAns;
inputLayer = length(inputs(:,1));
outputLayer = length(outputs(:,1));
hiddenLayer = [30];
nodeLayers = [inputLayer, hiddenLayer, outputLayer];
numEpochs = 30;
batchSize = 10;
eta = 1.0;
lmbda = 5.0;
mu_momentum = 0.3;
scheduleLearningRateFactor = 1.0;
% For cost functions, "cross" = cross-entropy, "quad" = quadratic, "log" = log-likelihood.
cost_function = 'log';

% For transfer functions: sigmoid, tanh, softmax, ReLU
activationF_hidden = @sigmoid;
activationF_derivative_hidden = @sigmoid_derivative;
activationF_output = @softmax;
activationF_derivative_output = @softmax_derivative;

% Initialize all weights and biases
[weights, biases] = NetworkBuilder(nodeLayers);
% Data Seperation
split = [0.8,0.1,0.1];
[trainInputs, trainOutputs, valInputs, valOutputs, testInputs, testOutputs] = splitData(inputs, inputs, outputs, split);
% SGD
fileID = fopen('MNIST Network - Epoch=30, hidden nodes=30, batchSize=10, eta=1.0, trans=softmax, cost=log, mom=0.3, reg=5.0.txt', 'w');
fprintf(fileID, 'MNIST Network : Epoch=30, hidden nodes=30, batchSize=10, eta=1.0, trans=softmax, cost=log, mom=0.3, reg=5.0\n');
fprintf(fileID, '        |                   TRAIN                   ||                VALIDATION                ||                   TEST                   \n');
[outWeights, outBiases] = SGD(fileID, trainInputs, trainOutputs, weights, biases, ...
    numEpochs, batchSize, cost_function, eta, lmbda, mu_momentum, scheduleLearningRateFactor, ...
    activationF_hidden, activationF_derivative_hidden, activationF_output, activationF_derivative_output, ...
    valInputs, valOutputs, testInputs, testOutputs);
fclose(fileID);

%% xor Network - Epoch=20, hidden nodes=[3 2], batchSize=1, eta=0.1, trans=sigmoid, cost=cross, mom=0.3, reg=5.0
% import data as Num Matrix
data = xor(:,:);
inputs = (data(:,1:end-1))';
outputs = (data(:,length(data(1,:))))';
inputLayer = 2;
outputLayer = 1;
hiddenLayer = [3 2];
nodeLayers = [inputLayer, hiddenLayer, outputLayer];
numEpochs = 20;
batchSize = 1;
eta = 0.1;
lmbda = 5.0;
mu_momentum = 0.3;
scheduleLearningRateFactor = 1.0;
% For cost functions, "cross" = cross-entropy, "quad" = quadratic, "log" = log-likelihood.
cost_function = 'cross';

% For transfer functions: sigmoid, tanh, softmax, ReLU
activationF_hidden = @sigmoid;
activationF_derivative_hidden = @sigmoid_derivative;
activationF_output = @sigmoid;
activationF_derivative_output = @sigmoid_derivative;

% Initialize all weights and biases
[weights, biases] = NetworkBuilder(nodeLayers);
% Data Seperation
trainInputs = inputs;
trainOutputs = outputs;
valInputs = inputs;
valOutputs = outputs;
testInputs = inputs;
testOutputs = outputs;

% SGD
fileID = fopen('xor Network - Epoch=20, hidden nodes=[3 2], batchSize=1, eta=0.1, trans=sigmoid, cost=cross, mom=0.3, reg=5.0.txt', 'w');
fprintf(fileID, 'xor Network: Epoch=20, hidden nodes=[3 2], batchSize=1, eta=0.1, trans=sigmoid, cost=cross, mom=0.3, reg=5.0\n');
fprintf(fileID, '        |                   TRAIN                   ||                VALIDATION                ||                   TEST                   \n');
[outWeights, outBiases] = SGD(fileID, trainInputs, trainOutputs, weights, biases, ...
    numEpochs, batchSize, cost_function, eta, lmbda, mu_momentum, scheduleLearningRateFactor, ...
    activationF_hidden, activationF_derivative_hidden, activationF_output, activationF_derivative_output, ...
    valInputs, valOutputs, testInputs, testOutputs);
fclose(fileID);

%% xor Network - Epoch=20, hidden nodes=[3 2], batchSize=1, eta=0.1, trans=tanh, cost=cross, mom=0.3, reg=5.0
% import data as Num Matrix
data = xor(:,:);
inputs = (data(:,1:end-1))';
outputs = (data(:,length(data(1,:))))';
inputLayer = 2;
outputLayer = 1;
hiddenLayer = [3 2];
nodeLayers = [inputLayer, hiddenLayer, outputLayer];
numEpochs = 20;
batchSize = 1;
eta = 0.1;
lmbda = 5.0;
mu_momentum = 0.3;
scheduleLearningRateFactor = 1.0;
% For cost functions, "cross" = cross-entropy, "quad" = quadratic, "log" = log-likelihood.
cost_function = 'cross';

% For transfer functions: sigmoid, tanh, softmax, ReLU
activationF_hidden = @tanh_rescale;
activationF_derivative_hidden = @tanh_derivative;
activationF_output = @tanh_rescale;
activationF_derivative_output = @tanh_derivative;

% Initialize all weights and biases
[weights, biases] = NetworkBuilder(nodeLayers);
% Data Seperation
trainInputs = inputs;
trainOutputs = outputs;
valInputs = inputs;
valOutputs = outputs;
testInputs = inputs;
testOutputs = outputs;

% SGD
fileID = fopen('xor Network - Epoch=20, hidden nodes=[3 2], batchSize=1, eta=0.1, trans=tanh, cost=cross, mom=0.3, reg=5.0.txt', 'w');
fprintf(fileID, 'xor Network: Epoch=20, hidden nodes=[3 2], batchSize=1, eta=0.1, trans=tanh, cost=cross, mom=0.3, reg=5.0\n');
fprintf(fileID, '        |                   TRAIN                   ||                VALIDATION                ||                   TEST                   \n');
[outWeights, outBiases] = SGD(fileID, trainInputs, trainOutputs, weights, biases, ...
    numEpochs, batchSize, cost_function, eta, lmbda, mu_momentum, scheduleLearningRateFactor, ...
    activationF_hidden, activationF_derivative_hidden, activationF_output, activationF_derivative_output, ...
    valInputs, valOutputs, testInputs, testOutputs);
fclose(fileID);


%% xor Network - Epoch=20, hidden nodes=[3 2], batchSize=1, eta=0.1, trans=relu, cost=cross, mom=0.3, reg=5.0
% import data as Num Matrix
data = xor(:,:);
inputs = (data(:,1:end-1))';
outputs = (data(:,length(data(1,:))))';
inputLayer = 2;
outputLayer = 1;
hiddenLayer = [3 2];
nodeLayers = [inputLayer, hiddenLayer, outputLayer];
numEpochs = 20;
batchSize = 1;
eta = 0.1;
lmbda = 5.0;
mu_momentum = 0.3;
scheduleLearningRateFactor = 1.0;
% For cost functions, "cross" = cross-entropy, "quad" = quadratic, "log" = log-likelihood.
cost_function = 'cross';

% For transfer functions: sigmoid, tanh, softmax, ReLU
activationF_hidden = @ReLU;
activationF_derivative_hidden = @ReLU_derivative;
activationF_output = @sigmoid;
activationF_derivative_output = @sigmoid_derivative;

% Initialize all weights and biases
[weights, biases] = NetworkBuilder(nodeLayers);
% Data Seperation
trainInputs = inputs;
trainOutputs = outputs;
valInputs = inputs;
valOutputs = outputs;
testInputs = inputs;
testOutputs = outputs;

% SGD
fileID = fopen('xor Network - Epoch=20, hidden nodes=[3 2], batchSize=1, eta=0.1, trans=relu, cost=cross, mom=0.3, reg=5.0.txt', 'w');
fprintf(fileID, 'xor Network: Epoch=20, hidden nodes=[3 2], batchSize=1, eta=0.1, trans=relu, cost=cross, mom=0.3, reg=5.0\n');
fprintf(fileID, '        |                   TRAIN                   ||                VALIDATION                ||                   TEST                   \n');
[outWeights, outBiases] = SGD(fileID, trainInputs, trainOutputs, weights, biases, ...
    numEpochs, batchSize, cost_function, eta, lmbda, mu_momentum, scheduleLearningRateFactor, ...
    activationF_hidden, activationF_derivative_hidden, activationF_output, activationF_derivative_output, ...
    valInputs, valOutputs, testInputs, testOutputs);
fclose(fileID);