function [outWeightsEnsemble, outBiasesEnsemble] = trainNN(baggingSize, numNetwork, nodeLayers, trainInputs, trainOutputs, numEpochs, batchSize, cost_function, eta, lmbda, mu_momentum, scheduleLearningRateFactor, activationF_hidden, activationF_derivative_hidden, activationF_output, activationF_derivative_output, firstHiddenNeuronsCount, dropout, testInputs, testOutputs)
    weightsEnsemble = cell(1, numNetwork);
    biasesEnsemble = cell(1, numNetwork);
    
    % Initialize all weights and biases
    % [weights, biases] = NetworkBuilder(nodeLayers);
    for i = 1:numNetwork
        [weightsEnsemble{i}, biasesEnsemble{i}] = NetworkBuilder(nodeLayers);
    end
    
    % SGD
    fileID = fopen('MNIST Network.txt', 'w');
    fprintf(fileID, '        |                       TRAIN                      ||                     TEST                     \n');
    %fprintf(fileID, '        |                   TRAIN                   ||                VALIDATION                ||                   TEST                   \n');
    %[outWeights, outBiases] = SGD(fileID, trainInputs, trainOutputs, weights, biases, ...
    [outWeightsEnsemble, outBiasesEnsemble] = ensembleSGD(baggingSize, numNetwork, fileID, trainInputs, trainOutputs, weightsEnsemble, biasesEnsemble, ...
        numEpochs, batchSize, cost_function, eta, lmbda, mu_momentum, scheduleLearningRateFactor, ...
        activationF_hidden, activationF_derivative_hidden, activationF_output, activationF_derivative_output, ...
        firstHiddenNeuronsCount, dropout, ...
        testInputs, testOutputs);
    fclose(fileID);
end

