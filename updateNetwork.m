function [newWeights, newBiases, newPreviousWeightDelta] = updateNetwork(errors_minibatch, eta, lmbda, activations, weights, biases, batchSize, mu_momentum, previousWeightDelta, trainSize)
    newWeights = cell(1, length(weights));
    newBiases = cell(1, length(biases));
    newPreviousWeightDelta = cell(1, length(previousWeightDelta));
    for i = length(activations):-1:2
        % add momentum
        curWeightDelta = - eta/batchSize * (errors_minibatch{i-1} * (activations{i-1})'); 
        weightDelta = curWeightDelta + mu_momentum .* previousWeightDelta{i-1};
        
        newWeights{i-1} = (1-eta*lmbda/trainSize) * weights{i-1} + weightDelta;
        newPreviousWeightDelta{i-1} = weightDelta;
        newBiases{i-1} = biases{i-1} - eta/batchSize * sum(errors_minibatch{i-1},2);
    end
end

