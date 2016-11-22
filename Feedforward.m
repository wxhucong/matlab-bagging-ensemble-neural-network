function [ weightedInputs, activations ] = Feedforward( minibatch_x, weights, biases_minibatch, activationF_hdden, activationF_output, dropoutWeight)
    weightedInputs = cell(1, length(weights));
    activations = cell(1, length(weights)+1);
    activations{1} = minibatch_x;
    for i = 1:length(weights)
        weight = weights{i};
        bias = biases_minibatch{i};
        weightedInputs{i} = weight*activations{i} + bias;
        if (i == length(weights))
            activations{i+1} = activationF_output(weightedInputs{i});
        elseif (i==1)
            % dropout 1st hidden layer neurons
            activations{i+1} = dropoutWeight * activationF_hdden(weightedInputs{i});
        else
            activations{i+1} = activationF_hdden(weightedInputs{i});
        end
    end
end

