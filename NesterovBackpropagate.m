function [ errors ] = NesterovBackpropagate(minibatch_y, weightedInputs, activations, weights, cost_function, activationF_hidden, activationF_derivative_hidden, activationF_output, activationF_derivative_output, mu_momentum, previousWeightDelta, dropoutWeight)
    errors = cell(1, length(activations)-1);
    if strcmp(cost_function, 'log')
        errors{end} = (activations{end} - minibatch_y);
    elseif strcmp(cost_function, 'cross')
        % softmax as output neuron activationF when Cost function is "cross"
        if(isequal(activationF_output, @softmax))
            errors{end} = (activations{end} - minibatch_y);
        else
            errors{end} = (activationF_derivative_output(weightedInputs{end})) ./ activations{end} ./ (1-activations{end}) .* (activations{end} - minibatch_y);
        end
    else %strcmp(cost_function, 'quad')
        errors{end} = (activations{end} - minibatch_y) .* (activationF_derivative_output(weightedInputs{end}));
    end
    for i = length(activations)-1:-1:2
        % here we are using weights{i} + mu_momentum .*
        % previousWeightDelta{i} instead of weights{i}, this means we are
        % using Nesterov accelerated gradient
        errors{i-1} = ((weights{i} + mu_momentum .* previousWeightDelta{i})' * errors{i}) .* (activationF_derivative_hidden(weightedInputs{i-1}));
        if(i==2)
            errors{i-1} = dropoutWeight * errors{i-1};
        end
    end
end
