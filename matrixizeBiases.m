function [biases_matrix] = matrixizeBiases(biases, size)
    biases_matrix = cell(1, length(biases));
    for k = 1:length(biases)
        bias = repmat(biases{k},1, size);
        biases_matrix{k} = bias;
    end
end

