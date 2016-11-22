function y = ReLU_derivative(x)

y = 1.0 ./ (1.0 + exp(-x));

end