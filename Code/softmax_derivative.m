function y = softmax_derivative(x)

y = x .* (1-x);

end