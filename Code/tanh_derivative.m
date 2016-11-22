function y = tanh_derivative(x)

%y = 1 - tanh(x) .^ 2;
y = 0.25 .* (1-tanh(x/2.0) .^ 2);

end