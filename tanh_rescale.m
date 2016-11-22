function y = tanh_rescale(x)

y = (1 + tanh(x/2.0)) ./ 2.0;

end