r = randi(5000) % take a random image out of our 5000 samples 
displayData(X(r,:));
p = predict(Theta1, Theta2, X(r,:));
fprintf('The prediction digit for this image is: %i\n', p)
  