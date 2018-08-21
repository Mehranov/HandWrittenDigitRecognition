function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
X=[ones(m,1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

z=zeros(m,num_labels);
for i=1:m
  z(i,y(i))=1;
endfor

a2=sigmoid(X*Theta1');%5000*25
a2=[ones(m,1) a2];%5000*26
h=sigmoid(a2*Theta2');%5000*10
h1=log(h);
h2=log(1-h);
J1=(1/m)*((-z).*h1 - (1-z).*(h2));
J2=sum(J1);
J=sum(J2); %scalar
% -------------------------------------------------------------
T1=Theta1(:,2:end);
T2=Theta2(:,2:end);
J=J+(lambda/(2*m))*(sum(sum(T1.*T1)) + sum(sum(T2.*T2))); %scalar




% =========================================================================
%{
Keywords: ex4 tutorial backpropagation nnCostFunction

===============================

You can design your code for backpropagation based on analysis of the dimensions of all of the data objects. This tutorial uses the vectorized method, for easy comprehension and speed of execution.

Reference the four steps outlined on Page 9 of ex4.pdf.

---------------------------------

Let:

m = the number of training examples

n = the number of training features, including the initial bias unit.

h = the number of units in the hidden layer - NOT including the bias unit

r = the number of output classifications

-------------------------------

1: Perform forward propagation, see the separate tutorial if necessary.

2: \delta_3d 
3
?	  or d3 is the difference between a3 and the y_matrix. The dimensions are the same as both, (m x r).

3: z2 comes from the forward propagation process - it's the product of a1 and Theta1, prior to applying the sigmoid() function. Dimensions are (m x n) \cdot· (n x h) --> (m x h). In step 4, you're going to need the sigmoid gradient of z2. From ex4.pdf section 2.1, we know that if u = sigmoid(z2), then sigmoidGradient(z2) = u .* (1-u).

4: \delta_2d 
2
?	  or d2 is tricky. It uses the (:,2:end) columns of Theta2. d2 is the product of d3 and Theta2 (without the first column), then multiplied element-wise by the sigmoid gradient of z2. The size is (m x r) \cdot· (r x h) --> (m x h). The size is the same as z2.

Note: Excluding the first column of Theta2 is because the hidden layer bias unit has no connection to the input layer - so we do not use backpropagation for it. See Figure 3 in ex4.pdf for a diagram showing this.

5: \Delta_1? 
1
?	  or Delta1 is the product of d2 and a1. The size is (h x m) \cdot· (m x n) --> (h x n)

6: \Delta_2? 
2
?	  or Delta2 is the product of d3 and a2. The size is (r x m) \cdot· (m x [h+1]) --> (r x [h+1])

7: Theta1_grad and Theta2_grad are the same size as their respective Deltas, just scaled by 1/m.

Now you have the unregularized gradients. Check your results using ex4.m, and submit this portion to the grader.

===== Regularization of the gradient ===========

Since Theta1 and Theta2 are local copies, and we've already computed our hypothesis value during forward-propagation, we're free to modify them to make the gradient regularization easy to compute.

8: So, set the first column of Theta1 and Theta2 to all-zeros. Here's a method you can try in your workspace console:

9: Scale each Theta matrix by \lambda/m?/m. Use enough parenthesis so the operation is correct.

10: Add each of these modified-and-scaled Theta matrices to the un-regularized Theta gradients that you computed earlier.

-------------------

You're done. Use the test case (from the Resources menu) to test your code, and the ex4 script, then run the submit script.

The additional Test Case for ex4 includes the values of the internal variables discussed in the tutorial.

---------------------

Appendix:

Here are the sizes for the Ex4 digit recognition example, using the method described in this tutorial.

NOTE: The submit grader, the gradient checking process, and the additional test case all use different sized data sets.

a1: 5000x401

z2: 5000x25

a2: 5000x26

a3: 5000x10

d3: 5000x10

d2: 5000x25

Theta1, Delta1 and Theta1_grad: 25x401

Theta2, Delta2 and Theta2_grad: 10x26
%}



delta3=h-z; % 5000*10
delta2=delta3*(Theta2(:,2:end)).*sigmoidGradient(X*Theta1'); %5000*25
D1=delta2'*X;%25*401
D2=delta3'*a2;%10*26
Theta1_grad=D1/m;%25*401
Theta2_grad=D2/m;%10*26

Theta1_grad(:,2:end)=Theta1_grad(:,2:end)+(lambda/m)*Theta1(:,2:end);
Theta2_grad(:,2:end)=Theta2_grad(:,2:end)+(lambda/m)*Theta2(:,2:end);
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

  

    


end
