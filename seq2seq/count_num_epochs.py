import math
num_samples = 949090
num_steps = pow(2, 18)
num_epochs = (num_steps*64)/num_samples 
print(math.ceil(num_epochs))
