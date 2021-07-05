import numpy as np
from NeuralNet import *


model  = DirectNet_20(1, 1)


# traced_fn = torch.jit.trace(model , (torch.rand(1, 1),))

traced_fn = torch.jit.script(model)
#my_script_module = torch.jit.script

# omega = np.zeros(100, dtype="float32")

omega = np.zeros(100)

model(torch.reshape(torch.tensor(omega, dtype=torch.float32), (100,1))).detach().numpy()