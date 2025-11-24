# learn_correspondances.py

# This script learns the correspondances between two point clouds using a neural network.

# Import necessary libraries
import numpy as np
from CorrespondanceNet import CorrespondanceNet

def learn_correspondances(P1, P2):
    """
    Learn the correspondances between two point clouds using a neural network.
    """
    # Create the neural network
    correspondance_net = CorrespondanceNet()
    # Learn the correspondances
    correspondances = correspondance_net.compute_correspondances(P1, P2)
    
    # Output:
    print(f"Correspondances: {correspondances}")
    print(f"Correspondances shape: {correspondances.shape}")
    
    return correspondances


def __main__():
    # Create example point clouds
    P1 = np.random.rand(100, 3)
    P2 = np.random.rand(100, 3)
    # Learn the correspondances
    correspondances = learn_correspondances(P1, P2)
    # Save the correspondances
    np.savetxt("correspondances.csv", correspondances, delimiter=",")

if __name__ == "__main__":
    __main__()