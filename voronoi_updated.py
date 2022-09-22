""" 
Program to compute Voronoi diagram using JFA.

@author yisiox
@version September 2022
"""

import matplotlib.pyplot as plt
import cupy as cp
from random import sample

# global variables
x_dim = 256
y_dim = 256
noSeeds = 8

# diagram is represented as a 2d array where each element is
# x coord of source * y_dim + y coord of source
ping = cp.full((x_dim, y_dim), -1, dtype = int)
pong = None

#for colours
cmap = plt.get_cmap("hsv")
cmap.set_under("black")

def generateRandomSeeds(n):
    """
    Function to generate n random seeds.

    @param n The number of seeds to generate.
    """
    global ping, pong

    if n > x_dim * y_dim:
        print("Error: Number of seeds greater than number of pixels.")
        return

    # take sample of cartesian product
    coords = [(x, y) for x in range(x_dim) for y in range(y_dim)]
    seeds = sample(coords, n)
    for i in range(n):
        x, y = seeds[i]
        ping[x, y] = x * y_dim + y
    pong = cp.copy(ping)


# Elementwise kernel that applies basic hash function for colour mapping.
displayKernel = cp.ElementwiseKernel(
        "int64 x",
        "int64 y",
        f"y = (x < 0) ? x : x % 103",
        "displayTransform")

def displayDiagram(frame, graph):
    """
    Function to display and save the current state of the diagram.

    @param frame The current frame.
    """
    output = cp.asnumpy(displayKernel(graph))
    plt.imshow(output, cmap = cmap, vmin = 0.0, interpolation = "none")
    plt.savefig(f"frames/voronoi_frame_{frame}.png")
    plt.show()


# CUDA Kernel for making 1 pass of JFA
# Python int is long long in C
voronoiKernel = cp.RawKernel(r"""
    extern "C" __global__
    void voronoiPass(const long long step, const long long xDim, const long long yDim, 
                     const long long *ping, long long *pong) {
        
        /* Index the point being processed */
        long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        long long stp = blockDim.x * gridDim.x;
        long long N   = xDim * yDim;
        for (long long i = idx; i < N; i += stp) {
            
            /* Enumerate neighbours */
            int dydx[] = {-1, 0, 1};
            for (int j = 0; j < 3; ++j) 
            for (int k = 0; k < 3; ++k) {
                
                /* Get index of current neighbour being processed */
                long long dx  = step * dydx[j] * yDim;
                long long dy  = step * dydx[k];
                long long s   = i + dx + dy;

                /* Check if invalid neighbour */
                if (s < 0 || s >= N || ping[s] == -1)
                    continue;

                /* Check if current point is unpopulated and populate if so */
                if (pong[i] == -1) {
                    pong[i] = ping[s];
                    continue;
                }

                /* Calculate distances */
                long long x1, y1, x2, y2, x3, y3;
                x1 = i / yDim;
                y1 = i % yDim;
                x2 = pong[i] / yDim;
                y2 = pong[i] % yDim;
                x3 = ping[s] / yDim;
                y3 = ping[s] % yDim;
                long long curr_dist = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
                long long jump_dist = (x1 - x3) * (x1 - x3) + (y1 - y3) * (y1 - y3);

                if (jump_dist < curr_dist)
                    pong[i] = ping[s];
            }
        }
    }
    """, "voronoiPass")


def JFAVoronoiDiagram():
    global ping, pong
    # compute initial step size
    step = max(x_dim, y_dim) // 2
    # initalise frame number and display original state
    frame = 0
    displayDiagram(frame, ping)
    # iterate while step size is greater than 0
    while step:
        #grid size, block size and arguments
        voronoiKernel((min(x_dim, 1024),), (min(y_dim, 1024),), (step, x_dim, y_dim, ping, pong))
        #swap read and write graphs and update variables
        ping, pong = pong, ping
        frame += 1
        step //= 2
        #display current state
        displayDiagram(frame, ping)

# driver code
def main():
    generateRandomSeeds(noSeeds)
    JFAVoronoiDiagram()

if __name__ == "__main__":
    main()
