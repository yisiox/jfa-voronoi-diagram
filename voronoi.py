""" 
Program to compute Voronoi diagram using JFA.

@author yisiox
@version September 2022
"""

import matplotlib.pyplot as plt
import cupy as cp
from random import sample

# global variables
x_dim = 4096
y_dim = 4096
noSeeds = 1024

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

displayKernel = cp.ElementwiseKernel(
        "int64 x",
        "int64 y",
        f"y = (x < 0) ? x : x % 103",
        "displayTransform")

def displayDiagram(frame, graph):
    """
    Function to display the current state of the diagram.

    @param frame The current frame.
    """
    output = cp.asnumpy(displayKernel(graph))
    plt.imshow(output, cmap = cmap, vmin = 0.0, interpolation = "none")
    plt.savefig(f"voronoi_frame_{frame}.png")
    plt.show()


voronoiKernel = cp.RawKernel(r"""
    extern "C" __global__
    void voronoiPass(const long long step, const long long xDim, const long long yDim, const long long *ping, long long *pong) {
        long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        long long stp = blockDim.x * gridDim.x;

        for (long long k = idx; k < xDim * yDim; k += stp) {
            long long dydx[] = {-1, 0, 1};
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    long long dx = (step * dydx[i]) * yDim;
                    long long dy = step * dydx[j];
                    long long src = k + dx + dy;
                    if (src < 0 || src >= xDim * yDim) 
                        continue;
                    if (ping[src] == -1)
                        continue;
                    if (pong[k] == -1) {
                        pong[k] = ping[src];
                        continue;
                    }
                    long long x1 = k / yDim;
                    long long y1 = k % yDim;
                    long long x2 = pong[k] / yDim;
                    long long y2 = pong[k] % yDim;
                    long long x3 = ping[src] / yDim;
                    long long y3 = ping[src] % yDim;
                    long long curr_dist = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
                    long long jump_dist = (x1 - x3) * (x1 - x3) + (y1 - y3) * (y1 - y3);
                    if (jump_dist < curr_dist)
                        pong[k] = ping[src];
                }
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
        voronoiKernel((min(x_dim, 1024),), (min(y_dim, 1024),), (step, x_dim, y_dim, ping, pong))
        ping, pong = pong, ping
        frame += 1
        step //= 2
        displayDiagram(frame, ping)

# driver code
def main():
    generateRandomSeeds(noSeeds)
    JFAVoronoiDiagram()

if __name__ == "__main__":
    main()
