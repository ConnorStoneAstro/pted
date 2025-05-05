import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.spatial.distance import cdist
np.random.seed(42)

N = 100


r = np.random.normal(loc = 1, scale = 0.1, size = N)
t = np.random.uniform(0, 2*np.pi, size = N)
x = np.stack((r * np.cos(t), r * np.sin(t)), axis=-1)
y = np.stack((np.random.uniform(-1.5, 1.5, size = N), np.random.normal(loc =0, scale = 0.1, size = N)), axis=-1)

plt.scatter(x[:,0], x[:,1], color = "b")
plt.scatter(y[:,0], y[:,1], color = "r")
plt.show()

XX, YY = np.meshgrid(np.linspace(-1.5, 1.5, 100), np.linspace(-1.5, 1.5, 100))
ZZ = np.stack((XX.flatten(), YY.flatten()), axis=-1)
Ex = np.sum(np.linalg.norm(x.reshape(1,N,2) - ZZ.reshape(-1, 1, 2), axis = -1), axis = 1).reshape(XX.shape)
Ey = np.sum(np.linalg.norm(y.reshape(1,N,2) - ZZ.reshape(-1, 1, 2), axis = -1), axis = 1).reshape(YY.shape)

# plt.imshow(Ex, cmap = "winter", extent = (-1.5, 1.5, -1.5, 1.5))
# plt.scatter(x[:,0], x[:,1], color = "b")
# plt.show()
# plt.imshow(Ey, cmap = "autumn", extent = (-1.5, 1.5, -1.5, 1.5))
# plt.scatter(y[:,0], y[:,1], color = "r")
# plt.show()

def plot_demo(x, y, name, E_permute=None):

    fig, axarr = plt.subplots(1,3, figsize = (13,4))
    
    axarr[0].scatter(x[:,0], x[:,1], color = "b", label="x")
    axarr[0].scatter(y[:,0], y[:,1], color = "r", label="y")
    axarr[0].legend()
    axarr[0].set_xlabel("Dimension 1")
    axarr[0].set_ylabel("Dimension 2")
    axarr[0].set_title("Samples")
    axarr[0].set_xlim([-1.5,1.5])
    axarr[0].set_ylim([-1.5,1.5])
    axarr[0].set_aspect("equal")
    z = np.concatenate((x, y), axis = 0)
    D = cdist(z, z)
    
    hist, bins = np.histogram(D[:N, :N].ravel(), bins = 50, density=True)
    axarr[1].step(bins[:-1], hist, where="post", color = "b", linewidth = 3)
    hist, bins = np.histogram(D[N:, N:].ravel(), bins = 50, density=True)
    axarr[1].step(bins[:-1], hist, where="post", color = "r", linewidth = 3)
    hist, bins = np.histogram(D[:N, N:].ravel(), bins = 50, density=True)
    axarr[1].step(bins[:-1], hist, where="post", color = "purple", linewidth = 3)
    
    bx, by = 0.7, 0.7
    dx, dy = 0.13, 0.08
    axarr[1].add_patch(plt.Rectangle((bx,by+dy), dx, dx,fill=True, facecolor= "b", transform = axarr[1].transAxes))
    axarr[1].add_patch(plt.Rectangle((bx,by), dx, dy,fill=True, facecolor= "purple", transform = axarr[1].transAxes))
    axarr[1].add_patch(plt.Rectangle((bx+dx,by+dy), dy, dx,fill=True, facecolor= "purple", transform = axarr[1].transAxes))
    axarr[1].add_patch(plt.Rectangle((bx+dx,by), dy, dy,fill=True, facecolor= "r", transform = axarr[1].transAxes))
    axarr[1].text(bx,by+dy+0.5*dx, "x", transform=axarr[1].transAxes, va="center", ha="right")
    axarr[1].text(bx,by+0.5*dy, "y", transform=axarr[1].transAxes, va="center", ha="right")
    axarr[1].text(bx+0.5*dx,by+dy+dx, "x", transform=axarr[1].transAxes, va="bottom", ha="center")
    axarr[1].text(bx+dx+0.5*dy,by+dy+dx, "y", transform=axarr[1].transAxes, va="bottom", ha="center")
    axarr[1].set_xlabel("Distance [sample pairs]")
    axarr[1].set_title("Distance Matrix")

    E_test = (2 * np.sum(D[:N,N:]) - np.sum(D[:N,:N]) - np.sum(D[N:,N:])) / N**2
    if E_permute is not None:
        hist, bins = np.histogram(E_permute, bins = 15)
        axarr[2].step(bins[:-1], hist, where="post", color = "grey", linewidth = 3, label="permutations")
    axarr[2].axvline(E_test, linewidth = 3, color="k", label="test")
    axarr[2].legend()
    axarr[2].set_xlabel("Energy [2 * Exy - Exx - Eyy]")
    axarr[2].set_title(f"Energy Distribution (p={np.mean(E_permute > E_test):.2f})")
    axarr[2].set_xlim([0.0,0.2])
    plt.savefig(f"{name}_PTED.png", dpi=300, bbox_inches="tight")
    plt.show()

z = np.concatenate((x, y), axis = 0)
E_permute = []
for i in range(1000):
    np.random.shuffle(z)
    D = cdist(z,z)
    E_permute.append((2 * np.sum(D[:N,N:]) - np.sum(D[:N,:N]) - np.sum(D[N:,N:])) / N**2)
plot_demo(x,y, "test", E_permute)
np.random.shuffle(z)
x, y = z[:N], z[N:]
plot_demo(x,y, "permute", E_permute)
