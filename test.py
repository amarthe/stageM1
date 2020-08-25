#%%
import matplotlib.pyplot as plt
import numpy as np
import math

from gaussian_dictionary_learning import DictionaryLearningAlgorithm
from utils import Gaussians, MetricsUtils

from sklearn.decomposition import PCA

# load the demo data, this is represents one grain in a sample
X = np.load("Data/data_gaussian.npy")

phases = np.array([np.load("Data/aspim020_3ph_nps_gauss_noisemap_p" + str(i) + ".npy") for i in range(3)])
spctrs = np.array([np.loadtxt("Data/aspim020_3ph_nps_gauss_noisespectrum_p" + str(i) + ".asc") for i in range(3)])


# load the scale of the spectra (used to plot spectra)
x_scale = Gaussians().x

p_matr = []
a_matr = []
d_matr = []
dls = []

#%%
# chose the parameters for the algorithm
NUM_PHASES = 3
MU_LAPL = 0.
MU_ABS = 0.
MU_FROs = [1]
#MU_FROs = [0., 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 0., 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
c = [(i < 9) for i in range(18)]
#%% create the algorithm instance and fit the data
for i in range(len(MU_FROs)):
    print("i = " + str(i))
    MU_FRO = MU_FROs[i]
    
    dl = DictionaryLearningAlgorithm(num_phases=NUM_PHASES, mu_lapl=MU_LAPL, mu_abs=MU_ABS, mu_fro=MU_FRO)
    p_mat, a_mat = dl.fit(X, max_iter=2000, tol_function=0., initialize_matrix=False, first_column_fro=c[i])
    p_matr.append(p_mat)
    a_matr.append(a_mat)
    d_matr.append(dl.d_matr)
    dls.append(dl)

#%% Permutations
for i in range(len(MU_FROs)):
    p1 = np.argmin([MetricsUtils.spectral_angle(spctrs[0], d_matr[i][:,0]), MetricsUtils.spectral_angle(spctrs[0], d_matr[i][:,1]), MetricsUtils.spectral_angle(spctrs[0], d_matr[i][:,2])])
    p2 = np.argmin([MetricsUtils.spectral_angle(spctrs[1], d_matr[i][:,0]), MetricsUtils.spectral_angle(spctrs[1], d_matr[i][:,1]), MetricsUtils.spectral_angle(spctrs[1], d_matr[i][:,2])]) 
    p3 = np.argmin([MetricsUtils.spectral_angle(spctrs[2], d_matr[i][:,0]), MetricsUtils.spectral_angle(spctrs[2], d_matr[i][:,1]), MetricsUtils.spectral_angle(spctrs[2], d_matr[i][:,2])])
    
    permutation = [p1,p2,p3]
    
    d_matr = np.array(d_matr)
    a_matr = np.array(a_matr)
    
    temp_d = d_matr[i]
    temp_a = a_matr[i]
    
    d_matr[i] = temp_d[:,permutation]
    a_matr[i] = temp_a[permutation,:]

#%% Plot convergences
for i in range(len(MU_FROs)):
    
    name1 = "convergence_"
    name2 = "new_fro_" if c[i] else "old_fro_"
    name3 = str(MU_FROs[i])
    folder = "results/Fro vs new fro/"
    name = folder + name1 + name2 + name3 + ".png"
    
    fig1 = plt.figure(figsize=(15, 3))
    
    ax1 = fig1.add_subplot(1, 6, 1)
    ax1.plot(dls[i].losses)
    ax1.set_title('F(A,D)')
    ax2 = fig1.add_subplot(1, 6, 2)
    ax2.plot(np.maximum(np.array(dls[i].losses[:-2]) - dls[i].losses[-1], 0.01))
    ax2.set_yscale('log')
    ax2.set_title('F(A,D)_t - F(A,D)_T')
    ax3 = fig1.add_subplot(1, 6, 3)
    ax3.plot(dls[i].a_update)
    ax3.set_yscale('log')
    ax3.set_title('A update')
    ax4 = fig1.add_subplot(1, 6, 4)
    ax4.plot(dls[i].p_update)
    ax4.set_yscale('log')
    ax4.set_title('D update')
    ax6 = fig1.add_subplot(1, 6, 5)
    ax6.plot(dls[i].a_norm)
    ax6.set_title('A_Norm')
    ax7 = fig1.add_subplot(1, 6, 6)
    ax7.plot(dls[i].p_norm)
    ax7.set_title('D_Norm')
    fig1.tight_layout()
    fig1.savefig(name)

#%% Plot spectras
for i in range(len(MU_FROs)):
    
    name1 = "spectres_"
    name2 = "new_fro_" if c[i] else "old_fro_"
    name3 = str(MU_FROs[i])
    folder = "results/Fro vs new fro/"
    name = folder + name1 + name2 + name3 + ".png"

    d_mat = d_matr[i]
    
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.plot(x_scale, d_mat[: , 0], label='learned')
    plt.plot(x_scale, spctrs[0], label='real')
    plt.legend(loc="upper right")
    plt.xlim(0, 8)
    plt.xlabel("Energy [kEv]")
    plt.ylabel("Intensity")
    plt.title("Learned spectrum of phase 1")
    
    plt.subplot(132)
    plt.plot(x_scale, d_mat[ :, 1], label="learned")
    plt.plot(x_scale, spctrs[1], label="real")
    plt.legend(loc="upper right")
    plt.xlim(0, 8)
    plt.xlabel("Energy [kEv]")
    plt.ylabel("Intensity")
    plt.title("Learned spectrum of phase 2")
    
    plt.subplot(133)
    plt.plot(x_scale, d_mat[ :, 2], label="learned")
    plt.plot(x_scale, spctrs[2], label="real")
    plt.legend(loc="upper right")
    plt.xlim(0, 8)
    plt.xlabel("Energy [kEv]")
    plt.ylabel("Intensity")
    plt.title("Learned spectrum of phase 3")
    
    fig1.savefig(name)

#%% Projection
i += 0
Y = X.reshape((X.shape[0] * X.shape[1], X.shape[2]))  # flatten X

pca = PCA(n_components=2)
pca.fit(Y)
Y = pca.transform(Y)
for i in range(len(MU_FROs)):
    
    name1 = "projection_"
    name2 = "new_fro_" if c[i] else "old_fro_"
    name3 = str(MU_FROs[i])
    folder = "results/Fro vs new fro/"
    name = folder + name1 + name2 + name3 + ".png"

    s1 = pca.transform([d_matr[i][:,0]])
    s2 = pca.transform([d_matr[i][:,1]])
    s3 = pca.transform([d_matr[i][:,2]])
    ss = np.array([s1,s2,s3])
    
    fig = plt.figure()
    fig.scatter(Y[:,0],Y[:,1], color='blue')
    fig.scatter(ss[:,0,0],ss[:,0,1], color='red')
    fig.savefig
    plt.show(name)
    
#%%
#i += 1

fig2 = plt.figure(figsize=(16.5, 4.5))
plt.subplot(131)
plt.imshow(a_matr[i].T.reshape(*dl.x_shape[:2], NUM_PHASES)[:, :, 0], cmap="viridis")
#plt.imshow(phases.T.reshape(*dl.x_shape[:2], NUM_PHASES)[:, :, 0].T, cmap="viridis")
plt.grid(b=30)
plt.title("Activations of first spectrum")
plt.colorbar()
plt.clim(0, 1)

plt.subplot(132)
plt.imshow(a_matr[i].T.reshape(*dl.x_shape[:2], NUM_PHASES)[:, :, 1], cmap="viridis")
#plt.imshow(phases.T.reshape(*dl.x_shape[:2], NUM_PHASES)[:, :, 1].T, cmap="viridis")
plt.grid(b=30)
plt.title("Activations of first spectrum")
plt.colorbar()
plt.clim(0, 1)

plt.subplot(133)
plt.imshow(a_matr[i].T.reshape(*dl.x_shape[:2], NUM_PHASES)[:, :, 2], cmap="viridis")
#plt.imshow(phases.T.reshape(*dl.x_shape[:2], NUM_PHASES)[:, :, 2].T, cmap="viridis")
plt.grid(b=30)
plt.title("Activations of first spectrum")
plt.colorbar()
plt.clim(0, 1)
#%%
i = 9;

#%%
d_mat = d_matr[i]

plt.figure(figsize=(20, 5))
plt.subplot(131)
plt.plot(x_scale, d_mat[: , 0], label='learned')
plt.plot(x_scale, spctrs[0], label='real')
plt.legend(loc="upper right")
plt.xlim(0, 8)
plt.xlabel("Energy [kEv]")
plt.ylabel("Intensity")
plt.title("Learned spectrum of phase 1")

plt.subplot(132)
plt.plot(x_scale, d_mat[ :, 1], label="learned")
plt.plot(x_scale, spctrs[1], label="real")
plt.legend(loc="upper right")
plt.xlim(0, 8)
plt.xlabel("Energy [kEv]")
plt.ylabel("Intensity")
plt.title("Learned spectrum of phase 2")

plt.subplot(133)
plt.plot(x_scale, d_mat[ :, 2], label="learned")
plt.plot(x_scale, spctrs[2], label="real")
plt.legend(loc="upper right")
plt.xlim(0, 8)
plt.xlabel("Energy [kEv]")
plt.ylabel("Intensity")
plt.title("Learned spectrum of phase 3")

#%%
#measures
angle_res = []
mse_res   = []


for i in range(len(MU_FROs)):
    angle_res.append([min(MetricsUtils.spectral_angle(spctrs[0], d_matr[i][:,0]), MetricsUtils.spectral_angle(spctrs[0], d_matr[i][:,1]), MetricsUtils.spectral_angle(spctrs[0], d_matr[i][:,2])),   \
                      min(MetricsUtils.spectral_angle(spctrs[1], d_matr[i][:,0]), MetricsUtils.spectral_angle(spctrs[1], d_matr[i][:,1]), MetricsUtils.spectral_angle(spctrs[1], d_matr[i][:,2])),   \
                      min(MetricsUtils.spectral_angle(spctrs[2], d_matr[i][:,0]), MetricsUtils.spectral_angle(spctrs[2], d_matr[i][:,1]), MetricsUtils.spectral_angle(spctrs[2], d_matr[i][:,2]))])
        
angle_res = np.array(angle_res)       
# =============================================================================
#     mse_res.append([MetricsUtils.MSE_map(phases[0],a_matr[i,0].reshape(*dl.x_shape[:2])),                \
#                     MetricsUtils.MSE_map(phases[1],a_matr[i,1].reshape(*dl.x_shape[:2])),                 \
#                     MetricsUtils.MSE_map(phases[2],a_matr[i,2].reshape(*dl.x_shape[:2])) ])
# =============================================================================

#%% Projection
i += 0
Y = X.reshape((X.shape[0] * X.shape[1], X.shape[2]))  # flatten X

pca = PCA(n_components=2)
pca.fit(Y)
Y = pca.transform(Y)
s1 = pca.transform([d_matr[i][:,0]])
s2 = pca.transform([d_matr[i][:,1]])
s3 = pca.transform([d_matr[i][:,2]])
ss = np.array([s1,s2,s3])

plt.figure()
plt.scatter(Y[:,0],Y[:,1], color='blue')
plt.scatter(ss[:,0,0],ss[:,0,1], color='red')
plt.show()

#%%
n_tot = 9
avg_time = [sum([dls[i+(n_tot)*j].algo_time for i in range(n_tot)])/(n_tot) for j in range(2)]
ecarttype_time = [math.sqrt(sum([(dls[i+(n_tot)*j].algo_time - avg_time[j])**2 for i in range(n_tot)])/(n_tot)) for j in range(2)]

angle_avg = [[sum(angle_res[j*n_tot:(j+1)*n_tot,i])/n_tot for i in range(3)] for j in range(2)]
ecarttype_angle = [[math.sqrt(sum([(angle_res[k+(n_tot*j),i] - angle_avg[j][i])**2 for k in range(n_tot)])/n_tot) for i in range(3)] for j in range(2)]

avg_iter = [sum([dls[i+(n_tot)*j].num_iterations for i in range(n_tot)])/(n_tot) for j in range(2)]
ecarttype_iter = [math.sqrt(sum([(dls[i+(n_tot)*j].num_iterations - avg_iter[j])**2 for i in range(n_tot)])/(n_tot)) for j in range(2)]

#%% plot boxes 
data = [iters[n_tot*j:n_tot*(j+1)] for j in range(2)]
plt.boxplot(data)
plt.title("Num iterations (s) ")
pylab.xticks([1,2],['avec init','sans init'])
plt.show()