import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import minimize

# Charger l'image
image_path = "modified_image.png"
image_originale = Image.open(image_path).convert('L')
image_originale = np.array(image_originale, dtype=float)

# Fonction pour ajouter du bruit gaussien à une image
def bruit(x):
    im = np.array(x, dtype=float)
    v = im.copy()
    for i in range(np.shape(im)[0]):
        for j in range(np.shape(im)[1]):
            v[i, j] = im[i, j] + np.random.normal(0, 10)
    return v

# Ajouter du bruit à l'image
v = bruit(image_originale)

# Afficher les deux images côte à côte dans le même subplot
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image_originale, cmap='gray')
axes[0].set_title('Image Originale')
axes[0].axis('off')
axes[1].imshow(v, cmap='gray')
axes[1].set_title('Image Bruitée')
axes[1].axis('off')
plt.savefig("images_originale_et_bruitee.png")
plt.show()

# Sauvegarder l'image bruitée
v_image = Image.fromarray(v.astype(np.uint8))
v_image.save("image_bruitee.png")

def computePhi(s, alpha):
    s = np.asarray(s)
    abs_s = np.abs(s)
    phi_alpha = abs_s - alpha * np.log((alpha + abs_s) / alpha)
    return phi_alpha, abs_s

# Définir les valeurs de s et alpha
s = np.linspace(-5, 5, 100)
alphas = [0.05, 0.1, 0.25, 0.5, 1, 1.25, 1.5, 2]

# Tracer les courbes pour les valeurs de alpha
plt.figure(figsize=(12, 6))
for alpha in alphas:
    phi_alpha1, abs_s = computePhi(s, alpha)
    plt.plot(s, phi_alpha1, label=f'φ_{alpha}(s)')

plt.plot(s, abs_s, label=f'|s|', linestyle='--')

plt.title('Fonctions φ(s) pour différentes valeurs de α')
plt.xlabel('s')
plt.ylabel('Valeur de φ_α(s) et |s|')
plt.axhline(0, color='grey', linewidth=0.5)
plt.axvline(0, color='grey', linewidth=0.5)
plt.legend()
plt.grid(True)
plt.savefig("fonctions_phi_alpha.png")
plt.show()

def phi_alpha(s, alpha):
    return np.abs(s) - alpha * np.log((alpha + np.abs(s)) / alpha)

def phi_prime(grad_u, alpha):
    return np.sign(grad_u) * (1 - alpha / (alpha + np.abs(grad_u)))

# Afficher les fonctions phi et phi prime
s = np.linspace(-5, 5, 100)
alpha = 0.5
phi_alpha1 = phi_alpha(s, alpha)
phi_prime1 = phi_prime(s, alpha)

plt.figure(figsize=(12, 6))
plt.plot(s, phi_alpha1, label=f'φ_{alpha}(s)')
plt.plot(s, phi_prime1, label=f'φ\'_{alpha}(s)')
plt.title('Fonctions φ et φ\' pour α = 0.5')
plt.xlabel('s')
plt.ylabel('Valeur de φ_α(s) et φ\'_α(s)')
plt.axhline(0, color='grey', linewidth=0.5)
plt.axvline(0, color='grey', linewidth=0.5)
plt.legend()
plt.grid(True)
plt.show()

def grad(x):
    m, n = x.shape
    dx = np.zeros((m, n))
    dy = np.zeros((m, n))
    dx[:m-1, :] = x[1:m, :] - x[:m-1, :]
    dy[:, :n-1] = x[:, 1:n] - x[:, :n-1]
    return dx, dy

def div(gx, gy):
    m, n = gx.shape
    dxgx = np.zeros((m, n))
    dxgx[0, :] = gx[0, :]
    dxgx[1:m-1, :] = gx[1:m-1, :] - gx[0:m-2, :]
    dxgx[m-1, :] = -gx[m-2, :]
    dygy = np.zeros((m, n))
    dygy[:, 0] = gy[:, 0]
    dygy[:, 1:n-1] = gy[:, 1:n-1] - gy[:, 0:n-2]
    dygy[:, n-1] = -gy[:, n-2]
    return dxgx + dygy

def BFGS_optimization_enhanced(func, grad, x0, v, lam, alpha, nitermax):
    iterates = [x0.copy()]
    callback = lambda xk: iterates.append(xk.copy())

    # The scipy minimize function call
    res = minimize(func, x0, args=(v, lam, alpha), jac=grad, method='BFGS',
                   options={'disp': True, "maxiter": nitermax, "gtol": 1e-6, "eps": 1e-8},
                   callback=callback)

    return res.x, res.nit, iterates

# Function to compute the objective (J)
def compute_J(u_flat, v, lam, alpha):
    u2d = u_flat.reshape(v.shape)
    grad_u_x, grad_u_y = grad(u2d)
    regularization_term = np.sum(phi_alpha(grad_u_x, alpha)) + np.sum(phi_alpha(grad_u_y, alpha))
    return 0.5 * np.linalg.norm(v - u2d) ** 2 + lam * regularization_term

# Function to compute the gradient of J
def compute_gradient_J(u_flat, v, lam, alpha):
    u2d = u_flat.reshape(v.shape)
    grad_u_x, grad_u_y = grad(u2d)
    phi_prime_x = phi_prime(grad_u_x, alpha)
    phi_prime_y = phi_prime(grad_u_y, alpha)
    div_phi_prime = div(phi_prime_x, phi_prime_y)
    return (u2d - v - lam * div_phi_prime).flatten()

def calculate_rmse(denoised, original):
    return np.sqrt(np.mean((denoised - original) ** 2))

# BFGS call (Now correctly integrated)
lam = 4
alpha = 0.01
epsilon = 1e-6
nitermax = 100
u0 = v.flatten()

u_bfgs, iter_bfgs, _ = BFGS_optimization_enhanced(compute_J, compute_gradient_J, u0, v, lam, alpha, nitermax)

rmse_original = calculate_rmse(v, image_originale)
rmse_bfgs = calculate_rmse(u_bfgs.reshape(v.shape), image_originale)

print("RMSE avant traitement:", rmse_original)
print("RMSE BFGS:", rmse_bfgs)

# Affichage des résultats
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
axes[0].imshow(image_originale, cmap='gray')
axes[0].set_title('Originale RMSE')
axes[0].axis('off')

axes[1].imshow(v, cmap='gray')
axes[1].set_title(f'Noising RMSE: {rmse_original:.2f}')
axes[1].axis('off')

axes[2].imshow(u_bfgs.reshape(v.shape), cmap='gray')
axes[2].set_title(f'BFGS RMSE: {rmse_bfgs:.2f}')
axes[2].axis('off')

plt.tight_layout()
plt.show()
