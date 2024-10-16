import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import minimize_scalar

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

def line_search_function(eta, x, d, v, lam, alpha):
    # Calculate new point along the direction d
    x_new = x + eta * d
    # Compute objective function at the new point
    return compute_J(x_new, v, lam, alpha)

def BFGS_optimization(x, epsilon, nitermax, lam=4):
    B = np.eye(x.size)
    niter = 0
    while np.linalg.norm(compute_gradient_J(x, v, lam, alpha)) > epsilon and niter < nitermax:
        grad = compute_gradient_J(x, v, lam, alpha).flatten()
        d = -B.dot(grad)
        # Use minimize_scalar to perform an efficient line search
        eta = minimize_scalar(line_search_function, args=(x, d, v, lam, alpha), bounds=(0, 1), method='bounded').x
        x_new = x + eta * d
        s = x_new - x
        y = compute_gradient_J(x_new, v, lam, alpha).flatten() - grad
        if np.dot(y, s) > 0:
            Bs = B.dot(s)
            B = B + np.outer(s, s) / np.dot(s, y) - np.outer(Bs, Bs) / np.dot(s, Bs)
        x = x_new
        niter += 1
    return x, niter

def DFP_optimization(x, epsilon, nitermax):
    B = np.eye(x.size)
    niter = 0
    while np.linalg.norm(compute_gradient_J(x, v, lam, alpha)) > epsilon and niter < nitermax:
        grad = compute_gradient_J(x, v, lam, alpha).flatten()
        d = -B.dot(grad)
        # Use minimize_scalar for line search
        eta = minimize_scalar(line_search_function, args=(x, d, v, lam, alpha), bounds=(0, 1), method='bounded').x
        x_new = x + eta * d
        s = x_new - x
        y = compute_gradient_J(x_new, v, lam, alpha).flatten() - grad
        if np.dot(s, y) > 0:
            B_update = np.outer(s, s) / np.dot(s, y)
            B_term = np.outer(B.dot(y), y.dot(B)) / np.dot(y, B.dot(y))
            B = B + B_update - B_term
        x = x_new
        niter += 1
    return x, niter


def gradient_pas_fixe(u0, v, lam, alpha, tol=1e-6, max_iter=100):
    u = u0.copy()
    niter = 0

    for _ in range(max_iter):
        grad = compute_gradient_J(u, v, lam, alpha).flatten()
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break

        # Compute step size based on the norm of gradient
        step_size = 1e-3  # This could be a predefined constant or adaptive
        u -= step_size * grad
        niter += 1

    return u.reshape(v.shape), niter

lam = 4
epsilon = 1e-6
nitermax = 100
alpha = 0.1

u0 = v.flatten()

def calculate_rmse(denoised, original):
    return np.sqrt(np.mean((denoised - original) ** 2))

u_dfp, iter_dfp = DFP_optimization(u0, epsilon, nitermax)
u_bfgs, iter_bfgs = BFGS_optimization(u0, epsilon, nitermax)
u_fixed_step, iter_fixed_step  = gradient_pas_fixe(u0, v, lam, alpha, tol=1e-6, max_iter=100)

# Calcul de RMSE pour les différentes méthodes
rmse_original = calculate_rmse(v, image_originale)
rmse_dfp = calculate_rmse(u_dfp.reshape(v.shape), image_originale)
rmse_bfgs = calculate_rmse(u_bfgs.reshape(v.shape), image_originale)
rmse_fixed_step = calculate_rmse(u_fixed_step.reshape(v.shape), image_originale)

print("RMSE avant traitement:", rmse_original)
print("RMSE DFP:", rmse_dfp)
print("RMSE BFGS:", rmse_bfgs)
print("RMSE Gradient à pas fixe:", rmse_fixed_step)

'''
lambdas = [0.01, 0.05, 0.1, 0.15, 0.18, 0.2, 0.3, 0.5, 0.6, 0.8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 18, 20]
rmse_values = []
for lambda_reg in lambdas:
    print(f"Lambda: {lambda_reg}")
    u_bfgs1, iter_bfgs = BFGS_optimization(u0, epsilon, nitermax, lam=lambda_reg)
    u_bfgs_image1 = u_bfgs1.reshape(v.shape)
    rmse = calculate_rmse(u_bfgs_image1, image_originale)
    rmse_values.append(rmse)

# Tracé de l'évolution du RMSE en fonction de λ
plt.plot(lambdas, rmse_values, marker='o')
plt.xscale('log')
plt.xlabel('Lambda')
plt.ylabel('RMSE')
plt.title('Évolution du RMSE en fonction de Lambda (BFGS)')
plt.savefig("rmse_lambda_BFGS_2.png")
plt.show()
'''

# Affichage des résultats
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
axes[0].imshow(image_originale, cmap='gray')
axes[0].set_title('Originale RMSE')
axes[0].axis('off')

axes[1].imshow(v, cmap='gray')
axes[1].set_title(f'Noising RMSE: {rmse_original:.2f}')
axes[1].axis('off')

axes[2].imshow(u_dfp.reshape(v.shape), cmap='gray')
axes[2].set_title(f'DFP RMSE: {rmse_dfp:.2f}')
axes[2].axis('off')

axes[3].imshow(u_bfgs.reshape(v.shape), cmap='gray')
axes[3].set_title(f'BFGS RMSE: {rmse_bfgs:.2f}')
axes[3].axis('off')

axes[4].imshow(u_fixed_step.reshape(v.shape), cmap='gray')
axes[4].set_title(f'Gradient à pas fixe RMSE: {rmse_fixed_step:.2f}')
axes[4].axis('off')

plt.tight_layout()
plt.savefig("denoising_result_a=0,1.png")
plt.show()