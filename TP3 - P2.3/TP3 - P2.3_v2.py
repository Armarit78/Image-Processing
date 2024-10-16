import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import minimize_scalar
np.set_printoptions(threshold=np.inf)

# Charger l'image originale
image_path = "modified_image.png"
image_originale = Image.open(image_path).convert('L')
image_originale = np.array(image_originale, dtype=float)

# Créer un masque pour les zones masquées
mask = np.ones_like(image_originale)
mask[20:24, 20:24] = 0
print(mask.shape)

# Appliquer le masque à l'image
v = mask * image_originale

# Afficher les images et le masque
fig, axes = plt.subplots(1, 3, figsize=(24, 6))
axes[0].imshow(image_originale, cmap='gray')
axes[0].set_title('Image Originale')
axes[0].axis('off')
axes[1].imshow(mask, cmap='gray')
axes[1].set_title('Masque')
axes[1].axis('off')
axes[2].imshow(v, cmap='gray')
axes[2].set_title('Image avec Masque Appliqué')
axes[2].axis('off')
plt.show()

# Sauvegarder l'image avec le masque appliqué
v_masque_image = Image.fromarray(v.astype(np.uint8))
v_masque_image.save("image_avec_masque.png")

# Définir les fonctions de régularisation
def phi_alpha(s, alpha):
    return np.abs(s) - alpha * np.log((alpha + np.abs(s)) / alpha)

def grad_phi_alpha(grad_u, alpha):
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

# Fonction pour calculer J(u)
def compute_J(u_flat, v, lam, alpha, mask):
    u2d = u_flat.reshape(v.shape)
    R_u = mask * u2d
    grad_u_x, grad_u_y = grad(u2d)
    reg_term = np.sum(phi_alpha(grad_u_x, alpha)) + np.sum(phi_alpha(grad_u_y, alpha))
    fit_term = 0.5 * np.linalg.norm(R_u - v) ** 2
    return fit_term + lam * reg_term

# Fonction pour calculer le gradient de J(u)
def compute_gradient_J(u_flat, v, lam, alpha, mask):
    u2d = u_flat.reshape(v.shape)
    dx, dy = grad(u2d)
    norm_grad = np.sqrt(dx ** 2 + dy ** 2)
    R_u = mask * u2d

    lambda_ = 0.00001  # Choisissez un petit lambda
    regularized_mask = mask + lambda_ * np.eye(mask.shape[0])  # Ajoute un lambda sur la diagonale
    mask_inv = np.linalg.inv(regularized_mask)

    R_star_Ru = np.dot(mask_inv.T, R_u)
    R_star_v = np.dot(mask_inv.T, v)

    phi = grad_phi_alpha(norm_grad, alpha)
    phi_x = phi * dx
    phi_y = phi * dy
    div_phi = div(phi_x, phi_y)
    gradient = (2 * R_star_v - 2 * R_star_Ru + lam * div_phi).flatten()
    return gradient

def line_search_function(eta, x, d, v, lam, alpha, mask):
    # Calculate new point along the direction d
    x_new = x + eta * d
    # Compute objective function at the new point
    return compute_J(x_new, v, lam, alpha, mask)

# Méthodes d'optimisation
def BFGS_optimization(x, epsilon, nitermax, mask):
    B = np.eye(x.size)
    niter = 0
    while np.linalg.norm(compute_gradient_J(x, v, lam, alpha, mask)) > epsilon and niter < nitermax:
        grad = compute_gradient_J(x, v, lam, alpha, mask).flatten()
        d = -B.dot(grad)
        eta = minimize_scalar(line_search_function, args=(x, d, v, lam, alpha, mask), bounds=(0, 1), method='bounded').x
        x_new = x + eta * d
        s = x_new - x
        y = compute_gradient_J(x_new, v, lam, alpha, mask).flatten() - grad
        if np.dot(y, s) > 0:
            Bs = B.dot(s)
            B = B + np.outer(s, s) / np.dot(s, y) - np.outer(Bs, Bs) / np.dot(s, Bs)
        x = x_new
        niter += 1
    return x, niter

def DFP_optimization(x, epsilon, nitermax, mask):
    B = np.eye(x.size)
    niter = 0
    while np.linalg.norm(compute_gradient_J(x, v, lam, alpha, mask)) > epsilon and niter < nitermax:
        grad = compute_gradient_J(x, v, lam, alpha, mask).flatten()
        d = -B.dot(grad)
        eta = minimize_scalar(line_search_function, args=(x, d, v, lam, alpha, mask), bounds=(0, 1), method='bounded').x
        x_new = x + eta * d
        s = x_new - x
        y = compute_gradient_J(x_new, v, lam, alpha, mask).flatten() - grad
        if np.dot(s, y) > 0:
            B_update = np.outer(s, s) / np.dot(s, y)
            B_term = np.outer(B.dot(y), y.dot(B)) / np.dot(y, B.dot(y))
            B = B + B_update - B_term
        x = x_new
        niter += 1
    return x, niter

def gradient_pas_fixe(u0, v, lam, alpha, mask, tol=1e-6, max_iter=100):
    u = u0.copy()
    niter = 0
    for _ in range(max_iter):
        grad = compute_gradient_J(u, v, lam, alpha, mask).flatten()
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        step_size = 1e-3
        u -= step_size * grad
        niter += 1
    return u.reshape(v.shape), niter

# Paramètres de l'optimisation
lam, alpha, epsilon, nitermax = 4, 0.1, 1e-6, 100
u0 = v.flatten()

# Appel des fonctions d'optimisation
u_dfp, iter_dfp = DFP_optimization(u0, epsilon, nitermax, mask)
u_bfgs, iter_bfgs = BFGS_optimization(u0, epsilon, nitermax, mask)
u_fixed_step, iter_fixed_step = gradient_pas_fixe(u0, v, lam, alpha, mask, epsilon, nitermax)

# Transformations après débruitage
denoised_dfp = u_dfp.reshape(v.shape)
denoised_bfgs = u_bfgs.reshape(v.shape)
denoised_fixed_step = u_fixed_step.reshape(v.shape)

# Créer un masque inversé pour la zone [20:24, 20:24]
inverse_mask_dfp = np.full_like(denoised_dfp, 255)
inverse_mask_dfp[20:24, 20:24] = denoised_dfp[20:24, 20:24]
inverse_mask_dfp = inverse_mask_dfp%255

inverse_mask_bfgs = np.full_like(denoised_bfgs, 255)
inverse_mask_bfgs[20:24, 20:24] = denoised_bfgs[20:24, 20:24]
inverse_mask_bfgs = inverse_mask_bfgs%255

inverse_mask_fixed_step = np.full_like(denoised_fixed_step, 255)
inverse_mask_fixed_step[20:24, 20:24] = denoised_fixed_step[20:24, 20:24]
inverse_mask_fixed_step = inverse_mask_fixed_step%255

v_dfp = inverse_mask_dfp + v
v_bfgs = inverse_mask_bfgs + v
v_fixed_step = inverse_mask_fixed_step + v

#afficher inversed_mask_dfp, inversed_mask_bfgs, inversed_mask_fixed_step
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
axes[0].imshow(inverse_mask_dfp, cmap='gray')
axes[0].set_title('dfp')
axes[0].axis('off')
axes[1].imshow(inverse_mask_bfgs, cmap='gray')
axes[1].set_title('bfgs')
axes[1].axis('off')
axes[2].imshow(inverse_mask_fixed_step, cmap='gray')
axes[2].set_title('fixed')
axes[2].axis('off')
plt.tight_layout()
plt.show()

########################

fig, axes = plt.subplots(1, 3, figsize=(20, 5))
axes[0].imshow(v, cmap='gray')
axes[0].set_title('Originale masked')
axes[0].axis('off')
axes[1].imshow(inverse_mask_dfp, cmap='gray')
axes[1].set_title('zone manquante dfp')
axes[1].axis('off')
axes[2].imshow(v_dfp, cmap='gray')
axes[2].set_title('Résultat Final dfp')
axes[2].axis('off')
plt.tight_layout()
plt.savefig("dfp_result.png")
plt.show()


fig, axes = plt.subplots(1, 3, figsize=(20, 5))
axes[0].imshow(v, cmap='gray')
axes[0].set_title('Originale masked')
axes[0].axis('off')
axes[1].imshow(inverse_mask_bfgs, cmap='gray')
axes[1].set_title('zone manquante bfgs')
axes[1].axis('off')
axes[2].imshow(v_bfgs, cmap='gray')
axes[2].set_title('Résultat Final bfgs')
axes[2].axis('off')
plt.tight_layout()
plt.savefig("bfgs_results.png")
plt.show()


fig, axes = plt.subplots(1, 3, figsize=(20, 5))
axes[0].imshow(v, cmap='gray')
axes[0].set_title('Originale masked')
axes[0].axis('off')
axes[1].imshow(inverse_mask_fixed_step, cmap='gray')
axes[1].set_title('zone manquante fixed_step')
axes[1].axis('off')
axes[2].imshow(v_fixed_step, cmap='gray')
axes[2].set_title('Résultat Final fixed_step')
axes[2].axis('off')
plt.tight_layout()
plt.savefig("fixed_step_results.png")
plt.show()

########################

def calculate_rmse(denoised, original):
    return np.sqrt(np.mean((denoised - original) ** 2))

rmse_original = calculate_rmse(v, image_originale)
rmse_dfp = calculate_rmse(v_dfp, image_originale)
rmse_bfgs = calculate_rmse(v_bfgs, image_originale)
rmse_fixed_step = calculate_rmse(v_fixed_step, image_originale)

print("RMSE avant traitement:", rmse_original)
print("RMSE DFP:", rmse_dfp)
print("RMSE BFGS:", rmse_bfgs)
print("RMSE Gradient à pas fixe:", rmse_fixed_step)

# Afficher les images finales
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
axes[0].imshow(image_originale, cmap='gray')
axes[0].set_title('Originale')
axes[0].axis('off')
axes[1].imshow(v, cmap='gray')
axes[1].set_title(f'Masked RMSE: {rmse_original:.6f}')
axes[1].axis('off')
axes[2].imshow(v_dfp, cmap='gray')
axes[2].set_title(f'DFP RMSE: {rmse_dfp:.6f}')
axes[2].axis('off')
axes[3].imshow(v_bfgs, cmap='gray')
axes[3].set_title(f'BFGS RMSE: {rmse_bfgs:.6f}')
axes[3].axis('off')
axes[4].imshow(v_fixed_step, cmap='gray')
axes[4].set_title(f'Gradient à pas fixe RMSE: {rmse_fixed_step:.6f}')
axes[4].axis('off')
plt.tight_layout()
plt.savefig("final_results.png")
plt.show()