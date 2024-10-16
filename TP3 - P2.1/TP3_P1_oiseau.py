import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import minimize
import time

# Charger l'image
image_path = "oiseau.png"
image = Image.open(image_path)

# Convertir l'image en niveaux de gris
gray_image = image.convert("L")

# Redimensionner l'image à 100x100 pixels
resized_image = gray_image.resize((100, 100))

# Recadrer une région de 50x50 pixels
cropped_image = resized_image.crop((25, 25, 75, 75))

def modify_image(image, new_mean, new_std):
    # Convertir l'image en tableau numpy
    img_array = np.array(image)
    image_array = img_array/255

    # Calculer la moyenne et l'écart-type actuels
    current_mean = np.mean(image_array)
    current_std = np.std(image_array)

    # Ajuster les valeurs des pixels
    modified_image_array = (image_array - current_mean) / current_std * new_std + new_mean
    modified_image_array = np.clip(modified_image_array, 0, 255)  # Limiter les valeurs entre 0 et 255
    modified_image_array = modified_image_array.astype(np.uint8)  # Convertir en entiers 8 bits

    # Convertir en image PIL
    modified_image = Image.fromarray(modified_image_array)
    return modified_image


# Modifier l'image pour augmenter le contraste
new_mean = 128
new_std = 64
modified_image = modify_image(cropped_image, new_mean, new_std)


def plot_histogram(image):
    # Convertir l'image en tableau numpy
    image_array = np.array(image)

    # Calculer l'histogramme
    hist, bins = np.histogram(image_array.flatten(), bins=256, range=[0, 256])

    # Afficher l'histogramme
    plt.plot(hist)
    plt.title("Histogramme de l'image")
    plt.xlabel("Valeurs de pixel")
    plt.ylabel("Fréquence")
    plt.show()


# Afficher les images à chaque étape et l'histogramme final
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes[0, 0].imshow(image)
axes[0, 0].set_title('Image originale')
axes[0, 0].axis('off')

axes[0, 1].imshow(gray_image, cmap='gray')
axes[0, 1].set_title('Image en niveaux de gris')
axes[0, 1].axis('off')

axes[0, 2].imshow(resized_image, cmap='gray')
axes[0, 2].set_title('Image redimensionnée (100x100)')
axes[0, 2].axis('off')

axes[1, 0].imshow(cropped_image, cmap='gray')
axes[1, 0].set_title('Image recadrée (50x50)')
axes[1, 0].axis('off')

axes[1, 1].imshow(modified_image, cmap='gray')
axes[1, 1].set_title('Image modifiée (contraste ajusté)')
axes[1, 1].axis('off')

axes[1, 2].axis('off')
plt.tight_layout()
plt.show()

# Afficher l'histogramme de l'image modifiée
plot_histogram(modified_image)

# Enregistrer l'image modifiée
modified_image.save("modified_image.png")


def grad(image):
    img_array = np.array(image, dtype=float) / 255.0  # Normaliser les valeurs des pixels entre 0 et 1
    m, n = img_array.shape

    grad_x = np.zeros((m, n))
    grad_y = np.zeros((m, n))

    # Calculer les dérivées partielles
    for i in range(m):
        for j in range(n):
            if i < m - 1:
                grad_x[i, j] = img_array[i + 1, j] - img_array[i, j]
            else:
                grad_x[i, j] = 0

            if j < n - 1:
                grad_y[i, j] = img_array[i, j + 1] - img_array[i, j]
            else:
                grad_y[i, j] = 0

    return grad_x, grad_y

def gradient_norm(image):
    grad_x, grad_y = grad(image)
    norm = np.sqrt(grad_x**2 + grad_y**2)
    return norm

# Charger l'image modifiée
image_path = "modified_image.png"
modified_image = Image.open(image_path)

# Calculer la norme du gradient
norm = gradient_norm(modified_image)

# Afficher la norme du gradient
plt.imshow(norm, cmap='gray')
plt.title('Norme du gradient de l\'image')
plt.axis('off')
plt.show()

# Afficher l'histogramme de la norme du gradient
plt.hist(norm.flatten(), bins=256, range=[0, 1])
plt.title("Histogramme de la norme du gradient")
plt.xlabel("Valeurs de gradient")
plt.ylabel("Fréquence")
plt.show()

# Enregistrer la norme du gradient
norm_image = Image.fromarray((norm * 255).astype(np.uint8))
norm_image.save("gradient_norm.png")

# Choisir une valeur seuil
threshold = 0.1
# Créer une image binaire où les valeurs dépassant le seuil sont définies comme True
edges = norm > threshold

# Afficher les contours détectés
plt.imshow(edges, cmap='gray')
plt.title('Contours détectés avec seuil de {}'.format(threshold))
plt.axis('off')
plt.show()

# Afficher l'histogramme de la norme du gradient
plt.hist(edges.flatten(), bins=256, range=[0, 1])
plt.title("Histogramme de la norme du gradient")
plt.xlabel("Valeurs de gradient")
plt.ylabel("Fréquence")
plt.show()

# Enregistrer la norme du gradient
edges_image = Image.fromarray((edges * 255).astype(np.uint8))
edges_image.save("edges.png")


def div(p):
    p1, p2 = p
    m, n = p1.shape
    div_p = np.zeros((m, n))

    # Calculer les dérivées partielles
    for i in range(m):
        for j in range(n):
            if i == 0:
                dpx = p1[i, j]
            elif i < m - 1:
                dpx = p1[i, j] - p1[i - 1, j]
            else:
                dpx = -p1[i - 1, j]

            if j == 0:
                dpy = p2[i, j]
            elif j < n - 1:
                dpy = p2[i, j] - p2[i, j - 1]
            else:
                dpy = -p2[i, j - 1]

            div_p[i, j] = dpx + dpy
    return div_p

# Calculer le gradient de l'image
grad_x, grad_y = grad(modified_image)

# Calculer la divergence du gradient pour obtenir le Laplacien
laplacian = div((grad_x, grad_y))

# Afficher le Laplacien de l'image
plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian of the image')
plt.colorbar(label='Laplacian values')
plt.axis('off')
plt.show()

# Afficher l'histogramme du Laplacien
plt.hist(laplacian.flatten(), bins=256, range=[laplacian.min(), laplacian.max()])
plt.title("Histogramme du Laplacien")
plt.xlabel("Valeurs du Laplacien")
plt.ylabel("Fréquence")
plt.show()

# Enregistrer le Laplacien
laplacian_image = Image.fromarray((laplacian * 255).astype(np.uint8))
laplacian_image.save("laplacian.png")

##################################

# Function to add Gaussian noise to an image
def bruit(x):
    im = np.array(x, dtype=float)
    v = im.copy()
    for i in range(np.shape(im)[0]):
        for j in range(np.shape(im)[1]):
            v[i, j] = im[i, j] + np.random.normal(0, 10)
    return v

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


# Define the objective function J(u)
def J(u, v, lambda_reg):
    u = u.reshape(v.shape)  # Assurez que u a la même forme que v
    grad_x, grad_y = grad(u)  # Calcul des gradients en x et y
    norm_squared = np.sum(grad_x**2 + grad_y**2)  # Somme des carrés des gradients à chaque point
    return 0.5 * np.sum((v - u)**2) + 0.5 * lambda_reg * norm_squared

# Define the gradient of the objective function
def grad_J(u, v, lambda_reg):
    u = u.reshape(v.shape)  # Assurez que u a la même forme que v
    grad_x, grad_y = grad(u)  # Calcul des gradients en x et y
    div_grad_u = div(grad_x, grad_y)  # Calcul de la divergence des gradients
    return (u - v - lambda_reg * div_grad_u).flatten()

# Add noise to the image
v = bruit(modified_image)

# Display both images side by side in the same subplot
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(modified_image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')
axes[1].imshow(v, cmap='gray')
axes[1].set_title('Noisy Image')
axes[1].axis('off')
plt.savefig("original_and_noisy_images.png")
plt.show()

# Save the noisy image
v_image = Image.fromarray(v.astype(np.uint8))
v_image.save("noisy_image.png")

# Calculate the gradient of the noisy image
grad_x_noisy, grad_y_noisy = grad(v)

# Calculate the divergence of the gradient to obtain the Laplacian of the noisy image
laplacian_noisy = div(grad_x_noisy, grad_y_noisy)

# Display the Laplacian of the noisy image
plt.imshow(laplacian_noisy, cmap='gray')
plt.title('Laplacian of the noisy image')
plt.colorbar(label='Laplacian values')
plt.axis('off')
plt.show()

# Save the Laplacian of the noisy image
laplacian_noisy_image = Image.fromarray((laplacian_noisy * 255).astype(np.uint8))
laplacian_noisy_image.save("laplacian_noisy.png")

# Initialize u0
u0 = v.flatten()
lambda_reg = 0.1

# Conjugate Gradient Method
def gradient_conjugue(func, grad, v, lambda_reg, x0, tol=1e-6, max_iter=100):
    x = np.array(x0, dtype=np.float64)
    r = -grad(x, v, lambda_reg)
    d = r.copy()
    iterates = [x.copy()]
    for i in range(max_iter):
        if np.linalg.norm(r) < tol:
            break
        Ap = grad(d, v, lambda_reg)
        alpha = np.dot(r, r) / np.dot(d, Ap)
        x += alpha * d
        iterates.append(x.copy())
        r_new = -grad(x, v, lambda_reg)
        if np.linalg.norm(r_new) < tol:
            break
        beta = np.dot(r_new, r_new) / np.dot(r, r)
        d = r_new + beta * d
        r = r_new
    return x, i + 1, iterates

# Execute the Conjugate Gradient method
u_cg, nit_cg, iterates_cg = gradient_conjugue(J, grad_J, v, lambda_reg, u0)

# Display the denoised image using Conjugate Gradient
u_cg_image = u_cg.reshape(v.shape)
plt.imshow(u_cg_image, cmap='gray')
plt.title('Denoised Image (Conjugate Gradient)')
plt.axis('off')
plt.savefig("denoised_cg_image.png")
plt.show()

# Save the values of the objective function for Conjugate Gradient
J_values_cg = [J(iterate, v, 0.1) for iterate in iterates_cg]
plt.plot(J_values_cg)
plt.title('Evolution of Objective Function (Conjugate Gradient)')
plt.xlabel('Iteration')
plt.ylabel('J(u)')
plt.savefig("objective_function_cg.png")
plt.show()

# BFGS Method
def methode_bfgs(func, grad, v, lambda_reg, x0):
    iterates = [x0.copy()]
    callback = lambda xk: iterates.append(xk.copy())
    res = minimize(func, x0, args=(v, lambda_reg), jac=grad, method='BFGS', options={'disp': True, "maxiter": 100, "gtol": 1e-6, "eps": 1e-8}, callback=callback)
    return res.x, res.nit, iterates

# Execute the BFGS method
u_bfgs, nit_bfgs, iterates_bfgs = methode_bfgs(J, grad_J, v, lambda_reg, u0)

# Display the denoised image using BFGS
u_bfgs_image = u_bfgs.reshape(v.shape)
plt.imshow(u_bfgs_image, cmap='gray')
plt.title('Denoised Image (BFGS)')
plt.axis('off')
plt.savefig("denoised_bfgs_image.png")
plt.show()

# Calculate and save the successive values of the objective function
J_values_bfgs = [J(iterate, v, 0.1) for iterate in iterates_bfgs]
plt.plot(J_values_bfgs)
plt.title('Evolution of Objective Function (BFGS)')
plt.xlabel('Iteration')
plt.ylabel('J(u)')
plt.savefig("objective_function_bfgs.png")
plt.show()

# Euler-Lagrange method for denoising
def euler_lagrange(u, v, lambda_reg, dt=0.1, num_iterations=100):
    for _ in range(num_iterations):
        grad_x, grad_y = grad(u)
        laplacian_u = div(grad_x, grad_y)
        u = u + dt * (v - u + lambda_reg * laplacian_u)
    return u

# Execute the Euler-Lagrange method
u_euler = euler_lagrange(u0.reshape(v.shape), v, lambda_reg)

# Display the denoised image using Euler-Lagrange
plt.imshow(u_euler, cmap='gray')
plt.title('Denoised Image (Euler-Lagrange)')
plt.axis('off')
plt.savefig("denoised_euler_image.png")
plt.show()

# Calculate and save the successive values of the objective function
J_values_euler = [J(iterate.flatten(), v, 0.1) for iterate in iterates_bfgs]
plt.plot(J_values_euler)
plt.title('Evolution of Objective Function (Euler-Lagrange)')
plt.xlabel('Iteration')
plt.ylabel('J(u)')
plt.savefig("objective_function_euler.png")
plt.show()

# Function to calculate l'Erreur quadratique moyenne racine (RMSE) between the denoised image and the original image => objectif proche de 0
def calculate_rmse(denoised, original):
    # Assurez-vous que denoised et original sont des numpy arrays de la même forme
    return np.sqrt(np.mean((denoised - original)**2))

original_image = modified_image

# Calculate RMSE after denoising for each method
rmse_bruited = calculate_rmse(u0.reshape(v.shape), original_image)  # Avant traitement
rmse_cg = calculate_rmse(u_cg.reshape(v.shape), original_image)  # Gradient Conjugué
rmse_bfgs = calculate_rmse(u_bfgs.reshape(v.shape), original_image)  # BFGS
rmse_euler = calculate_rmse(u_euler, original_image)  # Euler-Lagrange

print("RMSE avant traitement:", rmse_bruited)
print("RMSE Gradient Conjugué:", rmse_cg)
print("RMSE BFGS:", rmse_bfgs)
print("RMSE Euler-Lagrange:", rmse_euler)

# Test the influence of the parameter λ on the solution û
lambdas = [0.01, 0.05, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.5, 1.0]
rmse_values = []
for lambda_reg in lambdas:
    u_bfgs, nit_bfgs, iterates_bfgs = methode_bfgs(J, grad_J, v, lambda_reg, u0)
    u_bfgs_image1 = u_bfgs.reshape(v.shape)
    rmse = calculate_rmse(u_bfgs_image1, original_image)
    rmse_values.append(rmse)

# Plot the evolution of RMSE with respect to λ
plt.plot(lambdas, rmse_values, marker='o')
plt.xscale('log')
plt.xlabel('Lambda')
plt.ylabel('RMSE')
plt.title('Evolution of RMSE with respect to Lambda (BFGS)')
plt.savefig("rmse_lambda_BFGS.png")
plt.show()

# meileure valeur de lambda
best_lambda = lambdas[np.argmin(rmse_values)]
print("Meilleure valeur de lambda (BFGS) :", best_lambda)

# Test the influence of the parameter λ on the solution û with gradient conjugue
rmse_values_cg = []
for lambda_reg in lambdas:
    u_cg, nit_cg, iterates_cg = gradient_conjugue(J, grad_J, v, lambda_reg, u0)
    u_cg_image1 = u_cg.reshape(v.shape)
    rmse = calculate_rmse(u_cg_image1, original_image)
    rmse_values_cg.append(rmse)

# Plot the evolution of RMSE with respect to λ
plt.plot(lambdas, rmse_values_cg, marker='o')
plt.xscale('log')
plt.xlabel('Lambda')
plt.ylabel('RMSE')
plt.title('Evolution of RMSE with respect to Lambda (Conjugate Gradient)')
plt.savefig("rmse_lambda_GC.png")
plt.show()

# meileure valeur de lambda
best_lambda_cg = lambdas[np.argmin(rmse_values_cg)]
print("Meilleure valeur de lambda (Gradient Conjugué):", best_lambda_cg)

#affiche toutes les images dans le meme suplot (modified, bruité, denoised_cg, denoised_bfgs, denoised_euler)
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
axes[0].imshow(modified_image, cmap='gray')
axes[0].set_title('Image')
axes[0].axis('off')
axes[1].imshow(v, cmap='gray')
axes[1].set_title(f'Noisy Image : {rmse_bruited:.3f}')
axes[1].axis('off')
axes[2].imshow(u_cg_image, cmap='gray')
axes[2].set_title(f'Denoised (CG) : {rmse_cg:.3f}')
axes[2].axis('off')
axes[3].imshow(u_bfgs_image, cmap='gray')
axes[3].set_title(f'Denoised (BFGS) : {rmse_bfgs:.3f}')
axes[3].axis('off')
axes[4].imshow(u_euler, cmap='gray')
axes[4].set_title(f'Denoised (Euler-Lagrange) : {rmse_euler:.3f}')
axes[4].axis('off')
plt.tight_layout()
plt.savefig("bilan_denoised_images.png")
plt.show()

##################################

# débruitage de l'image u_euler avec euler-lagrange
u_euler2 = euler_lagrange(u_euler, v, lambda_reg, dt=0.1, num_iterations=1000)

# enregistrement des images débruitées
denoised_euler2_image = Image.fromarray(u_euler2.astype(np.uint8))
denoised_euler2_image.save("denoised_euler2_image.png")

# Calculate RMSE after denoising for each method
rmse_euler2 = calculate_rmse(u_euler2, original_image)  # Euler-Lagrange 2

print("RMSE Euler-Lagrange 2:", rmse_euler2)

# Display the denoised image using Euler-Lagrange  (modified, bruité, denoised_euler, denoised_euler2)
fig, axes = plt.subplots(1, 4, figsize=(30, 5))
axes[0].imshow(modified_image, cmap='gray')
axes[0].set_title('Image')
axes[0].axis('off')
axes[1].imshow(v, cmap='gray')
axes[1].set_title(f'Noisy Image : {rmse_bruited:.3f}')
axes[1].axis('off')
axes[2].imshow(u_euler, cmap='gray')
axes[2].set_title(f'Denoised (Euler-Lagrange) : {rmse_euler:.3f}')
axes[2].axis('off')
axes[3].imshow(u_euler2, cmap='gray')
axes[3].set_title(f'Denoised (Euler-Lagrange 2) : {rmse_euler2:.3f}')
axes[3].axis('off')
plt.tight_layout()
plt.savefig("bilan_test_denoised_images.png")
plt.show()


