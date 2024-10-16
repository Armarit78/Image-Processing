from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

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
    print(image_array)
    print(image_array.shape)

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

def bruit(x):
    im=np.array(x,dtype=float)
    v=im.copy()
    for i in range (np.shape(im)[0]):
        for j in range (np.shape(im)[1]):
            v[i,j]=im[i,j]+np.random.normal(0,10)
    return v

v=bruit(modified_image)

plt.imshow(modified_image, cmap='gray')
plt.title('Image de travail')
plt.axis('off')
plt.show()

plt.imshow(v, cmap='gray')
plt.title('bruit')
plt.axis('off')
plt.show()

# Enregistrer l'image bruitée
v_image = Image.fromarray(v.astype(np.uint8))
v_image.save("noisy_image.png")

# Calculer le gradient de l'image bruitée
grad_x_noisy, grad_y_noisy = grad(v)

# Calculer la divergence du gradient pour obtenir le Laplacien de l'image bruitée
laplacian_noisy = div((grad_x_noisy, grad_y_noisy))

# Afficher le Laplacien de l'image bruitée
plt.imshow(laplacian_noisy, cmap='gray')
plt.title('Laplacian of the noisy image')
plt.colorbar(label='Laplacian values')
plt.axis('off')
plt.show()

# Enregistrer le Laplacien de l'image bruitée
laplacian_noisy_image = Image.fromarray((laplacian_noisy * 255).astype(np.uint8))
laplacian_noisy_image.save("laplacian_noisy.png")



