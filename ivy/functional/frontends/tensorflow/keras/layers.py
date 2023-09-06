#IMPORTANT:
#this is a partial work, just to try the logic of the function before the deadline of 72h.

import numpy as np
import ivy
def avg_pooling(image, kernel_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_last'):
    # Récupérer les dimensions de l'image et du noyau
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel_size
    
    # Vérifier les arguments de format de données (channels_last ou channels_first)
    if data_format == 'channels_last':
        input_channels = 1  
    elif data_format == 'channels_first':
        input_channels = image.shape[0]  
    else:
        raise ValueError("Format de données invalide. Utilisez 'channels_last' ou 'channels_first'.")
    
    # Calculer le nombre de pixels pour lesquels le noyau peut être appliqué
    if padding == 'valid':
        output_height = (image_height - kernel_height) // strides[0] + 1
        output_width = (image_width - kernel_width) // strides[1] + 1
    elif padding == 'same':
        output_height = image_height // strides[0]
        output_width = image_width // strides[1]
    else:
        raise ValueError("Padding invalide. Utilisez 'valid' ou 'same'.")
    
    # Initialiser la matrice de sortie
    output = np.zeros((output_height, output_width))
    
    # Parcourir l'image en utilisant le noyau avec des strides
    for i in range(output_height):
        for j in range(output_width):
            # Coordonnées de début et de fin du noyau
            row_start = i * strides[0]
            row_end = row_start + kernel_height
            col_start = j * strides[1]
            col_end = col_start + kernel_width
            
            # Extraire la sous-image correspondant au noyau
            sub_image = image[row_start:row_end, col_start:col_end]
            
            # Calculer la moyenne des valeurs dans le noyau
            avg_value = np.mean(sub_image)
            
            # Stocker la valeur moyenne dans la matrice de sortie
            output[i, j] = avg_value
    
    return output

