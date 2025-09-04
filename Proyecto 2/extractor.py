import cv2
import csv
import glob
import numpy
import math
import os


def hu_moments_of_file(filename):
    """
    Genera los momentos de Hu para una imagen específica.
    Basado en el código original de hu_moments_generation.py
    """
    image = cv2.imread(filename)
    if image is None:
        print(f"Error: No se pudo cargar la imagen {filename}")
        return None
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 67, 2)

    # Invert the image so the area of the UAV is filled with 1's. This is necessary since
    # cv::findContours describes the boundary of areas consisting of 1's.
    bin = 255 - bin  # como sabemos que las figuras son negras invertimos los valores binarios para que esten en 1.

    kernel = numpy.ones((3, 3), numpy.uint8)  # Tamaño del bloque a recorrer
    # buscamos eliminar falsos positivos (puntos blancos en el fondo) para eliminar ruido.
    bin = cv2.morphologyEx(bin, cv2.MORPH_ERODE, kernel)

    contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)  # encuentra los contornos
    
    if not contours:
        print(f"Error: No se encontraron contornos en {filename}")
        return None
    
    shape_contour = max(contours, key=cv2.contourArea)  # Agarra el contorno de area maxima


    # Descomentar para chequear que estemos agarrando bien el contorno
    # Redimensionar la imagen para que quepa en la pantalla (máximo 800x600)
    # height, width = image.shape[:2]
    # max_width, max_height = 800, 600
    # 
    # if width > max_width or height > max_height:
    #     # Calcular el factor de escala
    #     scale = min(max_width/width, max_height/height)
    #     new_width = int(width * scale)
    #     new_height = int(height * scale)
    #     
    #     # Redimensionar la imagen
    #     resized_image = cv2.resize(image, (new_width, new_height))
    #     
    #     # Redimensionar el contorno correctamente
    #     resized_contour = []
    #     for point in shape_contour:
    #         x, y = point[0]
    #         new_x = int(x * scale)
    #         new_y = int(y * scale)
    #         resized_contour.append([[new_x, new_y]])
    #     resized_contour = numpy.array(resized_contour, dtype=numpy.int32)
    #     
    #     # Dibujar contorno en la imagen redimensionada
    #     cv2.drawContours(resized_image, [resized_contour], -1, (0, 255, 0), 2)
    #     cv2.imshow("test", resized_image)
    # else:
    #     # Si la imagen ya es pequeña, mostrarla tal como está
    #     cv2.drawContours(image, [shape_contour], -1, (0, 255, 0), 2)
    #     cv2.imshow("test", image)
    # 
    # cv2.waitKey(0)

    # Calculate Moments
    moments = cv2.moments(shape_contour)  # momentos de inercia
    # Calculate Hu Moments
    huMoments = cv2.HuMoments(moments)  # momentos de Hu
    # Log scale hu moments
    for i in range(0, 7):
        huMoments[i] = -1 * math.copysign(1.0, huMoments[i]) * math.log10(abs(huMoments[i]))  # Mapeo para agrandar la escala.
    return huMoments


def write_hu_moments_for_category(category_folder, writer):
    """
    Procesa todas las imágenes en una categoría específica y escribe sus momentos de Hu
    """
    # Obtener la ruta completa de la carpeta de categoría
    category_path = os.path.join('fotos', category_folder)
    
    # Buscar todas las imágenes en la carpeta (PNG, JPEG, JPG)
    image_extensions = ['*.png', '*.jpeg', '*.jpg']
    files = []
    for ext in image_extensions:
        files.extend(glob.glob(os.path.join(category_path, ext)))
        files.extend(glob.glob(os.path.join(category_path, ext.upper())))  # También extensiones en mayúsculas
    
    print(f"Procesando {len(files)} imágenes en la categoría '{category_folder}'")
    
    for file in files:
        # Obtener solo el nombre del archivo (sin la ruta)
        filename = os.path.basename(file)
        
        # Generar los momentos de Hu
        hu_moments = hu_moments_of_file(file)
        
        if hu_moments is not None:
            # Aplanar el array de momentos de Hu
            flattened = hu_moments.ravel()
            
            # Crear la fila con: [hu_moment_1, hu_moment_2, ..., hu_moment_7, category]
            row = numpy.append(flattened, category_folder)
            
            # Escribir la fila en el archivo CSV
            writer.writerow(row)
            print(f"  Procesado: {filename}")
        else:
            print(f"  Error procesando: {filename}")


def generate_hu_moments_file():
    """
    Función principal que genera el archivo CSV con todos los momentos de Hu
    """
    # Crear el directorio de archivos generados si no existe
    os.makedirs('generated-files', exist_ok=True)
    
    # Obtener todas las subcarpetas en la carpeta fotos
    fotos_path = 'fotos'
    if not os.path.exists(fotos_path):
        print(f"Error: La carpeta '{fotos_path}' no existe")
        return
    
    categories = [d for d in os.listdir(fotos_path) 
                  if os.path.isdir(os.path.join(fotos_path, d))]
    
    if not categories:
        print(f"No se encontraron subcarpetas en '{fotos_path}'")
        return
    
    print(f"Categorías encontradas: {categories}")
    
    # Crear el archivo CSV
    output_file = 'generated-files/musical-notes-hu-moments.csv'
    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Escribir el encabezado del CSV
        # header = ['hu_moment_1', 'hu_moment_2', 'hu_moment_3', 'hu_moment_4', 
        #           'hu_moment_5', 'hu_moment_6', 'hu_moment_7', 'category']
        # writer.writerow(header)
        
        # Procesar cada categoría
        for category in categories:
            print(f"\nProcesando categoría: {category}")
            write_hu_moments_for_category(category, writer)
    
    print(f"\nArchivo generado exitosamente: {output_file}")


if __name__ == "__main__":
    print("Iniciando extracción de momentos de Hu para notas musicales...")
    generate_hu_moments_file()
    print("Proceso completado.")
