import multiprocessing
import cv2
import numpy as np
import os
from PIL import Image
import time
from IPython.display import display

def cargar_imagen(ruta):
    return Image.open(ruta).convert('L')

def guardar_imagen(imagen, ruta):
    imagen.save(ruta)

def aplicar_filtro_y_estadisticas(imagen, filtro):
    # Definir dos kernels
    kernel1 = np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]], np.int8) # Ejemplo de kernel para enfocar
    
    kernel3 = np.array([[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, -1, 0, 0], [0, 0, 0, 0, 0],[0, 0, 0, 0, 0]], np.int8)

    # Convertir imagen a un array de numpy
    pixels = np.array(imagen)
    
    # Preparar el array de salida
    resultado = np.zeros_like(pixels)
    # Aplicar el filtro seleccionado
    if filtro == "filtro1":
       
        # Aplicar el filtro de Sobel solo en la dirección vertical
        for i in range(1, pixels.shape[0]-1):
            for j in range(1, pixels.shape[1]-1):
                gy = np.sum(np.multiply(pixels[i-1:i+2, j-1:j+2], kernel3))
                resultado[i, j] = min(255, np.abs(gy))
                
    elif filtro == "filtro2":
        # Calcula el tamaño del kernel
        kernel_size = kernel1.shape[0]
        half_kernel_size = kernel_size // 2
        
        for i in range(half_kernel_size, pixels.shape[0] - half_kernel_size):
            for j in range(half_kernel_size, pixels.shape[1] - half_kernel_size):
                # Aplica el kernel
                gy = np.sum(np.multiply(pixels[i-half_kernel_size:i+half_kernel_size+1, j-half_kernel_size:j+half_kernel_size+1], kernel1))
                resultado[i, j] = min(255, np.abs(gy))
    else:
        raise ValueError("Filtro no reconocido")
    
     # Guardar la imagen procesada
    #cv2.imwrite("Image_1_filtro1.jpg", imagen_procesada)
    imagen_procesada = Image.fromarray(resultado)
    path_resultado = 'imagen_con_bordes.jpg'
    guardar_imagen(imagen_procesada, path_resultado)
    
    # Calcular estadísticas
    dimensiones = resultado.shape
    valor_minimo = np.min(resultado)
    valor_maximo = np.max(resultado)
    valor_medio = np.mean(resultado)
    desviacion_estandar = np.std(resultado)

    return dimensiones, valor_minimo, valor_maximo, valor_medio, desviacion_estandar

def procesar_imagen(args):
    ruta_imagen, filtro = args
    imagen = cargar_imagen(ruta_imagen)
    
    # Generar un nombre de archivo de salida
    #nombre_archivo_salida = f"{ruta_imagen.split('.')[0]}_{filtro}.jpg"
    
    return aplicar_filtro_y_estadisticas(imagen, filtro)

def main():
    # Obtener la ruta actual del script
    script_dir = os.path.dirname(os.path.realpath(__file__))
    # Lista de rutas de imágenes y filtros a aplicar
    imagenes_y_filtros = [("downloads/cb-300f.jpg", "filtro2")]

    # Crear un pool de procesos
    pool = multiprocessing.Pool()
    
    # Procesar las imágenes en paralelo
    resultados = pool.map(procesar_imagen, imagenes_y_filtros)

    # Mostrar resultados
    for resultado in resultados:
        print(resultado)

if __name__ == "__main__":
    main()
