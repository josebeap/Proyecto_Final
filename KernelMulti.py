import multiprocessing
import numpy as np
import os
from PIL import Image
import time
from IPython.display import display

kernels = {
    'kernela' : np.array([[0, 1, 0], 
                        [0, -1, 0], 
                        [0, 0, 0]], np.int8),
    
    'kernelb' : np.array([[0, 0, 0, 0, 0], 
                        [0, 0, 1, 0, 0], 
                        [0, 0, -1, 0, 0], 
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0]], np.int8),
    
    'kernelc' : np.array([[0, 0, -1, 0, 0], 
                        [0, 0, 3, 0, 0], 
                        [0, 0, -3, 0, 0], 
                        [0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0]], np.int8),
    
    'kerneld' : np.array([[-1, 2, -1], 
                        [2, -4, 2], 
                        [-1, 2, -1]], np.int8),
    
    'kernele' : np.array([[-1, 2, -1], 
                        [2, -4, 2], 
                        [0, 0, 0]], np.int8),
    
    'kernelf' : np.array([[-1, 2, -2, 2, -1], 
                        [2, -6, 8, -6, 2], 
                        [-2, 8, -12, 8, -2], 
                        [2, -6, 8, -6, 2],
                        [-1, 2, -2, 2, -1]], np.int8),
    
    'kernelg' : np.array([[-1, 2, -2, 2, -1], 
                        [2, -6, 8, -6, 2], 
                        [-2, 8, -12, 8, -2], 
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0]], np.int8),
    
    'kernelh1sv' : np.array([[-1, 0, 1], 
                        [-2, 0, 2], 
                        [-1, 0, 1]], np.int8),
    
    'kernelh2sh' : np.array([[-1, -2, -1], 
                        [0, 0, 0], 
                        [1, 2, 1]], np.int8),
    
    'kerneli' : np.array([[-1, -1, -1], 
                        [-1, 8, -1], 
                        [-1, -1, -1]], np.int8),
    
    'kerneljpv' : np.array([[-1, 0, 1], 
                        [-1, 0, 1], 
                        [-1, 0, 1]], np.int8),
    
    'kerneljph' : np.array([[-1, -1, -1], 
                        [0, 0, 0], 
                        [1, 1, 1]], np.int8)
}

def cargar_imagen(ruta):
    return Image.open(ruta).convert('L')

def guardar_imagen(imagen, ruta):
    imagen.save(ruta)

def aplicar_filtro_y_estadisticas(imagen, filtro):

    # Convertir imagen a un array de numpy
    pixels = np.array(imagen)
    
    # Preparar el array de salida
    resultado = np.zeros_like(pixels)
    # Aplicar el filtro seleccionado
    if filtro == "filtroa":
       
        kernel_size = kernels['kernela'].shape[0]
        half_kernel_size = kernel_size // 2
        
        for i in range(half_kernel_size, pixels.shape[0] - half_kernel_size):
            for j in range(half_kernel_size, pixels.shape[1] - half_kernel_size):
                # Aplica el kernel
                gy = np.sum(np.multiply(pixels[i-half_kernel_size:i+half_kernel_size+1, j-half_kernel_size:j+half_kernel_size+1], kernels['kernela']))
                resultado[i, j] = min(255, np.abs(gy))
                
    elif filtro == "filtrob":
        # Calcula el tamaño del kernel
        kernel_size = kernels['kernelb'].shape[0]
        half_kernel_size = kernel_size // 2
        
        for i in range(half_kernel_size, pixels.shape[0] - half_kernel_size):
            for j in range(half_kernel_size, pixels.shape[1] - half_kernel_size):
                # Aplica el kernel
                gy = np.sum(np.multiply(pixels[i-half_kernel_size:i+half_kernel_size+1, j-half_kernel_size:j+half_kernel_size+1], kernels['kernelb']))
                resultado[i, j] = min(255, np.abs(gy))
                
    elif filtro == "filtroc":
        # Calcula el tamaño del kernel
        kernel_size = kernels['kernelc'].shape[0]
        half_kernel_size = kernel_size // 2
        
        for i in range(half_kernel_size, pixels.shape[0] - half_kernel_size):
            for j in range(half_kernel_size, pixels.shape[1] - half_kernel_size):
                # Aplica el kernel
                gy = np.sum(np.multiply(pixels[i-half_kernel_size:i+half_kernel_size+1, j-half_kernel_size:j+half_kernel_size+1], kernels['kernelc']))
                resultado[i, j] = min(255, np.abs(gy))
                
    elif filtro == "filtrod":
        # Calcula el tamaño del kernel
        kernel_size = kernels['kerneld'].shape[0]
        half_kernel_size = kernel_size // 2
        
        for i in range(half_kernel_size, pixels.shape[0] - half_kernel_size):
            for j in range(half_kernel_size, pixels.shape[1] - half_kernel_size):
                # Aplica el kernel
                gy = np.sum(np.multiply(pixels[i-half_kernel_size:i+half_kernel_size+1, j-half_kernel_size:j+half_kernel_size+1], kernels['kerneld']))
                resultado[i, j] = min(255, np.abs(gy))
                
    elif filtro == "filtroe":
        # Calcula el tamaño del kernel
        kernel_size = kernels['kernele'].shape[0]
        half_kernel_size = kernel_size // 2
        
        for i in range(half_kernel_size, pixels.shape[0] - half_kernel_size):
            for j in range(half_kernel_size, pixels.shape[1] - half_kernel_size):
                # Aplica el kernel
                gy = np.sum(np.multiply(pixels[i-half_kernel_size:i+half_kernel_size+1, j-half_kernel_size:j+half_kernel_size+1], kernels['kernele']))
                resultado[i, j] = min(255, np.abs(gy))
                
    elif filtro == "filtrof":
        # Calcula el tamaño del kernel
        kernel_size = kernels['kernelf'].shape[0]
        half_kernel_size = kernel_size // 2
        
        for i in range(half_kernel_size, pixels.shape[0] - half_kernel_size):
            for j in range(half_kernel_size, pixels.shape[1] - half_kernel_size):
                # Aplica el kernel
                gy = np.sum(np.multiply(pixels[i-half_kernel_size:i+half_kernel_size+1, j-half_kernel_size:j+half_kernel_size+1], kernels['kernelf']))
                resultado[i, j] = min(255, np.abs(gy))
                
                
    elif filtro == "filtrog":
        # Calcula el tamaño del kernel
        kernel_size = kernels['kernelg'].shape[0]
        half_kernel_size = kernel_size // 2
        
        for i in range(half_kernel_size, pixels.shape[0] - half_kernel_size):
            for j in range(half_kernel_size, pixels.shape[1] - half_kernel_size):
                # Aplica el kernel
                gy = np.sum(np.multiply(pixels[i-half_kernel_size:i+half_kernel_size+1, j-half_kernel_size:j+half_kernel_size+1], kernels['kernelg']))
                resultado[i, j] = min(255, np.abs(gy))
                
                
    elif filtro == "filtrohsv":
        # Calcula el tamaño del kernel
        kernel_size = kernels['kernelh1sv'].shape[0]
        half_kernel_size = kernel_size // 2
        
        for i in range(half_kernel_size, pixels.shape[0] - half_kernel_size):
            for j in range(half_kernel_size, pixels.shape[1] - half_kernel_size):
                # Aplica el kernel
                gy = np.sum(np.multiply(pixels[i-half_kernel_size:i+half_kernel_size+1, j-half_kernel_size:j+half_kernel_size+1], kernels['kernelh1sv']))
                resultado[i, j] = min(255, np.abs(gy))
                
                
    elif filtro == "filtrohsh":
        # Calcula el tamaño del kernel
        kernel_size = kernels['kernelh2sh'].shape[0]
        half_kernel_size = kernel_size // 2
        
        for i in range(half_kernel_size, pixels.shape[0] - half_kernel_size):
            for j in range(half_kernel_size, pixels.shape[1] - half_kernel_size):
                # Aplica el kernel
                gy = np.sum(np.multiply(pixels[i-half_kernel_size:i+half_kernel_size+1, j-half_kernel_size:j+half_kernel_size+1], kernels['kernelh2sh']))
                resultado[i, j] = min(255, np.abs(gy))
                
                
    elif filtro == "filtroi":
        # Calcula el tamaño del kernel
        kernel_size = kernels['kerneli'].shape[0]
        half_kernel_size = kernel_size // 2
        
        for i in range(half_kernel_size, pixels.shape[0] - half_kernel_size):
            for j in range(half_kernel_size, pixels.shape[1] - half_kernel_size):
                # Aplica el kernel
                gy = np.sum(np.multiply(pixels[i-half_kernel_size:i+half_kernel_size+1, j-half_kernel_size:j+half_kernel_size+1], kernels['kerneli']))
                resultado[i, j] = min(255, np.abs(gy))
                
                
    elif filtro == "filtrojpv":
        # Calcula el tamaño del kernel
        kernel_size = kernels['kerneljpv'].shape[0]
        half_kernel_size = kernel_size // 2
        
        for i in range(half_kernel_size, pixels.shape[0] - half_kernel_size):
            for j in range(half_kernel_size, pixels.shape[1] - half_kernel_size):
                # Aplica el kernel
                gy = np.sum(np.multiply(pixels[i-half_kernel_size:i+half_kernel_size+1, j-half_kernel_size:j+half_kernel_size+1], kernels['kerneljpv']))
                resultado[i, j] = min(255, np.abs(gy))
                
                
    elif filtro == "filtrojph":
        # Calcula el tamaño del kernel
        kernel_size = kernels['kerneljph'].shape[0]
        half_kernel_size = kernel_size // 2
        
        for i in range(half_kernel_size, pixels.shape[0] - half_kernel_size):
            for j in range(half_kernel_size, pixels.shape[1] - half_kernel_size):
                # Aplica el kernel
                gy = np.sum(np.multiply(pixels[i-half_kernel_size:i+half_kernel_size+1, j-half_kernel_size:j+half_kernel_size+1], kernels['kerneljph']))
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
    
    imagenes_y_filtros = [("imagenes/cb-300f.jpg", "filtrojph")]

    # Crear un pool de procesos
    pool = multiprocessing.Pool()
    
    # Procesar las imágenes en paralelo
    resultados = pool.map(procesar_imagen, imagenes_y_filtros)

    # Mostrar resultados
    for resultado in resultados:
        print(resultado)

if __name__ == "__main__":
    main()
