from mpi4py import MPI
import numpy as np
from PIL import Image
import os
import time


# Inicializar MPI
comm = MPI.COMM_WORLD


rank = comm.Get_rank()
size = comm.Get_size()


# Definir kernels
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


# Función para cargar una imagen desde una ruta
def cargar_imagen(ruta):
    return Image.open(ruta).convert('L')

# Función para guardar una imagen en una ruta específica
def guardar_imagen(imagen, ruta):
    imagen.save(ruta)

# Función para aplicar un filtro y calcular estadísticas a una porción de la imagen
def aplicar_filtro_y_estadisticas(imagen, filtro, rank, size):
    pixels = np.array(imagen)

    # Comunicar el kernel a todos los procesos
    kernel = None
    if rank == 0:
        kernel = kernels[filtro]
    kernel = comm.bcast(kernel, root=0)

    # Calcular la porción de la imagen que debe procesar este proceso
    rows_per_process = pixels.shape[0] // size
    start_row = rank * rows_per_process
    end_row = (rank + 1) * rows_per_process if rank != size - 1 else pixels.shape[0]

    resultado = np.zeros_like(pixels[start_row:end_row])

    # Aplicar el filtro a la porción de la imagen asignada a este proceso
    kernel_size = kernel.shape[0]
    half_kernel_size = kernel_size // 2

    for i in range(start_row + half_kernel_size, end_row - half_kernel_size):
        for j in range(half_kernel_size, pixels.shape[1] - half_kernel_size):
            gy = np.sum(np.multiply(pixels[i - half_kernel_size:i + half_kernel_size + 1, j - half_kernel_size:j + half_kernel_size + 1], kernel))
            resultado[i - start_row, j] = min(255, np.abs(gy))

    return resultado

# Función para procesar una imagen con un filtro específico
def procesar_imagen(args, rank, size):
    ruta_imagen, filtro = args
    imagen = cargar_imagen(ruta_imagen)
    return aplicar_filtro_y_estadisticas(imagen, filtro, rank, size)

def aplicar_filtro_y_estadisticasDos(imagen, filtro, nombre):

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
 
    imagen_procesada = Image.fromarray(resultado)
    path_resultado = str(nombre)+' imagen_procesada.jpg'
   

    guardar_imagen(imagen_procesada, path_resultado)
    
    # Calcular estadísticas
    dimensiones = resultado.shape
    valor_minimo = np.min(resultado)
    valor_maximo = np.max(resultado)
    valor_medio = np.mean(resultado)
    desviacion_estandar = np.std(resultado)

    return dimensiones, valor_minimo, valor_maximo, valor_medio, desviacion_estandar

# Función principal
def main():
    
    imagenes_y_filtros = [("imagenes/apple_pear.jpg", "kernelb")]

    resultados = []
    
    start_timeUno = time.time()
    # Procesar imágenes en paralelo
    for args in imagenes_y_filtros:
        resultado_parcial = procesar_imagen(args, rank, size)
        resultados.append(resultado_parcial)

    # Recopilar resultados en el proceso maestro
    if rank == 0:
        for i in range(1, size):
            resultado_parcial = comm.recv(source=i)
            resultados.extend(resultado_parcial)

        # Guardar la imagen procesada
        for idx, resultado in enumerate(resultados):
            imagen_procesada = Image.fromarray(resultado)
            path_resultado = f'imagen_procesada_{idx}.jpg'
            guardar_imagen(imagen_procesada, path_resultado)
    end_timeUno = time.time()
    tiempo_paralelo = end_timeUno - start_timeUno
    print("Tiempo Multiprocessing", end_timeUno - start_timeUno)      
    
    
    start_time = time.time()
    #Secuencial
    for i in imagenes_y_filtros:
        imagenUno = cargar_imagen(i[0])
        nombre = str(i[0])
        aplicar_filtro_y_estadisticasDos(imagenUno, "filtrob", nombre)
    end_time = time.time()
    tiempo_secuencial = end_time - start_time
    print("Tiempo secuencial", end_time - start_time)
    
    print("Aceleración", tiempo_secuencial / tiempo_paralelo)

if __name__ == "__main__":
    main()