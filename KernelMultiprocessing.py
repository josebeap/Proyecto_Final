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

def aplicar_filtro_y_estadisticas(imagen, filtro, nombre):

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
                gy = np.sum(np.multiply(pixels[i-half_kernel_size:i+half_kernel_size+1, 
                                               j-half_kernel_size:j+half_kernel_size+1], kernels['kernela']))
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
    path_resultado =str(nombre)+'imagen_procesada.jpg'
   

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
    nombre = str(ruta_imagen)
    print(nombre)
    # Generar un nombre de archivo de salida
    #nombre_archivo_salida = f"{ruta_imagen.split('.')[0]}_{filtro}.jpg"
    
    return aplicar_filtro_y_estadisticas(imagen, filtro, nombre)

def main():
    # Obtener la ruta actual del script
    # Obtenemos la lista de nombres de archivos en la carpeta
    imagenes = os.listdir("./imagenes")
    nombres_imagenes = []
    # Recorremos la lista de archivos
    for imagen in imagenes:
        # Leemos la imagen
        nombres_imagenes.append(imagen)

    print("imagenes/"+str(nombres_imagenes))
    
    # Lista de rutas de imágenes y filtros a aplicar
    #lista_imagenes = []
    imagenes_y_filtros = []
    for i in nombres_imagenes:
       print("imagenes/"+str(i))
       imagenes_y_filtros.append(("imagenes/"+str(i), "filtroi"))
       #lista_imagenes.append(imagenes_y_filtros)
       
    print("La lista de tuplas:", imagenes_y_filtros)


    # Crear un pool de procesos
    pool = multiprocessing.Pool(processes=4)
    
    start_timeUno = time.time()
    # Procesar las imágenes en paralelo
    resultados = pool.map(procesar_imagen, imagenes_y_filtros)
    end_timeUno = time.time()
    tiempo_paralelo = end_timeUno - start_timeUno
    print("Tiempo Multiprocessing", end_timeUno - start_timeUno)
    
    start_time = time.time()
    #Secuencial
    for i in imagenes_y_filtros:
        imagenUno = cargar_imagen(i[0])
        filtro = i[1]
        nombre = str(i[0])
        aplicar_filtro_y_estadisticas(imagenUno, filtro, nombre)
    end_time = time.time()
    tiempo_secuencial = end_time - start_time
    
    print("Tiempo secuencial", end_time - start_time)
    
    print("Aceleración", tiempo_secuencial / tiempo_paralelo)
    
    

    # Mostrar resultados
    for resultado in resultados:
        print(resultado)S

if __name__ == "__main__":
    main()
