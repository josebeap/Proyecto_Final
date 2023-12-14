from mpi4py import MPI
import numpy as np
from PIL import Image
import os


# Inicializar MPI
comm = MPI.COMM_WORLD


rank = comm.Get_rank()
size = comm.Get_size()


# Definir kernels
kernels = {
    'kernela': np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]], np.int8),
    'kernelb': np.array([[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, -1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], np.int8)
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

# Función principal
def main():
    
    imagenes_y_filtros = [("imagenes/cb-300f.jpg", "kernelb")]

    resultados = []

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
            path_resultado = f'imagen_con_bordes_{idx}.jpg'
            guardar_imagen(imagen_procesada, path_resultado)
            
    

if __name__ == "__main__":
    main()
