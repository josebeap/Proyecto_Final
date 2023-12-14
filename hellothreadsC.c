#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {

  // Definir el kernel
  int kernelb[] = {0, 0, 0, 0, 0,
                  0, 0, 1, 0, 0,
                  0, 0, -1, 0, 0,
                  0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0};

  // Cargar la imagen
  char *ruta = "image.jpg";
  Image *imagen = cargar_imagen(ruta);

  // Convertir la imagen a un array de numpy
  int *pixels = (int *)imagen->data;

  // Preparar el array de salida
  int *resultado = (int *)malloc(imagen->size * sizeof(int));

  // Aplicar el filtro
  #pragma omp parallel for
  for (int i = half_kernel_size; i < imagen->height - half_kernel_size; i++) {
    for (int j = half_kernel_size; j < imagen->width - half_kernel_size; j++) {
      // Aplica el kernel
      int gy = 0;
      for (int k = -half_kernel_size; k <= half_kernel_size; k++) {
        for (int l = -half_kernel_size; l <= half_kernel_size; l++) {
          gy += pixels[(i + k) * imagen->width + j + l] * kernelb[k + half_kernel_size] * kernelb[l + half_kernel_size];
        }
      }
      // Guarda el resultado
      resultado[i * imagen->width + j] = min(255, abs(gy));
    }
  }

  // Guardar la imagen
  guardar_imagen(imagen, "imagen_filtrada.jpg");

  // Liberar la memoria
  free(pixels);
  free(resultado);

  return 0;
}
