import multiprocessing
import numpy as np
import os
from PIL import Image
import time
from IPython.display import display

from pycuda import autoinit, driver, compiler, gpuarray

void aplicar_filtro_gpu(cv::Mat& imagen, const char* filtro) {
  // Obtén el ancho y el alto de la imagen
  int width = imagen.cols;
  int height = imagen.rows;

  // Obtén el tamaño del kernel
  int kernel_size = kernels[filtro].size();

  // Asigna memoria para los píxeles de la imagen y el resultado
  float* pixels_gpu;
  float* resultado_gpu;
  cudaMalloc(&pixels_gpu, imagen.total() * sizeof(float));
  cudaMalloc(&resultado_gpu, imagen.total() * sizeof(float));

  // Copia los píxeles de la imagen a la GPU
  cudaMemcpy(pixels_gpu, imagen.data, imagen.total() * sizeof(float), cudaMemcpyHostToDevice);

  // Ejecuta el kernel
  kernel_a<<<dim3(width / 16, height / 16), dim3(16, 16)>>>(pixels_gpu, resultado_gpu, kernels[filtro].data(), width, height, kernel_size, kernel_size / 2);

  // Copia el resultado de vuelta a la CPU
  cudaMemcpy(imagen.data, resultado_gpu, imagen.total() * sizeof(float), cudaMemcpyDeviceToHost);

  // Libera la memoria de la GPU
  cudaFree(pixels_gpu);
  cudaFree(resultado_gpu);
}


  
