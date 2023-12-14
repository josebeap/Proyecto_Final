#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <opencv2/opencv.hpp>

// Define estructuras de kernel
typedef struct {
  int8_t data[3][3];
} Kernel;

const Kernel kernels[] = {
  Kernel kernela = {
  {0, 1, 0},
  {0, -1, 0},
  {0, 0, 0}
};
};

// Función para cargar una imagen
cv::Mat load_image(const char* path) {
  // Carga la imagen con OpenCV
  cv::Mat image = cv::imread(path);
  if (image.empty()) {
    fprintf(stderr, "Error al cargar la imagen: %s\n", path);
    return cv::Mat();
  }
  return image;
}

// Función para guardar una imagen
void save_image(const cv::Mat& image, const char* path) {
  // Guarda la imagen con OpenCV
  cv::imwrite(path, image);
}

// Función para aplicar un filtro a una imagen
void apply_filter(cv::Mat& image, const Kernel* kernel, const char* filter_name) {
  int kernel_size = kernel->data[0][0];
  int half_kernel_size = kernel_size / 2;

  // Crea una máscara de kernel
  cv::Mat kernel_mat = cv::Mat(kernel_size, kernel_size, CV_8UC1);
  for (int i = 0; i < kernel_size; ++i) {
    for (int j = 0; j < kernel_size; ++j) {
      kernel_mat.at<uchar>(i, j) = kernel->data[i + half_kernel_size][j + half_kernel_size];
    }
  }

  // Aplica el filtro a la imagen
  cv::Mat filtered_image;
  cv::filter2D(image, filtered_image, -1, kernel_mat, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);

  // Reemplaza la imagen original con la imagen filtrada
  image = filtered_image;
}

// Función principal
int main(int argc, char* argv[]) {
  if (argc < 3) {
    printf("Uso: %s <ruta_imagen> <nombre_filtro>\n", argv[0]);
    return 1;
  }

  const char* image_path = argv[1];
  const char* filter_name = argv[2];

  // Cargar imagen
  cv::Mat image = load_image(image_path);
  if (image.empty()) {
    return 1;
  }

  // Encuentra el kernel correspondiente
  const Kernel* kernel = NULL;
  for (int i = 0; i < sizeof(kernels) / sizeof(kernels[0]); ++i) {
    if (strcmp(filter_name, kernels[i].name) == 0) {
      kernel = &kernels[i];
      break;
    }
  }

  if (!kernel) {
    printf("Error: Filtro desconocido '%s'\n", filter_name);
    return 1;
  }

  // Aplicar el filtro
  apply_filter(image, kernel, filter_name);

  // Guardar la imagen procesada
  char output_path[128];
  sprintf(output_path, "%s_%s.jpg", image_path, filter_name);
  save_image(image, output_path);

  // Mostrar la imagen original y la imagen filtrada
  cv::imshow("Imagen original", image);
  cv::imshow("Imagen filtrada", filtered_image);
  cv::waitKey(0);

  return 0;
}
