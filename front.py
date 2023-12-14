import streamlit as st
import os
from PIL import Image
from multiprocessing import Process, Queue
#from descargar_imagenes import iniciar_descarga  # Asegúrate de tener esta función definida en "descargar_imagenes.py"
import numpy as np
from os import listdir
from os.path import isfile, join
from scipy.signal import convolve2d
import re
def Main():
    # Definimos el título de la vista
    st.title("Vista con Streamlit")

    # Creamos un combo box con las opciones de kernel
    kernel_options = ["El primero de los Class 1","El primero de los Class 2",
                        "El primero de los Class 3","Square 3x3",
                        "El primero de los Edge 3x3",
                        "Square 5x5","El primero de los Edge 5x5",
                        "Sobel vertical y horizontalmente",
                        "Laplace","Prewitt vertical y horizontal"]
    kernel_choice = st.selectbox("Seleccionar kernel", kernel_options)

    #Creamos un combo box con las opciones de framework
    framework_options = ["C","OpenMP","Multiprocessing","MPI4PY","PyCUDA"]
    framework_choice = st.selectbox("Seleccionar Framework", framework_options)

    # Creamos un cuadro de texto para ingresar palabras
    palabras = st.text_input("Ingresar palabras")

    # Creamos un botón para aplicar los cambios
    if st.button("Aplicar"):
        # Imprimimos el kernel seleccionado
        st.write("Kernel seleccionado:", kernel_choice)
        # Imprimimos el kernel seleccionado
        st.write("Framework seleccionado:", framework_choice)
        
        # Imprimimos las palabras ingresadas
        st.write("Palabras ingresadas:", palabras)



        st.title('Aplicación de Filtros de Imagen con Streamlit')


if __name__ == '__main__':
    Main()
