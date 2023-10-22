# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 01:12:10 2023

@author: Bren Guzmán
"""

import numpy as np


class Principal_component:
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.covariance_matrix = None
        self.explained_variance_ratio_ = None  # Añade un atributo para la proporción de varianza explicada

    def fit(self, X):
        # 3. Calcular la matriz de covarianza
        self.covariance_matrix = np.cov(X, rowvar=False)  # Guarda la matriz de covarianza
        
        # 4. Calcular los valores y vectores propios
        eigenvalues, eigenvectors = np.linalg.eigh(self.covariance_matrix)
        
        # 5. Ordenar los vectores propios y valores propios
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        
        # Escoger los n vectores propios
        if self.n_components is not None:
            sorted_eigenvectors = sorted_eigenvectors[:, :self.n_components]
        
        self.eigenvalues = sorted_eigenvalues
        self.components = sorted_eigenvectors
        self.transformed_data = np.dot(X, self.components)
        
        # Calcular la proporción de varianza explicada
        total_variance = np.sum(self.eigenvalues)
        self.explained_variance_ratio_ = self.eigenvalues / total_variance

    def transform(self, X):
        # 6. Ocupar la matriz para transformar los datos originales
        transformed_data = np.dot(X, self.components)
        return transformed_data

