# PassGAN

**Generative Adversarial Network para la generación y análisis de contraseñas**

## Descripción general

PassGAN es un modelo basado en Generative Adversarial Networks (GANs) que aprende patrones contenidos en conjuntos de contraseñas filtradas públicamente con el fin de estudiar la capacidad de los modelos generativos para producir contraseñas plausibles. Este repositorio implementa una versión reproducible de PassGAN y desarrolla un análisis crítico sobre su eficacia, limitaciones y riesgos en el contexto de la ciberseguridad defensiva.

El propósito de este proyecto es estrictamente académico. Se centra en comprender cómo funcionan este tipo de modelos, evaluar su rendimiento y examinar los posibles impactos en la seguridad de los sistemas, especialmente en relación con la creación de políticas de contraseñas más robustas y la concienciación sobre hábitos de seguridad.

## Objetivos del proyecto

- Implementar o reproducir el modelo PassGAN en un entorno controlado
- Analizar la arquitectura generativa y discriminativa del modelo
- Evaluar la calidad y diversidad de las contraseñas generadas
- Comparar los resultados con otros enfoques de generación o cracking basado en estadísticas
- Estudiar los riesgos asociados a la disponibilidad de este tipo de técnicas
- Proponer recomendaciones defensivas basadas en los hallazgos

## ⚠️ Advertencia ética

**Este proyecto está orientado exclusivamente a investigación, docencia y análisis de riesgos en ciberseguridad.**

No está diseñado para obtener accesos no autorizados, vulnerar sistemas ni facilitar actividad maliciosa. Cada experimento debe realizarse únicamente en entornos propios, controlados y con autorización explícita.

## Estructura del repositorio
├── src/ # Código del modelo, entrenamiento y utilidades

├── data/ # Conjuntos de datos permitidos para investigación

├── experiments/ # Scripts de evaluación y generación

├── reports/ # Documentación técnica y resultados

├── requirements.txt

└── README.md

## Cómo ejecutar el proyecto

### 1. Clonar el repositorio

```bash
git clone https://github.com/usuario/passgan.git
cd passgan
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Preparar el conjunto de datos
Seguir las indicaciones del directorio data/.

### 4. Entrenar el modelo
```bash
python src/train.py
```

### 5. Generar contraseñas sintéticas para análisis
```bash
python src/generate.py
```
