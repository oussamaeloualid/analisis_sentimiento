# Análisis de Sentimiento para Reseñas de Amazon

Este proyecto utiliza técnicas de procesamiento de lenguaje natural (NLP) para analizar el sentimiento de las reseñas de clientes de Amazon. El proyecto está diseñado para identificar y clasificar los sentimientos como positivos, negativos o neutrales a partir de los textos de las reseñas.

## Tabla de Contenidos
- [Resumen del Proyecto](#resumen-del-proyecto)
- [Características](#características)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso](#uso)
- [Herramientas y Técnicas Utilizadas](#herramientas-y-técnicas-utilizadas)
- [Pruebas](#pruebas)

## Resumen del Proyecto

Este proyecto tiene como objetivo realizar un análisis de sentimiento sobre las reseñas de Amazon utilizando bibliotecas populares de Python como `pandas`, `nltk`, `transformers` y `seaborn`. Incluye funcionalidades para cargar datos, realizar análisis de sentimiento con `VADER` y `RoBERTa`, visualizar las distribuciones de sentimientos y filtrar las reseñas según su puntuación.

## Características

- Carga de datos desde un archivo CSV
- Tokenización y etiquetado de los textos
- Análisis de sentimiento con los modelos `VADER` y `RoBERTa`
- Visualización de los sentimientos en gráficos
- Filtrado de reseñas por puntuación de sentimiento

## Requisitos

- Python 3.x
- [pandas](https://pandas.pydata.org/)
- [nltk](https://www.nltk.org/)
- [transformers](https://huggingface.co/transformers/)
- [seaborn](https://seaborn.pydata.org/)

## Instalación

2. Instala las dependencias:
   ```bash
   pip install pandas nltk transformers seaborn
   ```

3. Descarga el conjunto de datos de reseñas de Amazon (`Reviews.csv`) y colócalo en el directorio del proyecto.

## Uso

- Carga los datos de las reseñas desde el archivo CSV.
- Ejecuta un análisis de sentimiento sobre las reseñas utilizando `VADER` y `RoBERTa`.
- Visualiza las distribuciones de sentimiento de las reseñas.
- Filtra las reseñas negativas o positivas según las puntuaciones de sentimiento.

## Herramientas y Técnicas Utilizadas

### Tokenización y Etiquetado con `nltk`
La **tokenización** consiste en dividir el texto en unidades más pequeñas llamadas tokens (palabras, símbolos, etc.), lo que permite entender y manipular mejor el texto. Una vez tokenizado, se utiliza la función `pos_tag()` para asignar categorías gramaticales (como sustantivo, verbo o adjetivo) a las palabras. Esto ayuda al modelo a comprender la estructura y el significado del texto.

Además, el método `chunk.ne_chunk()` permite agrupar los tokens etiquetados en "entidades nombradas" (por ejemplo, personas, lugares o organizaciones), lo que mejora la comprensión del texto al capturar información contextual.

### Análisis de Sentimiento con VADER
VADER (Valence Aware Dictionary and sEntiment Reasoner) es un método de análisis de sentimiento que asigna un valor (positivo, negativo o neutral) a cada palabra de una frase. Estos valores se combinan luego mediante una fórmula matemática para determinar la intensidad general del sentimiento de una frase.

Aunque VADER funciona bien para textos sencillos, no tiene en cuenta las relaciones entre las palabras en una frase. Esto puede ser un problema cuando se utilizan expresiones como sarcasmo o ironía.

**Explicación de VADER**:
VADER toma todas las palabras de una oración y asigna un valor a cada una que puede ser positivo, negativo o neutral. Luego combina estos valores para calcular la intensidad general del sentimiento de la oración. Esto se hace mediante una fórmula matemática que da como resultado un puntaje de sentimiento.

### Uso de `tqdm` para Seguimiento de Progreso
La biblioteca `tqdm` permite mostrar una barra de progreso mientras se ejecutan bucles sobre grandes cantidades de datos. Esto es útil para seguir el progreso de los análisis y mejorar la experiencia del usuario, especialmente cuando se procesan muchas reseñas.

### Análisis de Sentimiento con RoBERTa
RoBERTa (A Robustly Optimized BERT Pretraining Approach) es un modelo preentrenado que analiza cada palabra de una frase y le asigna una puntuación individual. A diferencia de VADER, RoBERTa tiene en cuenta las sutilezas del lenguaje humano, como el sarcasmo y el humor, lo que lo hace más adecuado para analizar textos complejos. Este modelo es especialmente útil cuando es necesario comprender el contexto más amplio de las palabras en una frase.

**Explicación de RoBERTa**:
RoBERTa evalúa cada palabra dentro del contexto completo de la oración, lo que le permite capturar los sentimientos en textos más complejos y con estructuras no tan directas. Este modelo ha demostrado ser más preciso que VADER en tareas de análisis de sentimientos más avanzadas.

## Pruebas

El proyecto incluye pruebas unitarias para verificar el correcto funcionamiento de las funcionalidades principales. Puedes ejecutar estas pruebas para asegurarte de que todo funciona correctamente.
