# Sistema_Rag_contestacion_automatica_correos
Este repositorio contiene el código de un sistema basado en Retrieval-Augmented Generation (RAG) diseñado para automatizar y agilizar la respuesta a correos electrónicos en las secretarías de la Universitat de València.
¿Qué problema resuelve?
Diariamente, los departamentos de atención al estudiante reciben un volumen masivo de correos. Alrededor del 70% son consultas repetitivas cuyas respuestas ya están detalladas explícitamente en normativas o guías oficiales. La lectura, búsqueda manual de la normativa y redacción de cada respuesta genera un cuello de botella administrativo significativo. Este proyecto automatiza ese flujo.  
¿Cómo funciona?
El pipeline del sistema gestiona de forma autónoma el ciclo de vida del correo:Identifica la intención del correo entrante y lo clasifica en uno de los dos dominios de conocimiento (Estudios de Postgrado o Prácticas Universitarias).  Recupera el contexto pertinente atacando una base de conocimiento vectorial generada a partir del corpus documental oficial (Reglamentos, FAQs y textos web en PDF/MD).  Redacta una respuesta precisa y formal basada exclusivamente en los documentos recuperados.  
Características Clave
Cero Alucinaciones: El sistema tiene un control estricto de la información. Si un correo plantea una duda que no está contemplada en la normativa, el modelo no inventa la respuesta; identifica la carencia y redacta un correo derivando al estudiante a la secretaría. 
Trazabilidad y Citación: Cada correo generado incluye una referencia explícita al documento exacto y al fragmento de texto del cual se ha extraído la respuesta, permitiendo una auditoría completa.  
Demo Interactiva: El proyecto incluye una prueba de concepto en tiempo real que permite introducir correos de prueba e inspeccionar tanto la respuesta generada como los fragmentos de normativa recuperados.
