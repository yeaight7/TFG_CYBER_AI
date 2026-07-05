# Preguntas para verificar comprensión real

## A. Comprensión general del TFM

1. Explica en dos minutos cuál es el problema de investigación del TFM sin usar las palabras del título.
2. ¿Por qué el trabajo no plantea simplemente un clasificador supervisado tradicional?
3. ¿Qué significa que las decisiones sean binarias en este TFM?
4. ¿Qué diferencia hay entre PERMIT y BLOCK como acciones experimentales y un bloqueo real en una red?
5. ¿Cuál es la contribución principal del trabajo: el algoritmo, el dataset, el protocolo de evaluación o la formulación metodológica?

## B. Conceptos de ciberseguridad y datos

6. ¿Qué es un flujo de red y qué información resume?
7. ¿Por qué se usan flujos en lugar de inspección profunda de paquetes?
8. ¿Qué problemas conocidos tiene CICIDS2017?
9. ¿Por qué un resultado alto en CICIDS2017 no demuestra que el modelo funcione en una empresa real?
10. ¿Qué variables podrían producir fuga de información y por qué?

## C. Preprocesamiento y esquema canónico

11. ¿Qué significa que el TFM use un esquema canónico de 76 variables?
12. ¿Por qué se añade una máscara de ausencia?
13. Si una variable no existe en el tráfico de laboratorio, ¿qué ocurre con ella en el vector de entrada?
14. ¿Por qué el escalado debe ajustarse solo con datos de entrenamiento?
15. ¿Qué pasaría si se normalizan los datos antes de separar entrenamiento y prueba?

## D. Aprendizaje por refuerzo

16. ¿Qué convierte este problema en un problema de aprendizaje por refuerzo?
17. ¿Cuál es el estado, cuál es la acción y cuál es la recompensa?
18. ¿Por qué el entorno se define como offline?
19. ¿Qué limitación tiene tratar un dataset estático como entorno?
20. ¿En qué sentido esta formulación no representa una defensa autónoma real?

## E. Función de recompensa y costes asimétricos

21. ¿Por qué un falso negativo es más grave que un falso positivo en este contexto?
22. ¿Cómo se refleja esa diferencia en la función de recompensa?
23. ¿Qué riesgo aparece si se penalizan demasiado los falsos negativos?
24. ¿Cómo interpretarías un modelo con recall muy alto pero muchos falsos positivos?
25. ¿Qué métrica mirarías primero si el objetivo es evitar ataques no detectados?

## F. QRDQN

26. ¿Qué es QRDQN, explicado de forma comprensible?
27. ¿En qué se diferencia QRDQN de un DQN estándar?
28. ¿Por qué podría tener sentido usar un método distribucional en este problema?
29. ¿Qué hiperparámetros del agente podrían influir mucho en los resultados?
30. ¿Qué evidencia necesitarías para afirmar que QRDQN aporta algo frente a Random Forest?

## G. Comparación con Random Forest

31. ¿Por qué Random Forest es un baseline razonable para este TFM?
32. ¿Qué significa una comparación bajo el mismo protocolo?
33. Si Random Forest obtiene resultados similares o mejores que QRDQN, ¿cómo debería interpretarse?
34. ¿Sería válido decir que QRDQN es mejor solo porque usa aprendizaje por refuerzo?
35. ¿Qué ventajas prácticas podría tener Random Forest frente a QRDQN?

## H. Evaluación y validación

36. ¿Qué problema tiene una partición aleatoria fila a fila en datasets de tráfico?
37. ¿Qué aporta una partición temporal o por día?
38. ¿Qué evalúa dejar fuera un CSV completo?
39. ¿Qué significa desplazamiento de dominio?
40. ¿Por qué la validación con tráfico de laboratorio no equivale a una validación en producción?

## I. Discusión crítica

41. ¿Cuál es la principal debilidad metodológica del TFM?
42. ¿Qué afirmación sería excesiva o no justificada a partir de este trabajo?
43. ¿Qué cambio harías si quisieras acercar el sistema a un escenario real?
44. ¿Qué información se pierde al convertir todas las etiquetas a benigno/ataque?
45. ¿Qué tipo de ataque o escenario podría no estar bien representado en este diseño?

## J. Preguntas de control para detectar uso superficial de IA

46. Señala una decisión metodológica del TFM con la que no estés completamente de acuerdo y justifica por qué.
47. ¿Qué parte del TFM te parece más débil y cómo la mejorarías?
48. Explica una limitación que no sea solo “faltan más datos”.
49. Si tuvieras que eliminar una sección por redundante, ¿cuál sería y por qué?
50. ¿Qué resultado experimental te haría cambiar la interpretación del trabajo?

## Preguntas especialmente útiles

Estas preguntas son las más diagnósticas porque obligan a razonar, no solo a repetir contenido.

1. “Explícame con un ejemplo concreto qué sería un falso positivo y un falso negativo en tu sistema.”
2. “¿Por qué dices que usas aprendizaje por refuerzo si realmente entrenas sobre un dataset etiquetado y estático?”
3. “¿Qué tendría que pasar para que Random Forest fuese una opción metodológicamente preferible a QRDQN?”
4. “¿Dónde podría colarse fuga de información aunque hayas quitado las columnas más obvias?”
5. “¿Qué no puedes afirmar con tus resultados?”
6. “Si un tribunal te dice que esto es clasificación disfrazada de RL, ¿cómo responderías?”
7. “¿Por qué la máscara de ausencia no es un detalle técnico menor?”
8. “¿Qué diferencia hay entre generalizar a otro CSV de CICIDS2017 y generalizar a tráfico real?”
9. “¿Qué cambiaría si en vez de dos acciones tuvieras varias acciones defensivas posibles?”
10. “Dime una decisión que hayas tomado por seguridad metodológica, aunque redujera el rendimiento aparente.”
