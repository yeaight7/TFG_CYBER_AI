# Gap y posicionamiento defendible para un TFG sobre defensa cibernética con RL

## Qué está ya bien estudiado

La detección de intrusiones basada en ML/DL sobre representaciones de flujo y benchmarks públicos ya es un campo muy poblado. Las revisiones sistemáticas recientes sobre NIDS anómalos y métodos deep learning muestran una literatura extensa en torno a clasificación binaria o multiclase con datasets públicos; además, CICIDS2017 fue creado precisamente como benchmark flow-based para evaluar algoritmos de aprendizaje sobre tráfico benigno y varias familias de ataque. Un survey reciente centrado en datasets de NIDS identifica 89 datasets públicos y observa que CIC-IDS2017 sigue entre los benchmarks más usados por la comunidad.

Tampoco sería defendible presentar como novedad el uso de RL en IDS/NIDS. Ya existen trabajos que reformulan la detección como aprendizaje por refuerzo sobre datos etiquetados, incluso sustituyendo el entorno vivo por un pseudoentorno que muestrea intrusiones registradas; otros usan Deep Q-Learning con datasets clásicos, arquitecturas context-aware y multiagente, marcos transferibles/adaptables y propuestas recientes orientadas a zero-day detection. Un survey específico sobre DRL para NIDS revisa precisamente estas líneas y concluye que el área es prometedora, pero claramente existente y reconocible como subcampo.

También está asentada la idea de que la comparabilidad entre trabajos depende mucho de la representación de las features. Una de las críticas consolidadas a los benchmarks de NIDS es la ausencia de un conjunto estándar de características entre datasets, lo que dificulta comparar métodos y estudiar generalización; de ahí que existan propuestas de feature sets comunes basados en NetFlow o equivalentes. Esto vuelve perfectamente razonable que tu TFG use preprocesado canónico de flow-features, pero también significa que ese componente debe venderse como decisión metodológica y no como novedad en sí misma.

## Qué está parcialmente estudiado pero sigue siendo limitado

Lo que sí aparece solo parcialmente cubierto es el espacio intermedio entre **“RL como clasificador sobre datasets etiquetados”** y **“RL como mitigador autónomo en simuladores/SDN”**. En varios trabajos de RL sobre NIDS, la acción termina correspondiendo a la clase o etiqueta del tráfico; por ejemplo, algunos artículos adaptan el problema haciendo que las features sean el estado y las etiquetas las acciones, o reemplazan el entorno por un pseudoentorno que muestrea datos ya registrados. En paralelo, la literatura de mitigación en SDN formula acciones como contramedidas reales y advierte que bloquear tráfico meramente sospechoso puede dañar la disponibilidad y funcionalidad de la red. A partir de ese patrón, es razonable **inferir** que una formulación *offline*, *flow-level* y explícitamente *defender-centric* de decisión binaria **PERMIT/BLOCK** está menos consolidada que cualquiera de esos dos polos por separado.

La asimetría de costes tampoco es nueva, pero sí está tratada de forma desigual. La literatura clásica de IDS lleva años señalando que el coste de un falso negativo puede equivaler al daño del ataque consumado, mientras que el falso positivo impone costes operativos o de disponibilidad; existen además propuestas cost-sensitive e imbalance-aware en NIDS modernos. Sin embargo, en RL la función de recompensa sigue apareciendo como uno de los puntos más delicados y a menudo simplificados, y el survey reciente de DRL para NIDS sigue destacando como retos la eficiencia de entrenamiento y la detección de clases minoritarias o desconocidas. Por eso, un *reward* explícitamente sesgado contra falsos negativos es defendible; lo que no sería defendible es venderlo como una idea inédita o como un problema ya resuelto por la literatura.

Algo parecido ocurre con la eficiencia de datos. La necesidad de adaptarse con pocas muestras o a nuevas clases ya ha dado lugar a trabajos *few-shot/class-incremental* y a propuestas RL orientadas a adaptación y transferibilidad con menos datos. Pero, en la literatura revisada, esto suele aparecer como una técnica especializada para adaptación o zero-day, no tanto como una comparación experimental muy controlada de **curvas de aprendizaje RL vs. supervisado** bajo la misma representación de features, los mismos *splits* y el mismo test interno. Esa comparación, planteada como contribución metodológica y experimental —no algorítmica—, sí parece suficientemente defendible para un TFG.

## Qué está débilmente validado en la literatura

Una parte importante de la literatura sigue débilmente validada en la propia calidad del benchmark. Engelen et al. revisaron CICIDS2017 y encontraron problemas en simulación de ataques, construcción de flujos, extracción de características y etiquetado; reportan además que más del 20% de las trazas originales tuvieron que reconstruirse o reetiquetarse y que más del 25% de los flujos eran artefactos sin significado útil para el aprendizaje. En paralelo, Lanvin et al. sostienen que la versión resumida de CIC-IDS2017 difícilmente tiene importación práctica por sí sola. Esto no invalida que uses CICIDS2017, pero sí obliga a una narrativa metodológica muy cauta y explícita sobre sus límites.

La segunda debilidad es el leakage evaluativo. En seguridad, Arp et al. documentan *pitfalls* como *data snooping*, ignorar dependencias temporales y explotar correlaciones espurias, todos ellos capaces de inflar resultados de forma artificial; además recomiendan aislar pronto train/validation/test y complementar experimentos sobre datasets muy conocidos con datos más recientes del dominio. Esa advertencia encaja de lleno con NIDS, donde Sommer y Paxson ya habían subrayado el alto coste de los errores, la falta de datos adecuados y la dificultad de realizar evaluaciones sólidas y operativamente relevantes.

La tercera debilidad es la generalización fuera del mismo dataset. Apruzzese et al. argumentan que la *cross-evaluation* de ML-NIDS recibió atención limitada y la proponen precisamente para descubrir riesgos y cualidades ocultas que el experimento intra-dataset no muestra. Cantone et al. son todavía más contundentes: varios clasificadores rozan resultados casi perfectos cuando entrenan y prueban dentro del mismo dataset, pero caen hacia el azar cuando el entrenamiento y el test se separan por dataset. Además, el gran survey de datasets observa que muchos grupos siguen capturando sus propios datos sin compartirlos, algo que dificulta la reproducibilidad y la validación externa comparable. En ese contexto, una validación con tráfico privado de laboratorio debe presentarse como **stress test exploratorio de distribution shift**, no como prueba concluyente de despliegue real.

También la robustez adversarial y la justificación operativa siguen siendo frágiles. La literatura reciente sobre DRL-based intrusion detection muestra que estos sistemas también son vulnerables a ejemplos adversarios y que su robustez depende de elecciones de arquitectura e hiperparámetros. Y, desde la literatura de mitigación, se insiste en que “actuar” sobre tráfico sospechoso no es trivial: bloquear sin suficiente justificación puede causar daños operativos. Todo esto refuerza que tu TFG debe evitar cualquier lenguaje de producción, bloqueo en línea o autonomía operacional.

## Qué hueco puede reclamar razonablemente tu TFG

La formulación más fuerte y a la vez más segura para tu trabajo es un **hueco metodológico/experimental**, no un hueco de “primera vez” ni de “solución definitiva”. Tu valor no está en inventar RL para IDS, sino en medir con cuidado qué aporta una formulación RL defensiva simple cuando se controla seriamente el pipeline, se compara contra supervisado y se documentan los riesgos. Esa es una forma de posicionamiento muy alineada con lo que la literatura crítica sobre NIDS pide desde hace años.

**Versión conservadora del gap.**  
Este TFG cubre un hueco de evaluación reproducible: aplicar un agente **QRDQN** a decisiones binarias **PERMIT/BLOCK** sobre flujos ya observados, con preprocesado canónico de features, *split* fijo y control explícito de leakage, comparándolo de forma directa contra al menos un baseline supervisado tabular sobre el mismo pipeline y midiendo cómo cambian los resultados al reducir el tamaño de entrenamiento. La contribución no estaría en la “novedad del RL”, sino en la limpieza experimental, la comparabilidad y la explicitación de riesgos de leakage y *distribution shift*.

**Versión equilibrada del gap.**  
Este TFG se sitúa entre dos tradiciones ya existentes: la de RL usado como clasificador sobre datasets etiquetados y la de RL usado para mitigación en entornos SDN o simulados. Su hueco defendible es proponer una formulación **offline, flow-level y defender-centric** de decisión binaria **PERMIT/BLOCK**, con *reward* coste-sensible que penaliza más los falsos negativos, y contrastarla frente a baselines supervisados bajo el mismo pipeline de datos, las mismas *budgets* de entrenamiento y el mismo test interno. La pregunta no es si RL “funciona” en abstracto, sino **cuándo** merece la pena frente a Random Forest u otros modelos tabulares en este framing concreto.

**Versión ambiciosa pero aún defendible del gap.**  
Este TFG puede reclamar una pequeña contribución de protocolo para la evaluación de RL defensivo sobre NIDS de flujos: representación canónica, entorno tipo dataset-as-environment, política binaria QRDQN, *reward* sensible a FN, curvas de eficiencia de datos, benchmark interno reproducible y validación externa separada con tráfico de laboratorio. Presentado así, el trabajo intenta tender un puente entre la literatura de clasificación y la de *cyber defense*, no para afirmar despliegue autónomo, sino para mostrar de forma honesta qué gana y qué pierde un framing de decisión defensiva con RL cuando se lo somete a comparación, escasez de datos y cambio de dominio.

Si tuviera que recomendarte una sola formulación para la memoria, escogería la **versión equilibrada**: es intelectualmente interesante, metodológicamente sólida y mucho menos vulnerable a objeciones de sobreventa.

## Qué no deberías afirmar en ningún caso

No deberías afirmar que **“RL para IDS/NIDS no se ha estudiado”** o que tu trabajo es la **primera** aplicación de RL a detección de intrusiones. Esa afirmación chocaría directamente con trabajos previos sobre pseudoentornos supervisados, DQL/DQN, sistemas context-aware y marcos adaptables/transferibles.

No deberías afirmar que tu framing **PERMIT/BLOCK** equivale a **bloqueo real en línea** o a una arquitectura lista para respuesta automática. La literatura de mitigación ya advierte que actuar sobre tráfico sospechoso puede afectar disponibilidad y funcionalidad, y la literatura clásica de NIDS insiste en la alta gravedad de los errores y en la brecha entre alerta y acción operativa.

No deberías afirmar que un buen resultado en **CICIDS2017** demuestra generalización al mundo real. La propia literatura sobre el dataset documenta errores y artefactos; además, los estudios *cross-dataset* muestran caídas severas de rendimiento fuera del mismo benchmark.

No deberías afirmar que tu tratamiento coste-sensible **resuelve** el problema FN/FP de forma general. Lo correcto es decir que **operacionaliza** esa asimetría para tu setting experimental, en línea con una preocupación ya reconocida en IDS y NIDS.

No deberías afirmar que **QRDQN** es la novedad central del trabajo. Con la literatura actual, es más prudente presentarlo como una elección razonable dentro de la familia de algoritmos *value-based*; la originalidad del TFG está mejor defendida en el diseño experimental, la comparación con supervisado, la eficiencia de datos y la separación entre benchmark interno y validación externa.

## Párrafos sugeridos para la tesis

**Versión corta**

Este trabajo no parte de la premisa de que el aprendizaje por refuerzo sea novedoso en detección de intrusiones, sino de que sigue faltando evidencia comparativa y reproducible sobre qué aporta una formulación RL simple y orientada a la decisión defensiva cuando se aplica a decisiones binarias **PERMIT/BLOCK** sobre flujos de red. Para ello, el TFG adopta un benchmark público basado principalmente en **CICIDS2017**, utiliza un pipeline canónico de preprocesado, compara **QRDQN** frente a baselines supervisados y evalúa explícitamente eficiencia de datos, riesgo de *leakage* y sensibilidad al cambio de dominio. Los resultados se interpretan como evidencia experimental acotada, no como validación de despliegue real.

**Versión media**

La literatura en NIDS ha estudiado de forma extensa tanto los enfoques supervisados y deep learning sobre benchmarks públicos como diversas aplicaciones de aprendizaje por refuerzo a la detección de intrusiones. Sin embargo, persisten limitaciones metodológicas relevantes: dependencia de un único dataset, heterogeneidad en el preprocesado de características, escasa atención a la evaluación *cross-dataset* y riesgo de *leakage* o correlaciones espurias. En este contexto, el presente TFG se posiciona no como una propuesta de novedad algorítmica absoluta, sino como una evaluación experimental reproducible de un agente **QRDQN** para decisiones binarias **PERMIT/BLOCK** a nivel de flujo, comparado bajo el mismo pipeline frente a modelos supervisados tabulares. El trabajo presta especial atención a la asimetría de costes entre falsos negativos y falsos positivos, a la eficiencia de datos mediante distintos tamaños de entrenamiento y, cuando sea viable, a una validación externa separada con tráfico capturado en laboratorio. Esta última se plantea como comprobación exploratoria de robustez frente a cambio de dominio, y no como prueba de despliegue en producción.

**Versión formal académica**

Desde una perspectiva de posicionamiento científico, este TFG se inserta en la intersección entre la investigación sobre NIDS basados en aprendizaje supervisado y la línea, ya existente, de aprendizaje por refuerzo aplicado a ciberseguridad. La aportación del trabajo no debe entenderse como la introducción ex nihilo de RL en intrusión de red, puesto que la literatura ya documenta formulaciones basadas en DQN/DQL, pseudoentornos construidos a partir de datasets etiquetados, arquitecturas multiagente y propuestas orientadas a adaptabilidad o detección de ataques no vistos. El interés del trabajo reside, más bien, en ofrecer una formulación experimental acotada, reproducible y metodológicamente prudente de decisión defensiva binaria **PERMIT/BLOCK** sobre flujos de red, apoyada en un benchmark público y en un pipeline canónico de extracción y normalización de características. Sobre esta base, se compara un agente **QRDQN** frente a baselines supervisados bajo condiciones homogéneas de entrenamiento y evaluación, incorporando además un análisis explícito de eficiencia de datos y una consideración coste-sensible en la que los falsos negativos reciben una penalización mayor que los falsos positivos. Finalmente, el trabajo separa de manera deliberada la validación interna sobre benchmark público de cualquier validación externa con tráfico de laboratorio, con el fin de no confundir desempeño intra-dataset con capacidad de generalización operativa. En consecuencia, la contribución del TFG se sitúa principalmente en el plano metodológico y experimental, y no en afirmaciones de despliegue real, producción o autonomía defensiva en tiempo real.

## Fuentes que sostienen el hueco

**Mapeo claim → claves**

**Uso extendido de benchmarks públicos y de CICIDS2017:** [SHARAFALDIN2018], [YANG2022], [NID-DATA-2025], [SARHAN2022].  
**RL para IDS/NIDS ya está estudiado:** [LOPEZMARTIN2020], [SETHI2020], [ALAVIZADEH2022], [HE2024], [ALAM2025], [YANG-DRL-2026].  
**Asimetría FN/FP y sensibilidad a costes:** [AXELSSON1999], [LEE2000], [GUPTA2022], [ATMOS2020].  
**Riesgos de leakage y artefactos del benchmark:** [ENGELEN2021], [LANVIN2023], [ARP2024], [SOMMERPAXSON2010].  
**Generalización débil y necesidad de validación externa separada:** [APRUZZESE2022], [CANTONE2024], [NID-DATA-2025].  
**Escasez de datos y adaptación con pocas muestras:** [DIMONDA2024], [HE2024], [YANG-DRL-2026].

**Leyenda de claves**

[SHARAFALDIN2018] Sharafaldin et al., *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization*; [YANG2022] Yang et al., *A systematic literature review of methods and datasets for anomaly-based network intrusion detection*; [NID-DATA-2025] *Network Intrusion Datasets: A Survey, Limitations, and Recommendations*; [SARHAN2022] Sarhan et al., *Towards a Standard Feature Set for Network Intrusion Detection System Datasets*.

[LOPEZMARTIN2020] López-Martín et al., *Application of deep reinforcement learning to intrusion detection for supervised problems*; [SETHI2020] Sethi et al., *A context-aware robust intrusion detection system*; [ALAVIZADEH2022] Alavizadeh et al., *Deep Q-Learning Based Reinforcement Learning Approach for Network Intrusion Detection*; [HE2024] He et al., *Reinforcement Learning Meets Network Intrusion Detection*; [ALAM2025] Alam et al., *Adaptive Defense: Zero-Day Attack Detection in NIDS with Deep Reinforcement Learning*; [YANG-DRL-2026] *A Survey for Deep Reinforcement Learning Based Network Intrusion Detection*.

[ENGELEN2021] Engelen et al., *Troubleshooting an Intrusion Detection Dataset*; [LANVIN2023] Lanvin et al., *Faulty use of the CIC-IDS 2017 dataset in information security research*; [ARP2024] Arp et al., *Pitfalls in Machine Learning for Computer Security*; [SOMMERPAXSON2010] Sommer y Paxson, *Outside the Closed World*; [APRUZZESE2022] Apruzzese et al., *The Cross-evaluation of Machine Learning-based Network Intrusion Detection Systems*; [CANTONE2024] Cantone et al., *On the Cross-Dataset Generalization of Machine Learning for Network Intrusion Detection*.

[AXELSSON1999] Axelsson, *The base-rate fallacy and its implications for the difficulty of intrusion detection*; [LEE2000] Lee et al., *Toward cost-sensitive modeling for intrusion detection and response*; [GUPTA2022] Gupta et al., *CSE-IDS*; [ATMOS2020] Akbari et al., *ATMoS: Autonomous Threat Mitigation in SDN using Reinforcement Learning*; [DIMONDA2024] Di Monda et al., *Few-Shot Class-Incremental Learning for Network Intrusion Detection Systems*.

## Handoff para Codex

El siguiente *prompt* traduce el posicionamiento anterior a una subsección de memoria centrada en rigor metodológico, comparabilidad y cautela frente a *leakage* y *distribution shift*.

```text
Escribe la subsección "Justificación y posicionamiento del trabajo" en lenguaje académico, con una extensión aproximada de 400 a 650 palabras.

Contexto del TFG:
- Tema: "RL-based cybersecurity defender for binary PERMIT/BLOCK decisions on network flows".
- Dataset benchmark principal: CICIDS2017.
- Preprocesado: flow-features canónicas.
- Entorno: formulación dataset-as-environment tipo Gymnasium.
- Agente: QRDQN.
- Acción: espacio binario PERMIT/BLOCK.
- Restricción clave: no hay bloqueo real en tiempo real; PERMIT/BLOCK es una abstracción experimental de decisión defensiva offline sobre flujos.
- Baseline supervisado: al menos Random Forest.
- Experimentos de eficiencia de datos: 100k / 250k / 500k / 1M / 2M, con el mismo test interno.
- Validación externa: tráfico privado de laboratorio, solo si es viable, y siempre reportado por separado del benchmark público.

Objetivo del texto:
- Posicionar el TFG como contribución metodológica y experimental.
- NO presentarlo como novedad absoluta.
- NO afirmar que RL para IDS no exista.
- NO afirmar despliegue real, producción, ni readiness operacional.
- NO confundir evaluación interna en benchmark con generalización al mundo real.

Ideas que DEBEN aparecer:
1. La literatura ya ha estudiado ampliamente ML/DL para NIDS y también RL para IDS/NIDS.
2. Aun así, persisten problemas de comparabilidad, leakage, artefactos del benchmark y mala generalización cross-dataset.
3. El hueco del TFG está en evaluar de forma reproducible una formulación RL simple, binaria y orientada a decisión defensiva sobre flujos, comparándola limpiamente con supervisado.
4. El trabajo enfatiza:
   - diseño experimental reproducible;
   - comparación RL vs. baseline supervisado bajo el mismo pipeline;
   - eficiencia de datos;
   - separación explícita entre benchmark interno y validación externa;
   - discusión honesta de riesgos metodológicos (leakage, distribución, artefactos del dataset).
5. El tratamiento coste-sensible debe formularse con prudencia:
   - se prioriza reducir falsos negativos;
   - pero no se afirma que el problema quede resuelto universalmente.

Claves bibliográficas a integrar en el texto:
- [LOPEZMARTIN2020]
- [ALAVIZADEH2022]
- [HE2024]
- [SHARAFALDIN2018]
- [ENGELEN2021]
- [LANVIN2023]
- [ARP2024]
- [SOMMERPAXSON2010]
- [APRUZZESE2022]
- [CANTONE2024]
- [SARHAN2022]
- [LEE2000]
- [DIMONDA2024]

Tono:
- Sobrio, preciso, sin marketing.
- Usar expresiones como "se posiciona", "se enmarca", "busca aportar evidencia experimental", "no pretende demostrar despliegue real".
- Evitar palabras como:
  "novel", "state-of-the-art", "production-ready", "real-time blocking", "fully autonomous defense", "first ever".

Estructura sugerida:
- Párrafo 1: estado del arte y por qué no hay novedad absoluta.
- Párrafo 2: limitaciones metodológicas de la literatura.
- Párrafo 3: hueco concreto y posicionamiento del TFG.
- Cierre: frase que subraye que la aportación es experimental/metodológica y que la validación externa, si existe, se interpreta como exploración de robustez ante cambio de dominio.

Si ves oportuno, usa una frase del estilo:
"En consecuencia, la contribución del trabajo se sitúa principalmente en el plano metodológico y experimental, al estudiar bajo un protocolo reproducible hasta qué punto una formulación RL de decisión binaria sobre flujos puede ofrecer ventajas o limitaciones frente a baselines supervisados en un escenario de NIDS basado en datasets públicos."
```
