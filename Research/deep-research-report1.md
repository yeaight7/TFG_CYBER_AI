# Dossier de investigación para el Estado del Arte

## Estructura propuesta del capítulo

La estructura más sólida para el capítulo de Estado del Arte, dado tu TFG y el tipo de afirmaciones que quieres poder defender ante tribunal, no debería organizarse “por moda algorítmica”, sino por una progresión desde el problema de ingeniería y seguridad hasta el posicionamiento exacto del prototipo. Eso reduce el riesgo de sobredimensionar la contribución del RL y ayuda a distinguir entre hechos asentados, prácticas habituales y afirmaciones todavía débiles en la literatura. Los NIDS forman parte de la familia más amplia de los IDPS; los sistemas basados en flujos son habituales porque la telemetría de red basada en flow records está estandarizada y es operativamente manejable, y porque varios benchmarks públicos de referencia ya se publican o derivan a nivel de flujo. citeturn28view1turn27view0turn19view0turn20view0turn25academia2turn24academia0

También conviene que la estructura haga visible, desde el principio, un hecho incómodo pero central: la literatura de NIDS basada en ML reporta con frecuencia resultados casi perfectos dentro del mismo dataset, pero esos resultados suelen degradarse bajo cambios de dominio, validación cruzada entre datasets o tráfico más realista. Esa tensión entre “benchmark interno” y “transportabilidad” es precisamente donde tu diseño experimental tiene más valor argumentativo. citeturn29academia1turn29academia0turn18academia2turn30academia0

Una estructura de capítulo defendible en español sería la siguiente:

| Subapartado propuesto | Función en el capítulo | Qué debe demostrar | Citas base sugeridas |
|---|---|---|---|
| **Contexto de la detección de intrusiones en red** | Introducir IDS/IDPS, NIDS, detección por firmas y detección por anomalías | Que el problema no nace con ML y que ML no sustituye todo el stack defensivo | [ScarfoneMell2007], [Ahmad2021] |
| **NIDS basados en tráfico de red y enfoques flow-based** | Explicar por qué se usan flujos y qué se gana/pierde frente a paquetes o payloads | Escalabilidad, estandarización, compatibilidad con tráfico cifrado, menor granularidad semántica | [Claise2013], [Sharafaldin2018], [UNB_IDS2018_Page], [Sarhan2020], [ElMahdaouy2026_weak] |
| **Datasets públicos para investigación en NIDS** | Presentar CICIDS2017, CSE-CIC-IDS2018, UNSW-NB15 y NSL-KDD | Que son benchmarks útiles pero no equivalen a despliegue real | [Sharafaldin2018], [UNB_IDS2018_Page], [MoustafaSlay2015], [Tavallaee2009], [Ring2019], [Goldschmidt2025_weak] |
| **Aprendizaje supervisado y deep learning para NIDS** | Situar el grueso de la literatura “mainstream” | Que la base comparativa razonable no es solo RL, sino también clasificadores supervisados fuertes | [Ahmad2021], [Maseer2023], [Sarhan2021], [Corea2024] |
| **Fundamentos de RL y DRL relevantes para el TFG** | Introducir formalismo MDP, acción, recompensa, valor, política, DQN, PPO, A2C, RL distribucional | Que QRDQN no aparece “de la nada” y deriva de una línea técnica conocida | [SuttonBarto2018], [Mnih2015], [vanHasselt2016], [Wang2016], [Mnih2016], [Schulman2017], [Bellemare2017], [Dabney2018] |
| **RL y DRL aplicados a NIDS y defensa ciber** | Revisar trabajos que usan RL para clasificación, detección, respuesta y defensa adaptativa | Que sí hay precedentes, pero heterogéneos y metodológicamente irregulares | [Alavizadeh2021], [Strickland2023], [Tellache2024], [Yang2024_weak], [Gueriani2024_weak], [Palmer2023_weak] |
| **Limitaciones metodológicas del área** | Criticar sesgos, leakage, splits aleatorios, ausencia de validación externa y reproducibilidad débil | Que el capítulo no venda exactitudes ilusorias | [Arp2020], [Layeghy2021], [Layeghy2022], [Cantone2024], [Ring2019] |
| **Posicionamiento y alcance del TFG** | Cerrar el capítulo conectando con tu implementación concreta | Que tu TFG aporta una formulación y una evaluación prudente, no una “solución definitiva” | repositorio del TFG + fuentes metodológicas |

## Síntesis narrativa

A continuación, la síntesis está organizada como material de trabajo para que luego Codex redacte prosa académica en español. No es todavía texto final de memoria.

| Subapartado | Idea central | Fuentes a citar | Lo que la tesis debería decir | Lo que la tesis debería evitar decir | Conexión con tu implementación |
|---|---|---|---|---|---|
| **Contexto de NIDS** | Un NIDS monitoriza tráfico de red para detectar actividad maliciosa o anómala; no es sinónimo de firewall ni garantiza prevención activa | [ScarfoneMell2007], [Ahmad2021] | Que los NIDS son una capa de monitorización/detección dentro de una arquitectura defensiva más amplia | Que “un NIDS bloquea ataques” por definición; eso corresponde a IPS/IDPS en línea | Tu prototipo toma decisiones binarias PERMIT/BLOCK, pero en la fase final sólo hace inferencia offline, no bloqueo activo. fileciteturn4file0L3-L3 |
| **Enfoque flow-based** | Los flujos agregan paquetes en registros con estadísticas; son útiles por coste, estandarización y disponibilidad en routers, colectores y datasets | [Claise2013], [Sharafaldin2018], [UNB_IDS2018_Page], [Sarhan2020], [ElMahdaouy2026_weak] | Que un enfoque flow-based es razonable para un TFG porque opera sobre telemetría disponible y reproducible, incluso cuando el payload no está disponible o está cifrado | Que los flujos “capturan toda la semántica” del ataque; pierden contexto de contenido y secuencia fina | Tu pipeline usa un esquema canónico fijo de 76 características flow-based y una máscara de missingness, produciendo observaciones de 152 dimensiones. fileciteturn4file0L3-L3 |
| **Datasets públicos** | Los datasets públicos hacen posible entrenar, comparar y reproducir; al mismo tiempo introducen riesgos de sobreajuste al benchmark | [Ring2019], [Sharafaldin2018], [UNB_IDS2018_Page], [MoustafaSlay2015], [Tavallaee2009], [Goldschmidt2025_weak] | Que CICIDS2017, CSE-CIC-IDS2018, UNSW-NB15 y NSL-KDD son referencias muy usadas, cada una con compromisos distintos entre actualidad, realismo y facilidad de uso | Que cualquiera de estos datasets “representa Internet real” sin reservas | Tu TFG usa CICIDS2017 como dataset principal y mantiene NSL-KDD sólo para benchmarking histórico, no como vía final hacia la fase de laboratorio. fileciteturn4file0L3-L3turn5file0L3-L3 |
| **Supervisado y deep learning** | La mayor parte de la literatura NIDS sigue siendo, de hecho, supervisada: árboles, RF, SVM, XGBoost, MLP, CNN, RNN/LSTM, híbridos y combinaciones con selección de características | [Ahmad2021], [Maseer2023], [Corea2024], [HadiMohammed2022] | Que cualquier evaluación de RL debe compararse al menos con uno o varios baselines supervisados fuertes | Que RL desplaza automáticamente al aprendizaje supervisado en este problema | Tu memoria debería justificar un baseline supervisado sobre el mismo vector canónico, no sólo comparar entre variantes RL |
| **Fundamentos de RL/DRL** | En RL el agente observa un estado, elige una acción y recibe una recompensa; DQN y sus variantes son especialmente relevantes cuando la acción es discreta | [SuttonBarto2018], [Mnih2015], [vanHasselt2016], [Wang2016], [Mnih2016], [Schulman2017], [Bellemare2017], [Dabney2018] | Que tu formulación convierte cada flujo en una observación y cada decisión de defensa en una acción binaria, con recompensa asimétrica para FP/FN | Que esta formulación reproduce fielmente toda la dinámica secuencial de una red operativa | Tu entorno define `0 = PERMIT` y `1 = BLOCK`, con recompensas diferenciadas para TP, FP y FN; eso es coherente con un problema de decisión discreta, pero secuencialmente simplificado. fileciteturn9file0L3-L3turn7file0L3-L3 |
| **RL/DRL para NIDS y defensa** | Sí existen trabajos que reformulan la detección o clasificación de intrusiones como problema RL, pero una parte importante de ellos se parece más a clasificación con recompensa que a control secuencial rico | [Alavizadeh2021], [Strickland2023], [Tellache2024], [Yang2024_weak], [Gueriani2024_weak], [Palmer2023_weak] | Que RL en ciberseguridad cubre al menos dos familias: detección/clasificación sobre datasets y defensa adaptativa/respuesta en entornos más complejos | Que todo trabajo RL para NIDS sea directamente comparable con tu prototipo | Tu proyecto está más cerca de la familia “dataset-as-environment, sample-as-state, action-as-label/decision” que de la defensa autónoma online sobre topologías o simuladores multiagente |
| **Limitaciones metodológicas** | La literatura sufre inflación de métricas, validaciones débiles y escasa generalización entre dominios | [Arp2020], [Layeghy2021], [Layeghy2022], [Cantone2024], [Ring2019] | Que el valor de tu TFG depende más de la prudencia metodológica que de perseguir una cifra máxima de accuracy | Que un score alto en un split aleatorio bastaría para hablar de aplicabilidad real | Tu repositorio ya distingue entre split aleatorio, split duro por día/CSV, checks anti-leakage y validación leave-one-CSV-out; eso es una fortaleza que conviene contar con precisión. fileciteturn4file0L3-L3turn5file0L3-L3 |
| **Posicionamiento del TFG** | La aportación razonable no es “inventar RL para NIDS”, sino estudiar una formulación binaria explícita de defensor RL con QRDQN y evaluación más cauta | [Yang2024_weak], [Gueriani2024_weak], [Alavizadeh2021], [Tellache2024], [Arp2020], [Cantone2024] | Que el TFG explora una configuración concreta, reproducible y acotada: PERMIT/BLOCK por flujo, entrenamiento offline en benchmark público y validación externa offline en tráfico privado de laboratorio | Que el TFG resuelve el despliegue operativo de un NIDS/IPS real | El repositorio deja claro que la fase 1 es entrenamiento/validación offline en datasets y la fase 2 es inferencia offline sobre tráfico propio; el bloqueo activo no está implementado. fileciteturn4file0L3-L3turn5file0L3-L3 |

Hay además una observación importante para la redacción final: en la literatura revisada, DQN y sus derivados tienen mucha más visibilidad que el RL distribucional explícito en NIDS; las encuestas recientes sobre DRL para NIDS enfatizan DQN, arquitecturas actor-critic e híbridos, y al mismo tiempo señalan que muchas tecnologías DRL recientes siguen poco exploradas. Por tanto, un encuadre prudente para QRDQN sería “algoritmo técnicamente plausible y poco representado en este subcampo”, no “nuevo estado del arte confirmado en NIDS”. citeturn15academia1turn15academia0turn16academia3turn14academia3

También conviene separar con mucho cuidado la lógica del entorno RL de la lógica del despliegue defensivo. En tu implementación, el entorno es un Gym de clasificación secuencial sobre muestras etiquetadas, donde la acción selecciona permitir o bloquear y la recompensa penaliza especialmente los falsos negativos; esto está bien para un prototipo académico, pero la tesis no debería presentar esa secuencia de muestras como una simulación fiel de una red en producción. fileciteturn9file0L3-L3turn7file0L3-L3

## Base de fuentes

### Matriz de fuentes

La matriz siguiente prioriza fuentes de mayor carga argumental: estándares, páginas oficiales de datasets, artículos fundacionales, trabajos de generalización/evaluación y los precedentes RL más cercanos. Cuando una fuente es preprint o la evidencia es aún débil, lo marco expresamente. La literatura y las páginas oficiales coinciden en que los datasets públicos siguen siendo imprescindibles por la escasez de trazas reales compartibles, pero también en que esa dependencia produce una brecha persistente entre benchmark y despliegue. citeturn19view0turn20view0turn20view2turn24academia7turn29academia1turn29academia0turn30academia0

| Citation key | Referencia completa | Año | Tipo | Tema | Método / algoritmo | Dataset(s) | Tarea | Métricas | Protocolo de evaluación | Contribución principal | Limitaciones | Relevancia para tu TFG | Sección recomendada | DOI / arXiv / enlace |
|---|---|---:|---|---|---|---|---|---|---|---|---|---|---|---|
| **ScarfoneMell2007** | Scarfone, K. A., Mell, P. M. *Guide to Intrusion Detection and Prevention Systems (IDPS)*, NIST SP 800-94 | 2007 | estándar/guía | IDS/IDPS | — | — | Marco conceptual | — | Guía técnica | Define clases de IDPS y contexto operativo | Antigua; no cubre ML moderno | Alta para marco base | Contexto de NIDS | NIST SP 800-94 |
| **Claise2013** | Claise, B., Trammell, B., Aitken, P. *RFC 7011: Specification of the IPFIX Protocol* | 2013 | estándar | Flow telemetry | IPFIX | — | Estandarización | — | Estándar IETF | Base formal del intercambio de flow information | No es fuente NIDS específica | Alta para justificar flujo | Flow-based | RFC 7011 |
| **Sharafaldin2018** | Sharafaldin, I., Lashkari, A. H., Ghorbani, A. A. *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization* | 2018 | dataset paper | CICIDS2017 | CICFlowMeter features | CICIDS2017 | Dataset / benchmark | — | Presentación del dataset | Documento canónico asociado a CICIDS2017 | Dataset sintético; no equivale a red productiva | Muy alta | Datasets | ICISSP 2018 |
| **UNB_IDS2017_Page** | UNB CIC. *Intrusion detection evaluation dataset (CIC-IDS2017)* | 2026 acc. | página oficial dataset | CICIDS2017 | >80 flow features | CICIDS2017 | Dataset / benchmark | — | Documentación oficial | Fechas, ataques, criterios de diseño, features | Página institucional, no paper revisado | Muy alta | Datasets | UNB dataset page |
| **UNB_IDS2018_Page** | UNB CIC. *CSE-CIC-IDS2018 on AWS* | 2026 acc. | página oficial dataset | CSE-CIC-IDS2018 | CICFlowMeter-V3 | CSE-CIC-IDS2018 | Dataset / benchmark | — | Documentación oficial | Describe motivación, listas de ataques y features | No sustituye a un paper canónico independiente | Alta | Datasets | UNB dataset page |
| **MoustafaSlay2015** | Moustafa, N., Slay, J. *UNSW-NB15: a comprehensive data set for network intrusion detection systems* | 2015 | dataset paper | UNSW-NB15 | Argus/Bro-derived 49 features | UNSW-NB15 | Dataset / benchmark | — | Train/test split publicado | Dataset moderno respecto a KDD, con 9 tipos de ataque | También sintético/híbrido | Muy alta | Datasets | UNSW page / MilCIS 2015 |
| **Tavallaee2009** | Tavallaee, M., Bagheri, E., Lu, W., Ghorbani, A. A. *A Detailed Analysis of the KDD CUP 99 Data Set* y base NSL-KDD | 2009 | dataset critique | NSL-KDD | — | KDD99 / NSL-KDD | Benchmark | — | Dataset redesign | Justifica NSL-KDD frente a redundancias de KDD99 | Sigue siendo poco realista para tráfico moderno | Alta, pero con caveat | Datasets | NSL-KDD / paper |
| **Ring2019** | Ring, M. et al. *A Survey of Network-based Intrusion Detection Data Sets* | 2019 | survey | datasets NIDS | — | múltiples | Survey de datasets | — | Revisión comparativa | Sintetiza propiedades de datasets NIDS | Preprint en la evidencia consultada | Muy alta | Datasets / metodología | arXiv:1903.02460 |
| **Goldschmidt2025_weak** | Goldschmidt, P., Chudá, D. *Network Intrusion Datasets: A Survey, Limitations, and Recommendations* | 2025 | survey, **preprint** | datasets NIDS | — | 89 datasets | SLR | — | SLR | Popularidad y recomendaciones recientes | A fecha de revisión, evidencia localizada como preprint | Moderada | Datasets / limitaciones | arXiv:2502.06688 |
| **Ahmad2021** | Ahmad, Z. et al. *Network intrusion detection system: A systematic study of machine learning and deep learning approaches* | 2021 | survey | ML/DL para NIDS | ML/DL variado | múltiples | Survey | accuracy, precision, recall, etc. | Revisión sistemática | Referencia panorámica ampliamente citada | No es benchmark propio | Muy alta | Supervisado/DL | TETT 2021 |
| **Maseer2023** | Maseer, Z. K. et al. *Meta-Analysis and Systematic Review for Anomaly NIDS* | 2023 | survey, **preprint** | métodos, datasets, validación | ML/DL | múltiples | SLR/meta-análisis | varias | revisión | Enfatiza validación y desafíos | Evidencia localizada como preprint | Moderada | Supervisado/DL / metodología | arXiv:2308.02805 |
| **Sarhan2020** | Sarhan, M. et al. *NetFlow Datasets for Machine Learning-based NIDS* | 2020 | benchmark/dataset | flow-based | NetFlow features | NF-UNSW, NF-CSE-CIC-IDS2018, etc. | dataset + baseline | binary/multiclass performance | Conversión a conjunto común de features | Defiende un feature space común y operativo | Resultados preliminares; no resuelve todo el dominio | Muy alta | Flow-based / datasets | arXiv:2011.09144 |
| **Sarhan2021** | Sarhan, M., Layeghy, S., Portmann, M. *Evaluating Standard Feature Sets Towards Increased Generalisability and Explainability of ML-based NIDS* | 2021 | benchmark/método | generalización | ML + SHAP | CSE-CIC-IDS2018, BoT-IoT, ToN-IoT | Binary/multiclass | accuracy, explainability | Cross-dataset feature comparison | Muestra utilidad de feature sets estandarizados | No estudia RL | Alta | Supervisado / metodología | arXiv:2104.07183 |
| **Layeghy2021** | Layeghy, S., Gallagher, M., Portmann, M. *Benchmarking the Benchmark: Analysis of Synthetic NIDS Datasets* | 2021 | benchmark critique | realismo de datasets | análisis estadístico | varios sintéticos + redes reales | comparación de datasets | distribuciones estadísticas | Comparación sintético vs real | Cuestiona transferencia a producción | No es evaluación de modelos RL | Muy alta | Limitaciones metodológicas | arXiv:2104.09029 |
| **Layeghy2022** | Layeghy, S., Portmann, M. *On Generalisability of ML-based NIDS* | 2022 | benchmark critique | generalización | 7 modelos sup./unsup. | 4 datasets | cross-dataset NIDS | varias | entrenamiento en un dominio, test en otro | Muestra mala generalización y asimetría entre dominios | No cubre RL | Muy alta | Limitaciones metodológicas | arXiv:2205.04112 |
| **Cantone2024** | Cantone, M., Marrocco, C., Bria, A. *On the Cross-Dataset Generalization of Machine Learning for NIDS* | 2024 | benchmark | cross-dataset | 4 clasificadores | CIC-IDS2017, CSE-CIC-IDS2018, LycoS... | generalización | accuracy, etc. | cross-dataset | Rendimiento casi aleatorio entre datasets en muchos casos | Preprint en evidencia revisada | Muy alta | Limitaciones / validación externa | arXiv:2402.10974 |
| **Arp2020** | Arp, D. et al. *Dos and Don'ts of Machine Learning in Computer Security* | 2020 | metodología | ML en seguridad | — | seguridad en general | guía crítica | — | estudio de 30 papers + análisis empírico | Identifica pitfalls generalizables a ciberseguridad | No es fuente NIDS específica | Muy alta | Metodología | arXiv:2010.09470 |
| **SuttonBarto2018** | Sutton, R. S., Barto, A. G. *Reinforcement Learning: An Introduction* | 2018 | libro fundacional | RL | RL general | — | teoría | — | — | Base conceptual de estado, acción, recompensa, política | No específica de NIDS | Muy alta | Fundamentos RL | MIT Press |
| **Mnih2015** | Mnih, V. et al. *Human-level Control through Deep Reinforcement Learning* | 2015 | method paper | DQN | DQN | Atari | control | score | benchmark RL | DQN con replay y target network | No ciberseguridad | Muy alta | Fundamentos RL | Nature / doi:10.1038/nature14236 |
| **vanHasselt2016** | van Hasselt, H., Guez, A., Silver, D. *Deep Reinforcement Learning with Double Q-learning* | 2016 | method paper | Double DQN | DDQN | Atari | control | score | benchmark RL | Reduce sobreestimación | No NIDS | Alta | Fundamentos RL | AAAI |
| **Wang2016** | Wang, Z. et al. *Dueling Network Architectures for Deep RL* | 2016 | method paper | Dueling DQN | Dueling DQN | Atari | control | score | benchmark RL | Separa valor de estado y ventaja | No NIDS | Alta | Fundamentos RL | ICML/PMLR |
| **Mnih2016** | Mnih, V. et al. *Asynchronous Methods for Deep RL* | 2016 | method paper | A3C/A2C family | actor-critic | Atari et al. | control | score | benchmark RL | Base de A3C; A2C deriva como variante síncrona práctica | No NIDS | Alta | Fundamentos RL | ICML |
| **Schulman2017** | Schulman, J. et al. *Proximal Policy Optimization Algorithms* | 2017 | method paper | PPO | PPO | múltiples | control | reward | benchmark RL | Algoritmo actor-critic robusto y popular | No NIDS | Alta | Fundamentos RL | arXiv:1707.06347 |
| **Bellemare2017** | Bellemare, M. G. et al. *A Distributional Perspective on Reinforcement Learning* | 2017 | method paper | distributional RL | C51 | Atari | control | score | benchmark RL | Introduce RL distribucional | No NIDS | Muy alta | Fundamentos RL | ICML/PMLR |
| **Dabney2018** | Dabney, W. et al. *Distributional Reinforcement Learning with Quantile Regression* | 2018 | method paper | QR-DQN | QRDQN | Atari | control | score | benchmark RL | Base primaria del algoritmo usado en tu TFG | No NIDS | Muy alta | Fundamentos RL / conexión al TFG | AAAI |
| **Alavizadeh2021** | Alavizadeh, H., Jang-Jaccard, J., Alavizadeh, H. *Deep Q-Learning based RL Approach for Network Intrusion Detection* | 2021 | primary paper | RL para NIDS | DQL | NSL-KDD | detección/clasificación | accuracy y clases | entrenamiento por episodios | Precedente directo de RL-clasificación para IDS | Dataset viejo; realismo limitado | Muy alta | RL para NIDS | arXiv:2111.13978 |
| **Strickland2023** | Strickland, C. et al. *DRL-GAN: A Hybrid Approach for Binary and Multiclass Network Intrusion Detection* | 2023 | primary paper | RL + datos sintéticos | DRL + GAN | NSL-KDD | binary/multiclass IDS | clasificación | comparación dataset real/sintético | Muestra uso de RL como clasificador reforzado | Mezcla dos técnicas; dataset viejo; evidencia preprint | Alta | RL para NIDS | arXiv:2301.03368 |
| **Tellache2024** | Tellache, A. et al. *Multi-agent Reinforcement Learning-based Network Intrusion Detection System* | 2024 | primary paper, **preprint** | RL para NIDS | multi-agent DQN mejorado | CIC-IDS2017 | detección y clasificación fina | FPR, detection rate | CICIDS2017 | RL más cercano a dataset moderno | Evidencia localizada como preprint; necesidad de auditoría metodológica | Alta | RL para NIDS | arXiv:2407.05766 |
| **Yang2024_weak** | Yang, W. et al. *A Survey for Deep Reinforcement Learning Based Network Intrusion Detection* | 2024 | survey, **preprint** | RL para NIDS | DQN, actor-critic, híbridos | varios | survey | varias | review | Señala concentración en ciertas familias y lagunas | Preprint | Moderada | RL para NIDS / gap | arXiv:2410.07612 |
| **Gueriani2024_weak** | Gueriani, A. et al. *Deep Reinforcement Learning for Intrusion Detection in IoT: A Survey* | 2024 | survey, **preprint** | DRL IDS IoT | varias familias DRL | varios | survey | varias | review | Resume datasets, métricas y categorías | IoT-centred; preprint | Moderada | RL para NIDS | arXiv:2405.20038 |
| **Palmer2023_weak** | Palmer, G. et al. *Deep Reinforcement Learning for Autonomous Cyber Defence: A Survey* | 2023 | survey, **preprint** | defensa autónoma | DRL general | entornos ACD | survey | — | review | Diferencia detección en datasets de defensa autónoma realista | No específica de per-flow classification | Alta | RL y cyber defense | arXiv:2310.07745 |
| **She2025_weak** | She, Y. *A Robust PPO-optimized Tabular Transformer Framework for Intrusion Detection in IIoT Systems* | 2025 | primary paper, **preprint** | RL + PPO como clasificador | PPO + TabTransformer | TON_IoT | IDS | macro-F1, accuracy | evaluación en benchmark | Señala precedentes recientes de PPO para clasificación IDS | Evidencia débil; preprint único en la muestra revisada | Moderada-baja | RL para NIDS / algoritmos | arXiv:2505.18234 |

### Selección imprescindible

Si tuvieras que reducir la bibliografía del capítulo a un núcleo muy defendible, esta sería la selección imprescindible.

**NIDS y enfoques flow-based**: [ScarfoneMell2007], [Claise2013], [Sarhan2020], [Sarhan2021], [ElMahdaouy2026_weak]. citeturn28view1turn27view0turn25academia2turn25academia0turn24academia0

**Datasets públicos**: [Sharafaldin2018], [UNB_IDS2017_Page], [UNB_IDS2018_Page], [MoustafaSlay2015], [Tavallaee2009], [Ring2019], [Goldschmidt2025_weak]. citeturn19view0turn20view0turn21view0turn20view2turn24academia7turn6academia0

**ML/DL supervisado para NIDS**: [Ahmad2021], [Maseer2023], [Sarhan2021], [Corea2024]. citeturn25search7turn9academia2turn25academia0turn7academia3

**Fundamentos de RL/DRL**: [SuttonBarto2018], [Mnih2015], [vanHasselt2016], [Wang2016], [Mnih2016], [Schulman2017], [Bellemare2017], [Dabney2018]. Estas son fuentes fundacionales estables y no requieren actualización reciente para ser válidas.

**RL/DRL para NIDS**: [Alavizadeh2021], [Strickland2023], [Tellache2024], [Yang2024_weak], [Gueriani2024_weak]. citeturn14academia3turn16academia1turn16academia3turn15academia1turn15academia0

**Cyber defense y defensa adaptativa**: [Palmer2023_weak], y como contraste ofensivo/adversarial [DeepPackGen2023]. citeturn16academia2turn17academia2

**Riesgos metodológicos**: [Arp2020], [Layeghy2021], [Layeghy2022], [Cantone2024]. citeturn30academia0turn29academia1turn29academia0turn18academia2

## Estado del arte cercano y crítica metodológica

### Trabajos más cercanos a tu proyecto

Los trabajos más cercanos no son, curiosamente, los de “autonomous cyber defence” con varios agentes o topologías ricas, sino los que convierten instancias etiquetadas de tráfico en estados y usan acciones discretas como clases o decisiones de respuesta. Ese subgrupo es conceptualmente el antecedente directo de tu TFG. citeturn14academia3turn16academia1turn16academia3turn17academia0

| Paper | Qué hace | Cercanía a tu proyecto | Similitudes | Diferencias | Fortaleza metodológica | Debilidad metodológica | ¿Citar como precedente directo? |
|---|---|---|---|---|---|---|---|
| **Alavizadeh2021** | Usa Deep Q-Learning para NIDS sobre NSL-KDD | **Muy alta** | RL como clasificador; acciones discretas; entrenamiento por recompensas; dataset etiquetado | Dataset antiguo; no usa CICIDS2017; no usa RL distribucional | Precedente claro del “sample-as-state, action-as-class/decision” | Riesgo de sobreinterpretar RL donde el problema se parece mucho a clasificación supervisada reforzada | **Sí, imprescindible** |
| **Strickland2023** | Combina GAN y DRL para detección binaria y multiclase en NSL-KDD | **Alta** | RL aplicado a detección/clasificación sobre dataset tabular | Añade generación sintética; enfoque híbrido | Interesante para ataques minoritarios | Dataset antiguo; más complejo y menos comparable de forma limpia | **Sí, como precedente cercano pero no isomorfo** |
| **Tellache2024** | IDS multiagente con DQN mejorado y cost-sensitive learning en CICIDS2017 | **Alta** | Usa CICIDS2017; RL para detección con foco en imbalance y FPR | Arquitectura multiagente, no binaria simple; objetivo más ambicioso | Más cercano en dataset moderno | A fecha de revisión lo localizado es preprint; exige lectura crítica de splits y leakage | **Sí** |
| **She2025_weak** | Usa PPO para optimizar decisiones de clasificación IDS sobre tabla | **Media** | Muestra que PPO también se está usando como mecanismo de decisión clasificatoria | IIoT; TabTransformer; otro benchmark | Útil para la discusión de algoritmos actor-critic en IDS | Evidencia débil y muy reciente; no debe sobrepesarse | **Sí, pero sólo como evidencia complementaria** |
| **Palmer2023_weak** | Survey de defensa autónoma con DRL | **Media conceptual** | Enmarca RL más allá de clasificar flujos | Enfocado a ACD completo, no per-flow binary decisions | Muy útil para delimitar alcance | No sirve como benchmark directo de tu prototipo | **Sí, para acotar lo que tu TFG no pretende hacer** |
| **Yang2024_weak** y **Gueriani2024_weak** | Surveys recientes de DRL para NIDS/IoT IDS | **Alta para contexto, no para comparación cuantitativa** | Identifican familias algorítmicas recurrentes | Surveys, no experimentos propios equiparables | Buenos para mapear el campo | Preprints; debes señalarlos como tales | **Sí** |

La conclusión útil para tu capítulo es ésta: el precedente más directo de tu planteamiento no es “RL para cyber defence” en abstracto, sino la reformulación de clasificación IDS como decisión RL sobre muestras etiquetadas. Eso te permite decir que tu problema **sí tiene precedentes**, pero también que esos precedentes mezclan con frecuencia la semántica de RL y la de clasificación supervisada, lo cual exige cautela interpretativa. citeturn14academia3turn16academia1turn16academia3turn15academia2

### Crítica metodológica del área

La primera debilidad recurrente es la **inflación de rendimiento intra-dataset**. Hay múltiples trabajos que reportan precisiones o F1 casi perfectos dentro del mismo benchmark, en especial con datasets sintéticos o semisintéticos, pero la literatura crítica destaca que esos resultados han traducido mal a entornos más realistas. Layeghy y colegas muestran diferencias estadísticas claras entre datasets sintéticos NIDS y tráfico real, y tanto sus trabajos como el de Cantone et al. muestran que el rendimiento cross-dataset puede caer drásticamente, a veces hasta niveles cercanos al azar. Arp et al., desde una perspectiva más general de ML en seguridad, argumentan que este patrón no es un accidente aislado sino un síntoma de pitfalls metodológicos más amplios. citeturn29academia1turn29academia0turn18academia2turn30academia0

La segunda debilidad es el **uso excesivo de splits aleatorios**. En NIDS tabulares, un split aleatorio puede mezclar flujos del mismo escenario, mismo día, mismo generador de tráfico o misma campaña entre entrenamiento y prueba. Eso no siempre implica leakage formal en el sentido clásico, pero sí puede generar una prueba demasiado fácil. Tu propio repositorio ya refleja esta preocupación al diferenciar entre split aleatorio, split duro por día/CSV, shuffled-label anti-leakage test y leave-one-exact-CSV-out, lo cual es exactamente el tipo de disciplina que la memoria debería subrayar. fileciteturn4file0L3-L3turn5file0L3-L3

La tercera debilidad es el **leakage o uso de proxies de etiqueta**. En datasets flow-based es muy fácil que identificadores, puertos, timestamps absolutos o artefactos de exportación actúen como atajos espurios. Lo mejor de tu implementación actual es que formaliza una política anti-leakage explícita: excluye IPs, timestamps absolutos, Flow IDs y campos de puerto cuando actúan como proxies de etiqueta. Eso conecta muy bien con la crítica metodológica general de Arp et al. y con la práctica prudente exigible en un TFG serio. fileciteturn4file0L3-L3turn5file0L3-L3 citeturn30academia0

La cuarta debilidad es la **desatención al desbalance y al coste asimétrico de los errores**. En NIDS, un falso negativo y un falso positivo no tienen el mismo coste operacional. Parte de la literatura RL justifica precisamente el uso de funciones de recompensa asimétricas para reflejar ese desbalance de costes, pero esa misma flexibilidad puede convertir la comparación entre papers en algo opaco si no se documentan de forma rigurosa las recompensas. Aquí tu memoria debería ser particularmente cuidadosa, porque en el repositorio aparece una tensión documental sobre el valor exacto de la penalización por FP entre documentación y código; por tanto, la tesis debería anclar cualquier afirmación numérica a la versión exacta del experimento o artefacto utilizado, no a una descripción genérica del repositorio. fileciteturn5file0L3-L3turn7file0L3-L3turn9file0L3-L3

La quinta debilidad es la **ausencia de validación externa**. Muchos papers entrenan y evalúan sobre el mismo dataset público, a veces incluso con múltiples transformaciones del mismo origen. Dado que la literatura reciente cuestiona fuertemente la transportabilidad cross-dataset, una evaluación adicional sobre tráfico de laboratorio propio, aunque sea offline y sin bloqueo activo, es valiosa como prueba de robustez bajo cambio de dominio. Pero esa validación externa debe presentarse como lo que es: un **stress test de transportabilidad** y no una demostración definitiva de validez operativa universal. citeturn29academia0turn18academia2turn30academia0

La sexta debilidad es la **falta de reproducibilidad fina**: seeds únicos, preprocesado no versionado, poca claridad sobre splits y ausencia de artefactos. En este punto tu repositorio va mejor encaminado que buena parte de la literatura, porque ya articula identificadores de ejecución y persistencia de artefactos bajo `runs/`, además de scripts específicos de validación; sin embargo, para que la memoria sea plenamente defendible, conviene añadir explícitamente múltiples seeds, curvas de eficiencia de datos y baselines supervisados bajo el mismo preprocesado. fileciteturn4file0L3-L3turn8file0L3-L3

### Cómo puede responder tu diseño experimental

La justificación más limpia para tu diseño sería la siguiente:

**Benchmark interno sobre CICIDS2017.** Es razonable porque CICIDS2017 sigue siendo un benchmark público ampliamente usado, con documentación oficial detallada, variedad de ataques y CSV flow-based directamente utilizables. Sirve para reproducibilidad, comparación interna entre variantes y depuración del pipeline. citeturn19view0turn24academia7

**Validación externa sobre tráfico privado de laboratorio.** Es razonable porque la literatura crítica cuestiona que el rendimiento intra-dataset sea transportable. Probar inferencia offline en flujos derivados de capturas privadas añade evidencia sobre robustez ante cambio de distribución, aunque no resuelve por sí solo el problema de despliegue real. citeturn29academia1turn29academia0turn18academia2turn30academia0

**Curva de eficiencia de datos.** Es útil porque muestra si el enfoque RL necesita volúmenes desproporcionados de datos frente a baselines supervisados. En un TFG esto puede ser más informativo que buscar únicamente el máximo score final.

**Baseline supervisado.** Es imprescindible porque el problema, tal como lo formulas, también puede entenderse como clasificación binaria tabular con coste asimétrico. Sin baseline fuerte, un tribunal podría interpretar que RL se ha elegido por novedad percibida, no por evidencia comparativa.

**Múltiples seeds.** Deben usarse para evitar que una única ejecución afortunada domine la conclusión.

**Error analysis.** En vez de centrarte sólo en accuracy global, conviene analizar falsos positivos y falsos negativos por familias de tráfico, días o ficheros, porque el coste de estos errores es diferente en defensa.

**Fallback strict split.** Si el tiempo no permite una validación externa amplia, el mínimo metodológico debería incluir al menos un split duro por día/CSV y, si es posible, leave-one-exact-CSV-out, que tu código ya contempla. fileciteturn4file0L3-L3turn5file0L3-L3

### Brecha que tu TFG puede reclamar sin exagerar

La brecha razonable no es “falta RL en ciberseguridad” ni “nadie ha usado RL para IDS”. Eso sería insostenible. La brecha defendible es más concreta:

1. **La literatura cercana a NIDS con RL está dominada por reformulaciones basadas en DQN u otros enfoques DRL genéricos, con pocos indicios claros de adopción de RL distribucional tipo QRDQN en este subcampo concreto.** La evidencia aquí es moderada, no fuerte, porque se apoya en surveys recientes y en la muestra revisada, no en un mapeo bibliométrico exhaustivo. citeturn15academia1turn15academia0

2. **Hay escasez de trabajos que, además de proponer un agente RL para decisiones binarias de defensa sobre flujos, documenten una evaluación explícitamente prudente frente a leakage, splits duros y validación externa fuera del dataset público de entrenamiento.** Esto sí enlaza bien con la crítica metodológica reciente. citeturn30academia0turn29academia0turn18academia2

3. **Tu TFG no aporta una nueva teoría RL ni un sistema IPS desplegado, sino una formulación experimental concreta y reproducible: per-flow binary defender, reward engineering consciente de FP/FN, QRDQN en un entorno dataset-as-environment, benchmark interno en dataset público y comprobación externa offline en laboratorio propio.** Eso es modesto, veraz y defendible. fileciteturn4file0L3-L3turn5file0L3-L3turn7file0L3-L3turn9file0L3-L3

## Afirmaciones seguras, afirmaciones a evitar y glosario

### Afirmaciones seguras

| Claim | Citation keys | Fuerza de evidencia | Caveat | Formulación sugerida en español |
|---|---|---|---|---|
| Los NIDS son una tecnología central de monitorización/detección, distinta de un firewall tradicional y de un IPS en línea | [ScarfoneMell2007] | **Fuerte** | Guía clásica, no ML-specific | “Los sistemas de detección de intrusiones en red constituyen una capa de monitorización y detección diferenciada de los mecanismos de filtrado tradicionales.” |
| Los enfoques flow-based son comunes porque se apoyan en telemetría estandarizada y operativamente extraíble | [Claise2013], [Sarhan2020], [UNB_IDS2017_Page], [UNB_IDS2018_Page] | **Fuerte-moderada** | No todos los NIDS operan sólo a nivel de flujo | “Los enfoques basados en flujos resultan especialmente atractivos cuando se priorizan escalabilidad, interoperabilidad y disponibilidad de telemetría.” |
| CICIDS2017, CSE-CIC-IDS2018, UNSW-NB15 y NSL-KDD siguen teniendo un papel importante como benchmarks públicos | [Ring2019], [UNB_IDS2017_Page], [UNB_IDS2018_Page], [MoustafaSlay2015], [Tavallaee2009] | **Fuerte-moderada** | Su popularidad no implica realismo pleno | “Estos conjuntos de datos siguen desempeñando un papel relevante como puntos de referencia experimentales, aunque presentan limitaciones conocidas.” |
| En NIDS, la literatura dominante es todavía supervisada o DL supervisado | [Ahmad2021], [Maseer2023] | **Moderada** | Algunas revisiones localizadas como preprint | “La mayor parte de la literatura reciente en NIDS continúa formulando el problema como clasificación supervisada o deep learning supervisado.” |
| Sí existen trabajos que formulan la detección de intrusiones como problema RL sobre muestras etiquetadas | [Alavizadeh2021], [Strickland2023], [Tellache2024] | **Moderada** | Heterogeneidad metodológica y distinto nivel de madurez | “Existen precedentes en los que la detección o clasificación de intrusiones se reformula como un problema de decisión por refuerzo sobre instancias etiquetadas.” |
| En la muestra revisada predominan DQN y variantes afines; QRDQN no aparece como opción ampliamente asentada en NIDS | [Yang2024_weak], [Gueriani2024_weak], [Alavizadeh2021], [Tellache2024] | **Moderada-baja** | Basado en surveys recientes y muestra revisada, no en estudio bibliométrico exhaustivo | “En la literatura revisada predominan formulaciones basadas en DQN y arquitecturas relacionadas, mientras que el uso de RL distribucional tipo QRDQN parece todavía poco representado.” |
| Los resultados casi perfectos intra-dataset no garantizan generalización fuera del dominio del benchmark | [Layeghy2021], [Layeghy2022], [Cantone2024], [Arp2020] | **Fuerte** | Magnitud exacta de la caída depende del par de datasets y del protocolo | “Los resultados obtenidos dentro de un mismo benchmark no deben interpretarse automáticamente como evidencia de transferibilidad a otros entornos.” |
| Una validación externa offline en tráfico de laboratorio es metodológicamente valiosa, aunque no equivalga a despliegue real | [Layeghy2022], [Cantone2024], [Arp2020] | **Moderada-fuerte** | Sigue siendo laboratorio privado, no producción | “La evaluación adicional sobre tráfico capturado en laboratorio aporta evidencia de robustez frente a cambio de distribución, aun sin constituir una validación operativa completa.” |
| Tu prototipo implementa un entorno dataset-as-environment con acciones binarias PERMIT/BLOCK y QRDQN, pero no bloqueo activo en tiempo real | repositorio TFG | **Fuerte** | La tesis debe fijar versión/artefacto exacto | “El prototipo desarrollado se formula como un entorno de decisión binaria sobre flujos etiquetados y se evalúa en modo offline; no implementa todavía prevención activa en línea.” |

### Afirmaciones a evitar

| Afirmación a evitar | Por qué es indefendible | Alternativa más segura |
|---|---|---|
| “Este TFG es el primero en aplicar RL a la detección de intrusiones.” | Falso: hay precedentes claros | “Este TFG explora una formulación concreta y acotada de RL para decisiones binarias PERMIT/BLOCK sobre flujos.” |
| “QRDQN es el algoritmo estándar o dominante en NIDS.” | No hay evidencia sólida para eso | “QRDQN pertenece a la familia del RL distribucional y, en la muestra revisada, aparece como una opción poco representada en NIDS.” |
| “Los resultados sobre CICIDS2017 demuestran eficacia en entornos reales.” | La literatura crítica cuestiona esa extrapolación | “Los resultados sobre CICIDS2017 muestran comportamiento en un benchmark ampliamente usado, pero requieren validación adicional fuera del dominio del dataset.” |
| “La fase de laboratorio valida el sistema en condiciones reales.” | Tu fase 2 es inferencia offline, no despliegue inline | “La fase de laboratorio aporta una validación externa offline bajo tráfico propio, útil como prueba de transportabilidad parcial.” |
| “El entorno RL modela fielmente una defensa de red real.” | El entorno es una simplificación por muestras/filas | “El entorno RL constituye una abstracción experimental del problema de decisión por flujo.” |
| “El RL supera de forma inherente al aprendizaje supervisado en NIDS.” | No hay apoyo general para tal afirmación | “El interés del RL aquí reside en su formulación de decisión bajo recompensa asimétrica; su ventaja empírica debe demostrarse frente a baselines supervisados.” |
| “La validación en un único split aleatorio es suficiente.” | Débil ante leakage y correlación de escenarios | “La evaluación debe complementarse con splits duros, múltiples seeds y, si es posible, validación externa.” |

### Glosario de términos en español

| Término | Definición en español |
|---|---|
| **NIDS** | Sistema de detección de intrusiones en red que analiza tráfico o telemetría de red para identificar actividad maliciosa o anómala. |
| **network flow** | Registro agregado de una comunicación de red entre dos extremos durante un intervalo, normalmente resumido mediante contadores y estadísticas. |
| **flow features** | Variables derivadas del flujo, como duración, bytes, paquetes, tasas o estadísticas temporales, usadas como entrada del modelo. |
| **CICIDS2017** | Dataset público del CIC/UNB con tráfico benigno y varios ataques capturados en julio de 2017, más PCAPs y CSVs etiquetados a nivel de flujo. |
| **supervised learning** | Paradigma de aprendizaje en el que el modelo se entrena con ejemplos etiquetados para predecir una clase o valor objetivo. |
| **reinforcement learning** | Paradigma en el que un agente observa un estado, elige una acción y aprende a maximizar recompensa acumulada. |
| **state / observation** | Información que recibe el agente en un instante para decidir; en tu TFG, el vector de características de un flujo. |
| **action** | Decisión tomada por el agente; en tu caso, `0 = PERMIT`, `1 = BLOCK`. |
| **reward** | Señal numérica que evalúa la calidad de la acción elegida. |
| **policy** | Regla o función que asigna acciones a observaciones. |
| **DQN** | Deep Q-Network, algoritmo value-based para acciones discretas que aproxima la función Q con una red neuronal. |
| **QRDQN** | Variante distribucional de DQN basada en regresión por cuantiles, que modela una distribución de retornos en vez de solo su esperanza. |
| **false positive** | Caso benigno clasificado como ataque o bloqueado indebidamente. |
| **false negative** | Caso malicioso clasificado como benigno o permitido indebidamente. |
| **data leakage** | Entrada accidental de información directa o indirectamente correlacionada con la etiqueta en entrenamiento/evaluación, hinchando el rendimiento. |
| **external validation** | Evaluación sobre datos distintos del benchmark principal de entrenamiento, idealmente de otro origen o dominio. |
| **distribution shift** | Cambio entre la distribución de los datos de entrenamiento y la de evaluación o despliegue. |

## Referencias operativas para redacción

### Bloque BibTeX esencial

Las entradas siguientes son **mínimas y utilizables**. En varios casos conviene que, antes de la entrega final, las normalices en Zotero/JabRef con páginas, DOI y venue exactos cuando proceda. No incluyo DOIs dudosos; cuando no los tengo con suficiente confianza, dejo `url` o `eprint`.

```bibtex
@techreport{ScarfoneMell2007,
  author = {Scarfone, Karen A. and Mell, Peter M.},
  title = {Guide to Intrusion Detection and Prevention Systems (IDPS)},
  institution = {National Institute of Standards and Technology},
  year = {2007},
  number = {NIST SP 800-94},
  url = {https://csrc.nist.gov/pubs/sp/800/94/final}
}

@misc{Claise2013,
  author = {Claise, Benoit and Trammell, Brian and Aitken, Paul},
  title = {RFC 7011: Specification of the IP Flow Information Export (IPFIX) Protocol for the Exchange of Flow Information},
  year = {2013},
  howpublished = {RFC 7011},
  url = {https://www.rfc-editor.org/rfc/rfc7011}
}

@inproceedings{Sharafaldin2018,
  author = {Sharafaldin, Iman and Habibi Lashkari, Arash and Ghorbani, Ali A.},
  title = {Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization},
  booktitle = {Proceedings of the International Conference on Information Systems Security and Privacy},
  year = {2018},
  url = {https://www.unb.ca/cic/datasets/ids-2017.html}
}

@misc{UNB_IDS2017_Page,
  author = {{Canadian Institute for Cybersecurity, UNB}},
  title = {Intrusion Detection Evaluation Dataset (CIC-IDS2017)},
  year = {n.d.},
  url = {https://www.unb.ca/cic/datasets/ids-2017.html},
  note = {Accessed 2026-05-14}
}

@misc{UNB_IDS2018_Page,
  author = {{Canadian Institute for Cybersecurity, UNB}},
  title = {CSE-CIC-IDS2018 on AWS},
  year = {n.d.},
  url = {https://www.unb.ca/cic/datasets/ids-2018.html},
  note = {Accessed 2026-05-14}
}

@inproceedings{MoustafaSlay2015,
  author = {Moustafa, Nour and Slay, Jill},
  title = {UNSW-NB15: A Comprehensive Data Set for Network Intrusion Detection Systems},
  booktitle = {Military Communications and Information Systems Conference},
  year = {2015},
  url = {https://research.unsw.edu.au/projects/unsw-nb15-dataset}
}

@misc{Tavallaee2009,
  author = {Tavallaee, Mahbod and Bagheri, Ebrahim and Lu, Wei and Ghorbani, Ali A.},
  title = {A Detailed Analysis of the KDD CUP 99 Data Set},
  year = {2009},
  note = {Basis for NSL-KDD discussion},
  url = {https://www.unb.ca/cic/datasets/nsl.html}
}

@article{Ring2019,
  author = {Ring, Markus and Wunderlich, Sarah and Scheuring, Deniz and Landes, Dieter and Hotho, Andreas},
  title = {A Survey of Network-based Intrusion Detection Data Sets},
  year = {2019},
  eprint = {1903.02460},
  archivePrefix = {arXiv},
  primaryClass = {cs.CR}
}

@article{Ahmad2021,
  author = {Ahmad, Zeeshan and Khan, Adnan Shahid and Shiang, Cheah Wai and Abdullah, Johari and Ahmad, Farhan},
  title = {Network Intrusion Detection System: A Systematic Study of Machine Learning and Deep Learning Approaches},
  journal = {Transactions on Emerging Telecommunications Technologies},
  year = {2021}
}

@article{Sarhan2020,
  author = {Sarhan, Mohanad and Layeghy, Siamak and Moustafa, Nour and Portmann, Marius},
  title = {NetFlow Datasets for Machine Learning-based Network Intrusion Detection Systems},
  year = {2020},
  eprint = {2011.09144},
  archivePrefix = {arXiv},
  primaryClass = {cs.CR}
}

@article{Sarhan2021,
  author = {Sarhan, Mohanad and Layeghy, Siamak and Portmann, Marius},
  title = {Evaluating Standard Feature Sets Towards Increased Generalisability and Explainability of ML-based Network Intrusion Detection},
  year = {2021},
  eprint = {2104.07183},
  archivePrefix = {arXiv},
  primaryClass = {cs.CR}
}

@article{Layeghy2021,
  author = {Layeghy, Siamak and Gallagher, Marcus and Portmann, Marius},
  title = {Benchmarking the Benchmark: Analysis of Synthetic NIDS Datasets},
  year = {2021},
  eprint = {2104.09029},
  archivePrefix = {arXiv},
  primaryClass = {cs.CR}
}

@article{Layeghy2022,
  author = {Layeghy, Siamak and Portmann, Marius},
  title = {On Generalisability of Machine Learning-based Network Intrusion Detection Systems},
  year = {2022},
  eprint = {2205.04112},
  archivePrefix = {arXiv},
  primaryClass = {cs.CR}
}

@article{Cantone2024,
  author = {Cantone, Marco and Marrocco, Claudio and Bria, Alessandro},
  title = {On the Cross-Dataset Generalization of Machine Learning for Network Intrusion Detection},
  year = {2024},
  eprint = {2402.10974},
  archivePrefix = {arXiv},
  primaryClass = {cs.CR}
}

@article{Arp2020,
  author = {Arp, Daniel and Quiring, Erwin and Pendlebury, Feargus and Warnecke, Alexander and Pierazzi, Fabio and Wressnegger, Christian and Cavallaro, Lorenzo and Rieck, Konrad},
  title = {Dos and Don'ts of Machine Learning in Computer Security},
  year = {2020},
  eprint = {2010.09470},
  archivePrefix = {arXiv},
  primaryClass = {cs.CR}
}

@book{SuttonBarto2018,
  author = {Sutton, Richard S. and Barto, Andrew G.},
  title = {Reinforcement Learning: An Introduction},
  edition = {2},
  publisher = {MIT Press},
  year = {2018}
}

@article{Mnih2015,
  author = {Mnih, Volodymyr and Kavukcuoglu, Koray and Silver, David and Rusu, Andrei A. and others},
  title = {Human-level Control through Deep Reinforcement Learning},
  journal = {Nature},
  volume = {518},
  number = {7540},
  pages = {529--533},
  year = {2015},
  doi = {10.1038/nature14236}
}

@inproceedings{vanHasselt2016,
  author = {van Hasselt, Hado and Guez, Arthur and Silver, David},
  title = {Deep Reinforcement Learning with Double Q-learning},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year = {2016}
}

@inproceedings{Wang2016,
  author = {Wang, Ziyu and Schaul, Tom and Hessel, Matteo and Hasselt, Hado van and Lanctot, Marc and de Freitas, Nando},
  title = {Dueling Network Architectures for Deep Reinforcement Learning},
  booktitle = {Proceedings of the 33rd International Conference on Machine Learning},
  year = {2016}
}

@inproceedings{Mnih2016,
  author = {Mnih, Volodymyr and Badia, Adri\`a Puigdom\`enech and Mirza, Mehdi and Graves, Alex and Lillicrap, Timothy and Harley, Tim and Silver, David and Kavukcuoglu, Koray},
  title = {Asynchronous Methods for Deep Reinforcement Learning},
  booktitle = {Proceedings of the 33rd International Conference on Machine Learning},
  year = {2016}
}

@article{Schulman2017,
  author = {Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec and Klimov, Oleg},
  title = {Proximal Policy Optimization Algorithms},
  year = {2017},
  eprint = {1707.06347},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG}
}

@inproceedings{Bellemare2017,
  author = {Bellemare, Marc G. and Dabney, Will and Munos, R{\'e}mi},
  title = {A Distributional Perspective on Reinforcement Learning},
  booktitle = {Proceedings of the 34th International Conference on Machine Learning},
  year = {2017}
}

@inproceedings{Dabney2018,
  author = {Dabney, Will and Rowland, Mark and Bellemare, Marc G. and Munos, R{\'e}mi},
  title = {Distributional Reinforcement Learning with Quantile Regression},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year = {2018}
}

@article{Alavizadeh2021,
  author = {Alavizadeh, Hooman and Jang-Jaccard, Julian and Alavizadeh, Hootan},
  title = {Deep Q-Learning based Reinforcement Learning Approach for Network Intrusion Detection},
  year = {2021},
  eprint = {2111.13978},
  archivePrefix = {arXiv},
  primaryClass = {cs.CR}
}

@article{Strickland2023,
  author = {Strickland, Caroline and Saha, Chandrika and Zakar, Muhammad and Nejad, Sareh and Tasnim, Noshin and Lizotte, Daniel and Haque, Anwar},
  title = {DRL-GAN: A Hybrid Approach for Binary and Multiclass Network Intrusion Detection},
  year = {2023},
  eprint = {2301.03368},
  archivePrefix = {arXiv},
  primaryClass = {cs.CR}
}

@article{Tellache2024,
  author = {Tellache, Amine and Mokhtari, Amdjed and Korba, Abdelaziz Amara and Ghamri-Doudane, Yacine},
  title = {Multi-agent Reinforcement Learning-based Network Intrusion Detection System},
  year = {2024},
  eprint = {2407.05766},
  archivePrefix = {arXiv},
  primaryClass = {cs.CR}
}

@article{Yang2024_weak,
  author = {Yang, Wanrong and Acuto, Alberto and Zhou, Yihang and Wojtczak, Dominik},
  title = {A Survey for Deep Reinforcement Learning Based Network Intrusion Detection},
  year = {2024},
  eprint = {2410.07612},
  archivePrefix = {arXiv},
  primaryClass = {cs.CR}
}

@article{Gueriani2024_weak,
  author = {Gueriani, Afrah and Kheddar, Hamza and Mazari, Ahmed Cherif},
  title = {Deep Reinforcement Learning for Intrusion Detection in IoT: A Survey},
  year = {2024},
  eprint = {2405.20038},
  archivePrefix = {arXiv},
  primaryClass = {cs.CR}
}

@article{Palmer2023_weak,
  author = {Palmer, Gregory and Parry, Chris and Harrold, Daniel J. B. and Willis, Chris},
  title = {Deep Reinforcement Learning for Autonomous Cyber Defence: A Survey},
  year = {2023},
  eprint = {2310.07745},
  archivePrefix = {arXiv},
  primaryClass = {cs.CR}
}
```

### Handoff preciso para Codex

Redacta el **capítulo de Estado del Arte / Antecedentes** de un TFG de Ingeniería Informática, en **español académico**, en prosa continua y bien cohesionada, **no** como apuntes ni listas. El tema del TFG es: **“Reinforcement Learning for cybersecurity: an RL-based network-flow defender for binary PERMIT/BLOCK decisions.”**

Debes crear los siguientes subapartados en este orden lógico:

**Contexto de la detección de intrusiones en red.**  
Usa principalmente: [ScarfoneMell2007], [Ahmad2021].  
Explica qué es un NIDS, cómo se diferencia de un IPS/IDPS y por qué la detección por anomalías motivó el uso de ML. No digas que ML sustituye todos los enfoques clásicos.

**NIDS basados en tráfico flow-based.**  
Usa: [Claise2013], [Sharafaldin2018], [UNB_IDS2017_Page], [UNB_IDS2018_Page], [Sarhan2020], [Sarhan2021].  
Explica qué es un flujo de red, qué tipo de características se extraen y por qué los enfoques flow-based son frecuentes. Menciona ventajas prácticas y límites: escalabilidad e interoperabilidad frente a menor granularidad semántica.

**Datasets públicos en investigación NIDS.**  
Usa: [Sharafaldin2018], [UNB_IDS2017_Page], [UNB_IDS2018_Page], [MoustafaSlay2015], [Tavallaee2009], [Ring2019], [Goldschmidt2025_weak].  
Describe el papel de CICIDS2017, CSE-CIC-IDS2018, UNSW-NB15 y NSL-KDD como benchmarks. Debes enfatizar que son útiles para reproducibilidad y comparación, pero no equivalen a tráfico real de producción. Señala explícitamente que CSE-CIC-IDS2018 se apoya en documentación oficial útil, aunque en la base revisada no aparece con la misma claridad un paper canónico independiente equivalente al de CICIDS2017.

**Aprendizaje supervisado y deep learning para NIDS.**  
Usa: [Ahmad2021], [Maseer2023], [Sarhan2021], [Corea2024].  
Resume que la línea dominante del área sigue siendo la clasificación supervisada sobre tráfico o flujos: árboles, random forest, SVM, XGBoost, MLP, CNN, RNN/LSTM, híbridos. Introduce aquí la idea de que cualquier propuesta RL debe compararse con baselines supervisados razonables.

**Fundamentos de RL/DRL relevantes para el TFG.**  
Usa: [SuttonBarto2018], [Mnih2015], [vanHasselt2016], [Wang2016], [Mnih2016], [Schulman2017], [Bellemare2017], [Dabney2018].  
Explica de forma breve y clara: estado, acción, recompensa, política, valor Q, DQN, Double DQN, Dueling DQN, PPO, familia actor-critic, y RL distribucional. Presenta QRDQN como una variante distribucional basada en regresión por cuantiles.

**RL/DRL para NIDS y ciberdefensa.**  
Usa: [Alavizadeh2021], [Strickland2023], [Tellache2024], [Yang2024_weak], [Gueriani2024_weak], [Palmer2023_weak].  
Distingue dos líneas:  
a) trabajos que reformulan la detección/clasificación IDS como problema RL sobre datasets etiquetados;  
b) trabajos de cyber defence autónoma más amplia.  
Debes decir que el TFG se parece más a la línea (a) que a la (b). Debes señalar que varios trabajos RL para IDS se parecen metodológicamente a reclasificación supervisada con reward engineering más que a control secuencial rico.

**Limitaciones metodológicas del área.**  
Usa: [Arp2020], [Layeghy2021], [Layeghy2022], [Cantone2024], [Ring2019].  
Debes ser crítico y mencionar: inflación de accuracy, splits aleatorios demasiado favorables, leakage mediante campos espurios, class imbalance, falta de validación externa, reproducibilidad limitada y mala generalización cross-dataset. Esta sección debe ser una de las más fuertes del capítulo.

**Posicionamiento del TFG.**  
Conecta la literatura con la implementación real del proyecto. Usa la información del repositorio: el prototipo trabaja con un esquema canónico fijo de 76 features flow-based y una observación de 152 dimensiones al añadir máscara de missingness; el entorno formula `0 = PERMIT` y `1 = BLOCK`; la fase 1 es entrenamiento/validación offline sobre datasets públicos; la fase 2 es inferencia offline sobre tráfico propio de laboratorio; no hay bloqueo activo en tiempo real; el algoritmo principal es QRDQN. fileciteturn4file0L3-L3turn5file0L3-L3turn7file0L3-L3turn9file0L3-L3  
Debes presentar el valor del TFG así: una formulación concreta y reproducible de defensor RL binario PERMIT/BLOCK sobre flujos, con benchmark interno en CICIDS2017 y validación externa offline sobre tráfico privado de laboratorio, bajo una lectura metodológica prudente.

**Afirmaciones seguras permitidas.**  
Puedes afirmar que:  
- los benchmarks públicos son necesarios, aunque insuficientes para demostrar despliegue real;  
- los enfoques flow-based son habituales y pragmáticos;  
- sí existen precedentes de RL para IDS;  
- el RL distribucional tipo QRDQN parece poco representado en la muestra revisada;  
- la aportación del TFG está más en la formulación experimental y la evaluación prudente que en reclamar una novedad absoluta.

**Afirmaciones prohibidas.**  
No escribas que:  
- el TFG es “el primero” en RL para NIDS;  
- QRDQN es el estándar del área;  
- el rendimiento en CICIDS2017 demuestra eficacia real;  
- la fase de laboratorio equivale a despliegue operativo;  
- el entorno dataset-as-environment reproduce fielmente una red real.

**Tono y estilo.**  
El tono debe ser sobrio, analítico y escéptico. Cuando una fuente sea preprint o la evidencia sea más débil, indícalo con lenguaje prudente. No uses marketing académico ni afirmaciones grandilocuentes. Da prioridad a precisión terminológica, alcance bien delimitado y transición clara hacia el diseño experimental del TFG.

### Preguntas abiertas y limitaciones de este dossier

La principal limitación documental de esta revisión es que **varias fuentes recientes sobre RL para NIDS localizadas en la exploración aparecen como preprints en arXiv**; por tanto, en la memoria final deberán etiquetarse explícitamente como evidencia de fuerza moderada o débil, no como consenso consolidado. Además, **no he verificado un DOI definitivo para todas las referencias recientes ni un paper canónico único para CSE-CIC-IDS2018 equivalente al de CICIDS2017**, por lo que conviene que la bibliografía final use con precisión las páginas oficiales cuando sea necesario. citeturn15academia1turn15academia0turn16academia3turn20view0turn22view2

Finalmente, el propio repositorio del TFG contiene una **pequeña tensión entre documentación y código** sobre la configuración exacta de recompensas en algunos artefactos e invariantes; eso no invalida el proyecto, pero sí refuerza la conveniencia de ligar toda afirmación experimental a una versión, script y `RUN_ID` concretos. fileciteturn5file0L3-L3turn7file0L3-L3