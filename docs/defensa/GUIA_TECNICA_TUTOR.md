# Guía técnica del TFG — guion para explicárselo al tutor

Documento de trabajo para preparar la sesión con el tutor. Tiene tres partes:

- **Parte 1 — El guion**: la explicación seguida del proyecto, de principio a fin, para leerse o contarse en 20–30 minutos. Cada término técnico lleva una traducción *«en cristiano»*.
- **Parte 2 — Banco de respuestas**: respuestas preparadas a las 50 preguntas del tutor más las 10 «estrella», cada una con una nota de qué comprueba y qué conviene no decir.
- **Apéndice — Glosario**: definiciones llanas, con analogía, de los términos técnicos que aparecen.

Todos los números (hiperparámetros, métricas, dimensiones, hashes) están verificados contra el código y los artefactos del repositorio.

---

## Parte 1 · El guion (20–30 min)

### Qué hago con los datos: de los CSV originales al vector que ve el agente

Empiezo por el dataset base, **CICIDS2017**, en su formato oficial: 8 ficheros CSV exportados por una herramienta llamada CICFlowMeter. Los cargo siempre en un orden fijo y determinista: Monday, Tuesday, Wednesday, Thursday (con 2 ficheros, WebAttacks e Infilteration) y Friday (con 3, Morning, PortScan y DDos). Por defecto el cargador solo admite esos 8 ficheros canónicos (`allow_non_official_csvs=False`); si falta cualquiera, lanza un error `FileNotFoundError`. Hago esto para garantizar que el conjunto de datos sea siempre exactamente el mismo. Cada CSV se lee en bloques de 250.000 filas (`chunksize`) con `low_memory=True` para no agotar la memoria RAM.

Aquí conviene explicar qué es un **flujo de red**, porque es la unidad básica de todo el trabajo. Un flujo es el resumen estadístico de una conversación entre dos máquinas: agrupa todos los paquetes que van y vienen entre un origen y un destino durante una sesión, y los condensa en números como la duración, cuántos bytes se enviaron en cada sentido, cada cuánto llegaban los paquetes o qué banderas TCP se activaron. *En cristiano: en lugar de mirar paquete a paquete, miro la "ficha resumen" de toda la conversación de red.* Trabajo con flujos en vez de inspeccionar el contenido de cada paquete porque los flujos son ligeros, no requieren leer datos que pueden ir cifrados, y describen el comportamiento (el patrón de la conexión) en lugar del contenido.

Sobre esos CSV aplico un **esquema canónico** congelado de 76 características numéricas de estadísticas de flujo (`FEATURES_CANON` en `src/canonical_schema.py`), cada una con su nombre en formato `lower_snake_case`. *En cristiano: un esquema canónico es una lista fija y estandarizada de las 76 medidas que siempre uso, con nombres unificados, da igual de dónde venga el dato.* El criterio para elegir esas 76 es deliberado: una característica entra solo si (1) existe en CICIDS2017, (2) se puede extraer de tráfico PCAP real mediante extractores de flujo como CICFlowMeter o Zeek, (3) no es un identificador ni un proxy de la etiqueta, (4) es una estadística numérica de flujo y (5) es estable y robusta, no una rareza del dataset. Las categorías que cubren son: estadísticas generales de flujo, longitudes de paquete hacia delante (forward) y hacia atrás (backward), tasas de flujo, tiempos entre llegadas de paquetes (los IAT, de flujo, forward y backward), banderas TCP (SYN, ACK, FIN, RST, PSH, URG, ECE, CWE), longitudes de cabecera, paquetes por segundo, estadísticas de longitud de paquete, ratios, estadísticas de bulk y de subflujo, ventana TCP y tiempos de actividad/inactividad. El mapeo de columnas originales a canónicas (`CICIDS2017_TO_CANON`) tiene exactamente 76 entradas, una por característica.

El vector de observación que ve el agente no tiene 76 dimensiones, sino **152** (`NUM_OBSERVATION_FEATURES = 76*2`). Son las 76 características (ya imputadas, es decir, con los huecos rellenados) concatenadas con una **máscara de ausencia** ("missingness") de otras 76 dimensiones: `obs = [x_1..x_76, m_1..m_76]`. La máscara funciona celda por celda, no por columna: `m_i = 1` si la característica estaba presente Y era un número finito en la fila de origen; `m_i = 0` si estaba ausente, era NaN o Inf y por tanto la rellené con `0.0` (`DEFAULT_IMPUTATION_VALUE`). *En cristiano: la máscara es una segunda lista de 76 unos y ceros que le dice al modelo "este dato es real" (1) o "este dato me lo he inventado porque no lo tenía" (0).*

Este punto lo tengo que poder defender, porque es sutil: **sobre CICIDS2017 nativo, el mapeo cubre las 76 características**, así que la máscara vale constantemente 1 en todas las filas y no aporta nada al resultado de CICIDS2017. La máscara codifica la presencia de la columna de origen, no del valor concreto. Solo se vuelve informativa en Fase 2 (tráfico de laboratorio) o con un dataset como NSL-KDD, cuyo mapeo cubre únicamente 3 de las 76 características canónicas (`flow_duration`, `total_length_of_fwd_packets` y `total_length_of_bwd_packets`, que vienen de las columnas `duration`, `src_bytes` y `dst_bytes` de NSL-KDD), dejando la máscara casi toda a 0. *En cristiano: en CICIDS2017 la máscara no sirve de nada porque siempre tengo todos los datos; solo empieza a tener valor cuando le doy al modelo tráfico al que le faltan columnas.*

#### Etiquetas, anti-fuga y limpieza

Las etiquetas las binarizo: 0 = BENIGN, 1 = ATTACK. La regla es simple: `(etiqueta en mayúsculas y sin espacios != 'BENIGN')`, así que todo lo que no sea BENIGN cuenta como ataque.

Aplico una **política anti-fuga** (anti-leakage) que descarta Flow ID, Timestamp, todas las columnas de IP y los puertos de origen y destino. *En cristiano: fuga de información es cuando una columna le "chiva" al modelo la respuesta de una forma que no funcionaría en el mundo real.* El caso más claro aquí es el **puerto de destino**: lo elimino a propósito porque ciertos ataques usan puertos específicos, y entonces el puerto actuaría como un proxy de la etiqueta. Eso no es solo quitar identificadores: es prevenir fuga. El descarte usa un conjunto exacto de nombres más reglas de subcadena (`' ip'`, terminar en `'ip'`, contener `'flow id'` o `'timestamp'`).

La limpieza convierte los `inf` en NaN, rellena los NaN de las características con 0 (la ausencia de valor en contadores y tasas de flujo significa ausencia de actividad, y la máscara complementa esta decisión) y descarta las filas cuya etiqueta sea NaN. Hay dos pasadas de imputación en etapas distintas: `_clean_rows` rellena antes del mapeo canónico, y `map_to_canonical` vuelve a sustituir valores no finitos por la imputación. Lo que realmente registra la fiabilidad del dato es la máscara.

#### Escalado y partición

El escalado lo hago con un `StandardScaler` de sklearn ajustado SOLO sobre el train y aplicado al test. *En cristiano: normalización o z-score significa restar la media y dividir por la desviación típica de cada característica, para que todas queden en una escala comparable (centradas en 0, con dispersión 1) y ninguna domine por tener números más grandes.* Lo ajusto solo con el train porque, si lo ajustara con todos los datos, el conjunto de prueba estaría influyendo en la transformación y eso sería una forma de fuga: el modelo habría "visto" indirectamente el test. Por eso calculo media y desviación con el train y luego aplico esa misma transformación al test, sin reajustar.

El split por defecto es una **partición estratificada** aleatoria 80/20 (`test_size=0.2`, `random_state=42`). *En cristiano: estratificada quiere decir que reparto las filas al azar pero manteniendo en train y en test la misma proporción de benignos y de ataques que había en el total; así ninguna de las dos partes queda sesgada hacia una clase.* Existe también un split alternativo por día/CSV, que es un group split real con rechazo de solape (train y test no pueden compartir ningún CSV) y que no usa `test_size` ni `random_state`.

La partición de test fija es el split aleatorio con seed 42 y preset `full`, verificado mediante hashes SHA-256 del contenido. El script `verify_fixed_test_split.py` reproduce el split y comprueba que coincide con el run oficial: `n_train=2.264.594`, `n_test=566.149`, `test_benign=454.620`, `test_attack=111.529`. El hash usa el prefijo `'dtype|shape|'` más los bytes contiguos, de modo que float32 frente a float64, o un array reformado, no colisionan. El `test_set_sha256` de referencia es `cb175377f462672b3aeb3ad3dd8bff3ab506d8663755cb336d40f3ef9765035b`.

Hay además un mecanismo de benchmark, `train_max_rows`, que submuestrea SOLO la partición de train después del split, con prefijos anidados estratificados deterministas (500k ⊂ 1M ⊂ full), dejando el test idéntico byte a byte. Cuando se usa, el escalado se difiere y el scaler se ajusta sobre el train submuestreado, no sobre el train completo. Y existe un análisis de duplicados de solo lectura (`analyze_duplicates.py`) que cuantifica duplicados exactos y fuga entre splits sin entrenar nada, comparando filas a nivel de bytes.

### Cómo preparo el entorno: RunPod, CUDA, Python y dependencias

El entrenamiento real corre en una caja GPU de RunPod, preferentemente una RTX 3090 Ti con 24 GB de VRAM (la L40S es opcional, y una A100 sería innecesaria), sobre Linux x86_64. Python está fijado a `>=3.12,<3.13`; el run principal corrió con la versión 3.12.11. El entorno es un `python -m venv venv` sencillo más `pip install -r requirements-runpod-cu130.txt`.

Hay dos ficheros de requisitos, por diseño. El de RunPod fija `torch==2.12.1+cu130` desde el índice de wheels de PyTorch para CUDA 13.0 (`--extra-index-url https://download.pytorch.org/whl/cu130`): es el conjunto exacto y pinneado para reproducir en GPU, y omite a propósito los extras de desarrollo (pytest, ruff) y de tuning (optuna). El `requirements.txt` genérico fija el mismo stack pero con `torch==2.12.1` sin el sufijo `+cu130`, para que pip elija la wheel de plataforma o de CPU en desarrollo local. El `pyproject.toml` declara las mismas dependencias y usa `[tool.uv.sources]` para enrutar torch al índice cu130 en Linux y al de CPU fuera de Linux, de modo que `uv sync` resuelve bien sin el flag `--index-strategy unsafe-best-match` (ese flag solo hace falta para un `uv pip install -r requirements-runpod-cu130.txt` crudo).

El stack ML pinneado es: torch 2.12.1+cu130 (reporta CUDA 13.0), stable-baselines3 2.8.0, sb3-contrib 2.8.0 (que es quien aporta QR-DQN), numpy 2.4.6, pandas 3.0.3, scikit-learn 1.9.0, gymnasium 1.2.3, joblib 1.5.3, y como extras de entrenamiento tensorboard 2.20.0 y matplotlib 3.10.9. Optuna 4.9.0 va en el extra de tuning. El procedimiento de arranque es: clonar, `cd`, `python -m venv venv`, activar, `pip install -U pip`, instalar los requisitos y un print de comprobación de que torch ve CUDA. El device se autodetecta: `'cuda' if torch.cuda.is_available() else 'cpu'`.

El arranque del entrenamiento siembra todos los generadores de números aleatorios (RNG) desde `--seed` (por defecto 42): `random.seed`, `np.random.seed`, `torch.manual_seed` y `torch.cuda.manual_seed_all` si hay CUDA. Esto fija la inicialización de la red, el muestreo del replay buffer y la exploración. El split de datos usa su propio `random_state` interno, así que la siembra global NO altera la partición fija seed-42; ese punto está comentado explícitamente en el código. Además se hace `vec_env.seed(seed)` y se pasa `seed=seed` al constructor de QRDQN.

### Qué es QR-DQN y qué hace por dentro

Antes de entrar en QR-DQN conviene fijar algunos conceptos de aprendizaje por refuerzo, porque aparecen una y otra vez.

**QR-DQN** (Quantile Regression DQN) es una extensión "distribucional" de DQN. Un DQN clásico aprende, para cada par estado-acción, un único número: el Q-valor, que es la media (la esperanza) de las recompensas futuras que cabe esperar si tomo esa acción. QR-DQN, en cambio, no aprende solo la media: aprende la **distribución completa** de retornos, representada como un conjunto fijo de N **cuantiles** (`n_quantiles`, que en este proyecto son 200). *En cristiano: en lugar de aprender "de media me darán tanto", aprende la forma entera de lo que puede pasar; los cuantiles son 200 puntos que dibujan esa distribución, como 200 marcas que reparten en partes iguales todos los resultados posibles.* RL distribucional es exactamente eso: aprender el reparto de resultados, no solo su promedio.

La red (`QuantileNetwork`) es un extractor de características tipo MLP más una cabeza que, para una observación, produce `action_dim * n_quantiles` valores, reformados a `(batch, n_quantiles, n_actions)`. Cada cuantil es un átomo equiprobable, de probabilidad `1/N`, que aproxima la distribución de retornos de esa acción. El DQN clásico solo aprende `E[retorno]`; QR-DQN aprende la función cuantil completa y recupera la media solo cuando la necesita para seleccionar la acción.

QR-DQN es un algoritmo **off-policy**. *En cristiano: off-policy significa que el agente puede aprender de experiencias que no generó con su política actual; guarda lo que le pasó y lo reutiliza más tarde, en vez de tener que aprender solo de lo que hace justo ahora.* Las transiciones se guardan en un **replay buffer** y de él se muestrean minibatches. *En cristiano: el replay buffer es una memoria donde se apuntan las experiencias pasadas (estado, acción, recompensa); un minibatch es un puñado de esas experiencias elegidas al azar, y muestrear minibatches es coger ese puñado para entrenar con él.* Se hace así para decorrelar las actualizaciones, es decir, para que el modelo no entrene con ejemplos demasiado parecidos y seguidos.

La exploración es **epsilon-greedy** con un schedule lineal desde `exploration_initial_eps=1.0` hasta `exploration_final_eps`. *En cristiano: epsilon-greedy es el equilibrio entre explorar y explotar; con probabilidad epsilon el agente prueba una acción al azar (explora, para descubrir cosas nuevas) y el resto del tiempo elige la que cree mejor (explota lo que ya sabe). Empieza explorando mucho (epsilon = 1.0, todo al azar) y va bajando hasta casi siempre explotar.*

Hay una **red objetivo** separada (`quantile_net_target`) que se actualiza por **copia dura** (`tau=1.0`) cada `target_update_interval` pasos de entorno, para estabilizar el objetivo de aprendizaje. *En cristiano: la red objetivo es una copia "congelada" de la red que se usa como referencia fija al calcular hacia dónde corregir; si usara la misma red que estoy moviendo, el objetivo se movería conmigo y el entrenamiento se volvería inestable. Actualización dura quiere decir que cada cierto número de pasos copio entera la red actual sobre la objetivo, de golpe, en lugar de mezclarlas poco a poco.*

La acción greedy (tanto para actuar como para elegir la siguiente acción en el target) se obtiene colapsando la dimensión de cuantiles a un Q-valor medio por acción y tomando el argmax: `self(obs).mean(dim=1).argmax(dim=1)`. Es decir: la distribución se usa para aprender y para representar el riesgo, pero la selección de acción se hace por la esperanza. La acción siguiente en el target la elige la propia red objetivo por su Q medio, no una red online separada: esto es **bootstrapping** de una sola red, NO el desacople estilo Double-DQN. *En cristiano: bootstrapping es estimar el valor de ahora apoyándose en mi propia estimación del valor del paso siguiente; me "tiro de mis propios cordones" usando una predicción para mejorar otra.*

#### El paso de gradiente, paso a paso

En secuencia: (1) se muestrea un minibatch del replay buffer; (2) bajo `no_grad` se calculan los cuantiles de la red objetivo para `next_obs`, se elige la siguiente acción greedy por argmax del Q medio, se recogen sus cuantiles y se forma el target `r + (1-done)*discount*next_quantiles`, sumando la recompensa escalar elemento a elemento sobre toda la distribución; (3) se calculan los cuantiles de la red online para `obs` y se recogen los de la acción tomada según el buffer, dando forma `(batch, n_quantiles)`; (4) se calcula la pérdida `quantile_huber_loss(current, target, sum_over_quantiles=True)`; (5) `zero_grad`, `backward`, recorte opcional de gradiente si `max_grad_norm` no es None, y `step`. Esto se repite `gradient_steps` veces.

La pérdida es la **quantile-Huber loss**. Construye `pairwise_delta = target - current` para todos los pares de cuantiles, aplica la Huber loss con un umbral kappa fijo a 1.0 (cuadrática si `|delta|<=1`, lineal `|delta|-0.5` más allá) y la pondera de forma asimétrica con `|cum_prob - (delta<0)|`, donde `cum_prob` son las fracciones cuantil de punto medio `(i+0.5)/N`. El indicador usa `delta.detach()`, así que el peso de asimetría no lleva gradiente: el gradiente fluye solo por la Huber loss. Con `sum_over_quantiles=True` se suma sobre el eje de cuantiles actuales y luego se promedia, como en el paper. El umbral kappa no es un parámetro del constructor: está hard-codeado a 1.0 y no se puede tunear por la API.

### Cómo funciona el RL en este proyecto: entorno, observación, acción y recompensa

El entorno es `RLDatasetDefenderEnv`, un `gym.Env` de Gymnasium que recasta la clasificación binaria supervisada (benigno frente a ataque) como un MDP para un agente "defensor" de red. El agente observa el vector de características de un único flujo etiquetado y emite una de dos acciones: 0 = PERMIT (dejar pasar el tráfico) o 1 = BLOCK (bloquearlo).

El espacio de observación es `Box(low=-inf, high=+inf, shape=(n_features,), dtype=float32)`, con `n_features` tomado de `X.shape[1]` en construcción, que es 152 en entrenamiento. Las cotas son infinitas, lo apropiado para características estandarizadas; el entorno no fuerza el escalado, esa es responsabilidad del código de aguas arriba. El espacio de acción es `Discrete(2)`; `step()` rechaza con `ValueError` cualquier acción fuera del espacio.

#### La recompensa y los costes asimétricos

La recompensa es función pura de (etiqueta verdadera, acción) e implementa una **matriz de coste de confusión asimétrica**. *En cristiano: una matriz de confusión es la tabla de cuatro casillas que cruza lo que era de verdad con lo que el modelo decidió: aciertos en ataque, aciertos en benigno, y los dos tipos de error.* Los cuatro casos aquí son:

- **Verdadero positivo** (ataque bloqueado) = **+1.5**
- **Falso positivo** (benigno bloqueado, una falsa alarma) = **−2.0**
- **Falso negativo** (ataque permitido, un ataque que se nos cuela) = **−5.0**
- Término de "omission", el equivalente al verdadero negativo (benigno permitido) = **0.0**

No existe una clave `tn`; el verdadero negativo se llama `omission` y vale 0.0 por defecto. La lógica: si es ataque y se hace BLOCK → tp; si es ataque y se hace PERMIT → fn; si es benigno y se hace PERMIT → omission; si es benigno y se hace BLOCK → fp. El config de recompensa se fusiona de forma superficial sobre los defaults, así que un override parcial conserva el resto.

Un **falso negativo** aquí es concreto: es un ataque que el modelo deja pasar, un intruso al que se le abre la puerta. Un **falso positivo** es bloquear tráfico legítimo, una falsa alarma que molesta al usuario legítimo. En seguridad, dejar pasar un ataque suele ser mucho más grave que una falsa alarma, y eso es justo lo que codifica la recompensa: el falso negativo (−5.0) es el penalizador de mayor magnitud, 2.5 veces el del falso positivo (−2.0).

Importante: **el desbalanceo de clases NO se trata con pesado por muestra, sobremuestreo ni reescalado por frecuencia** dentro del entorno. Se trata implícitamente, de forma económica, con las magnitudes asimétricas de recompensa. Eso codifica que dejar pasar un ataque es 2.5 veces peor que una falsa alarma, y empuja al agente hacia mayor recall en la clase de ataque, que es la rara. *En cristiano: en vez de duplicar artificialmente los ejemplos de ataque, hago que equivocarse en un ataque "duela" más, y así el agente aprende a no perdérselos.* El riesgo de penalizar demasiado los falsos negativos sería el contrario: que el agente bloquee de más y dispare los falsos positivos; por eso hay que mirar las dos cosas, recall y precisión, no solo una.

#### El punto clave: esto es un bandit contextual

El punto conceptual que debo poder defender: aunque el entorno presenta toda la interfaz episódica de Gymnasium (`reset`/`step`/`terminated`/`truncated`), es efectivamente un **bandit contextual**, un MDP de un solo paso por contexto. *En cristiano: un bandit contextual es como una máquina tragaperras a la que primero le enseñan una pista (el contexto, aquí el flujo) y tú eliges una palanca (permit o block); recibes premio o castigo al instante y la siguiente jugada no depende de la anterior. No hay "partida" que se construya paso a paso.*

No hay transiciones reales: la siguiente observación es una muestra extraída de forma independiente, completamente no correlacionada con la acción anterior; es simplemente la siguiente entrada de un array de índices permutado. La recompensa depende solo del (contexto, acción) actual, nunca del historial. Por tanto el agente no gana nada con la asignación temporal de crédito, y el **factor de descuento gamma** es irrelevante para la política óptima. *En cristiano: gamma es cuánto valora el agente las recompensas futuras frente a las inmediatas; un gamma cercano a 1 dice "el futuro importa casi tanto como el presente" y un gamma de 0 dice "solo me importa el premio de ahora mismo". Como aquí no hay futuro que dependa de mí, gamma da igual.*

El muestreo es **sin reemplazo** dentro de un episodio. `reset()` (si `shuffle=True`, que es el default) permuta in-place el array de índices usando el RNG de Gymnasium sembrado por `super().reset(seed=seed)`, y `step()` avanza un cursor que recorre esa permutación, viendo cada muestra exactamente una vez. No es muestreo i.i.d. por paso ni una muestra por episodio: es un episodio multipaso que consume secuencialmente el dataset permutado. `reset()` devuelve `(obs, {})` con info vacío; `step()` devuelve la 5-tupla `(obs, reward, terminated, truncated, info)`, con `info` conteniendo `sample_index` y `true_label` de la muestra sobre la que se acaba de actuar. La observación devuelta es la siguiente muestra cuando el episodio continúa. `terminated` se dispara cuando `current_idx >= n_samples` (se consumió el dataset) y `truncated` cuando `steps >= max_steps_per_episode`. En el paso terminal la observación devuelta es la última muestra sobre la que se actuó, para no indexar fuera del array. El entorno valida en construcción que X sea 2D, que y sea 1D y de longitud coherente, y que todas las etiquetas sean estrictamente binarias, fallando rápido si no es así.

### El proceso de entrenamiento de principio a fin

El orquestador es `src/train_rl_defender.py`, que entrena QR-DQN (`sb3_contrib.QRDQN`) sobre CICIDS2017 encuadrado como bandit contextual. El flujo completo es:

1. **Siembra de RNG** desde `--seed` (42), como ya he descrito.
2. **Resolución de preset/perfil**, hiperparámetros y timesteps. Hay dos presets (`fast`, `full`) y dos perfiles de entrenamiento (`default`, `main-experiment`). `--smoke` es solo un alias de `--preset fast`.
3. **Carga de datos** vía `load_cicids2017_split(scale=False)`. El escalado se gestiona dentro del script, no en el cargador: el scaler que devuelve el cargador se descarta. Se calculan los percentiles p0.5 y p99.5 sobre las primeras 76 columnas (`_N_CANON`) del `X_train` SIN escalar (en unidades de característica crudas, antes de ajustar el scaler), se ajusta un `StandardScaler` fresco sobre `X_train`, se transforman ambas particiones a float32 y se persiste el scaler con `joblib.dump`. Así el scaler guardado coincide exactamente con los datos que ven el entorno y el modelo.
4. **Construcción del entorno**: un `DummyVecEnv` (un solo entorno, no paralelo) que envuelve un `Monitor(RLDatasetDefenderEnv)` con `shuffle=True` y `max_steps_per_episode = min(10_000, len(X_train))`.
5. **Instanciación de QRDQN** con los hiperparámetros resueltos, el seed, el device y el `tensorboard_log`.
6. **`model.learn(total_timesteps, callback=checkpoint_callback, tb_log_name=RUN_ID, reset_num_timesteps=False)`**. El callback de checkpoint solo se adjunta si `checkpoint_freq>0`.
7. **Guardado del modelo** en `models/<RUN_ID>.zip` (global) y copia a `runs/cicids2017/<RUN_ID>/model.zip`.
8. **Evaluación determinista** sobre el test en lotes de 8192 (`eval_batch_size`) con `model.predict(..., deterministic=True)`, construyendo la matriz de confusión con `labels=[0,1]` que da `(tn,fp,fn,tp)`.
9. **Persistencia de artefactos**: `config.json` (escrito dos veces, estado 'started' y luego 'completed'), `metrics.json`, `scaler.joblib`, `train_percentiles.npz` (con `p_low` y `p_high`), `feature_names.json`, `environment.json` y `artifact_manifest.json` (schema 2.0, con checksums SHA-256 y rutas relativas añadidos al completar).

El `RUN_ID` codifica el run: `{prefix}_{algo}_cicids2017_{canon}_{exp_tag}_{timestamp}`, donde el prefijo es `MAIN` para `main-experiment` y `C03` en otro caso, el algo es `qrdqn`, y `exp_tag` es `{preset}_{split_mode}` con sufijo `_t{train_max_rows}` si aplica.

#### El hecho de diseño central: gamma = 0.0

**Gamma = 0.0 en los tres regímenes de hiperparámetros** (default-fast, default-full y main-experiment). Con gamma 0, el término de bootstrap se anula: `target_quantiles = rewards`, cada target cuantil es igual a la recompensa inmediata escalar, la red objetivo no contribuye al target y no hay asignación temporal de crédito. QR-DQN se convierte en un aprendiz puro de un solo paso, un bandit contextual: cada paso es una decisión independiente permit/block sobre un flujo, y el `reward_config` es toda la señal. La red objetivo y el replay buffer siguen funcionando, pero el bootstrap es un no-op (no hace nada).

Tengo que tener preparada la respuesta a la objeción "entonces esto es solo un clasificador". El encuadre honesto es: no es entropía cruzada sobre etiquetas, es **aprendizaje de valor sobre una matriz de coste asimétrica explícita** (FN −5 ≫ FP −2); la decisión es el argmax del valor aprendido entre PERMIT y BLOCK; y la cabeza distribucional de QR-DQN sigue aportando, porque modela la distribución completa de retornos, no solo la media. Es honestamente una formulación cost-sensitive de un solo paso, un bandit contextual, no un MDP secuencial. No lo voy a vender como una defensa autónoma secuencial, porque no lo es.

#### Tabla compacta de hiperparámetros

| Hiperparámetro | default-fast | default-full | main-experiment |
|---|---|---|---|
| total_timesteps | 25.000 | 100.000 | 3.000.000 |
| net_arch | [512, 256] | [512, 256] | [1024, 1024, 512] |
| n_quantiles | 200 | 200 | 200 |
| learning_rate | 1e-4 | 1e-4 | 5e-5 |
| buffer_size | 25.000 | 100.000 | 1.000.000 |
| learning_starts | 100 | 100 | 50.000 |
| batch_size | 512 | 2048 | 2048 |
| gamma | 0.0 | 0.0 | 0.0 |
| tau | 1.0 | 1.0 | 1.0 |
| train_freq | 50 | 100 | 100 |
| gradient_steps | 10 | 20 | 20 |
| target_update_interval | 1.000 | 10.000 | 10.000 |
| exploration_initial_eps | 1.0 | 1.0 | 1.0 |
| exploration_final_eps | 0.01 | 0.01 | 0.02 |
| exploration_fraction | 0.005 | 0.005 | 0.10 |
| max_grad_norm | None | None | 10.0 |
| checkpoint_freq | 0 | 0 | 250.000 |

`buffer_size` en el perfil default se resuelve como `min(200_000, max(total_timesteps, 10_000))`, de ahí 25.000 en fast y 100.000 en full. La `policy` es siempre `MlpPolicy`.

#### Qué hace cada parámetro y por qué esos valores

- **`gamma=0.0`**: descuento; a 0 colapsa el bootstrap y queda un bandit contextual de coste. Es la decisión que vertebra todo el proyecto.
- **`n_quantiles=200`**: número de átomos cuantil de la cabeza distribucional. Es el valor del paper de QR-DQN.
- **`net_arch`**: las capas ocultas del MLP. El run principal usa una red más profunda y ancha [1024,1024,512] frente a [512,256] del default, porque dispone de 3M de pasos para aprovecharla.
- **`learning_rate`**: el paso de Adam. Bajado a 5e-5 en el run de 3M para dar estabilidad en la corrida larga.
- **`buffer_size`**: capacidad del replay buffer, escalada a la longitud del run y capada a 200k en default; 1M en el run principal.
- **`learning_starts`**: pasos de exploración pura antes de empezar a actualizar. Diminuto (100) en default por ser un bandit; 50.000 en el run principal para llenar el buffer primero.
- **`batch_size`**: tamaño del minibatch por paso de gradiente.
- **`tau=1.0`**: coeficiente de actualización de la red objetivo; 1.0 = copia dura completa cada `target_update_interval`. QR-DQN es de actualización dura por defecto, pese a la maquinaria de soft-update.
- **`train_freq` / `gradient_steps`**: cada cuántos pasos de entorno se entrena, y cuántas actualizaciones de gradiente por disparo.
- **`target_update_interval`**: pasos entre copias duras de la red objetivo.
- **`exploration_*`**: epsilon inicial, final, y la fracción del entrenamiento sobre la que decae linealmente. En default el decaimiento es muy rápido (0.005); en el run principal, el 10%.
- **`max_grad_norm`**: recorte de norma de gradiente; activo (10.0) solo en el run largo, para estabilidad.

### Tuning con Optuna y una inconsistencia que debo reconocer

El tuning de hiperparámetros vive en un script SEPARADO, `src/tune_hparams.py`, que es un harness de búsqueda ligero, no un run reproducible completo. Carga datos con `load_cicids2017_binary` (`max_rows=50_000`, `scale=True`), crea un estudio `direction='maximize'`, entrena un QRDQN por trial (10.000 timesteps por defecto, 10 trials) y devuelve el F1 de la clase de ataque. No persiste modelos ni scaler por trial, solo el resumen del estudio en JSON. No hay pruner efectivo: no se pasa pruner ni sampler y el objetivo nunca llama a `trial.report()`/`should_prune()`, así que todos los trials corren hasta el final.

El espacio de búsqueda: `learning_rate` log-uniforme en [1e-5, 1e-3]; `batch_size` en {256,512,1024,2048}; `gradient_steps` en {10,50,100}; `net_arch` en {[256,128],[512,256],[256,256]}; **`gamma` en [0.95, 0.999]**; `train_freq` en {50,100,200}.

Aquí está la inconsistencia real que tengo que reconocer abiertamente: el tuner busca gamma en régimen de RL secuencial (entre 0.95 y 0.999), mientras que el entrenador de producción fija gamma=0.0 (bandit). Son encuadres conceptualmente distintos, así que los gamma tuneados no son transferibles a la config principal. Además, los hiperparámetros del run principal son un perfil fijo puesto a mano y congelado por tests (`test_main_experiment_profile_resolves_fixed_config`), NO una salida de Optuna; de hecho el espacio de búsqueda de Optuna excluye explícitamente los valores del run principal (gamma 0.95–0.999 frente a 0.0, y net_arch [1024,1024,512] no está en el espacio), por lo que el run principal no pudo salir de esa búsqueda.

### Validación, métricas y baseline

El subsistema de validación existe para demostrar que las métricas de QRDQN son genuinas y no producto de fuga de etiquetas, de un bug en el entorno ni de un seed afortunado.

`src/metrics_utils.py` es la única fuente de verdad: `confusion_to_metrics(tn,fp,fn,tp)` deriva de forma determinista accuracy, balanced_accuracy, MCC, precision/recall/F1 por clase, especificidad, FPR, FNR, block_rate y recompensa opcional, con división segura (`_safe_div` devuelve 0.0 si el denominador es 0, nunca NaN ni crash). La convención es 0 = PERMIT/benigno/negativo, 1 = BLOCK/ataque/positivo, y el orden de la matriz aplanada es `(tn, fp, fn, tp)`. La clase de ataque (1 = BLOCK) es la positiva, y los informes enfatizan precision/recall/F1 de ataque.

Conviene aclarar estas métricas en lenguaje llano:

- **Precisión** (precision): de todo lo que el modelo marcó como ataque, qué fracción era ataque de verdad. *En cristiano: cuando salta la alarma, ¿cuántas veces tiene razón?*
- **Recall** (sensibilidad): de todos los ataques reales, qué fracción detectó. *En cristiano: ¿cuántos de los ataques que había de verdad consiguió pillar?*
- **F1**: la media armónica de precisión y recall, un solo número que las equilibra. *En cristiano: una nota combinada que castiga si cualquiera de las dos es baja.*
- **MCC** (coeficiente de correlación de Matthews): una métrica que resume las cuatro casillas de la matriz de confusión en un número entre −1 y +1, y que es honesta incluso con clases desbalanceadas. *En cristiano: una "nota global" robusta que no se deja engañar porque haya muchos más benignos que ataques; 1 es perfecto, 0 es azar.*

Si el objetivo es evitar ataques no detectados, la métrica que miro primero es el recall de ataque (o, equivalentemente, el FNR, los ataques perdidos).

`src/validate_checks.py` implementa tres comprobaciones, todas con QRDQN en gamma=0.0 (bandit):

- **Check A — evaluación directa**: recorre `X_test`, llama a `model.predict(obs, deterministic=True)` y construye la matriz de confusión contra `y_test`, deliberadamente sin consultar el `info['true_label']` del entorno. Prueba que las métricas son modelo-frente-a-verdad real, no un artefacto del bookkeeping de etiquetas del entorno. Solo corre si el modelo pasa la verificación de integridad de artefactos (hash del manifest) vía `resolve_trusted_artifact`; si no, se omite silenciosamente salvo `--allow-unsafe-artifacts`.
- **Check B — anti-fuga con etiquetas barajadas**: copia `y_train`, lo baraja con `np.random.default_rng(seed=42)`, entrena un QRDQN fresco sobre las etiquetas barajadas (el default del CLI son 10.000 timesteps; el artefacto commiteado que cito más abajo se generó con 2.000) y evalúa sobre el test real. El baseline esperado es la clase mayoritaria del TEST. El umbral de fuga es `baseline_acc + 0.05`. PASS (sin fuga) significa que la accuracy con etiquetas barajadas SE QUEDA baja, cerca del baseline; un valor ALTO con etiquetas aleatorias es la señal de FALLO/fuga. Hay que tenerlo muy claro: `leakage_detected=True` es el resultado malo. *En cristiano: si rompo la relación entre datos y etiquetas y el modelo sigue acertando, es que estaba copiando algo que no debía; quiero que con etiquetas al azar fracase.*
- **Check C — split duro por día/CSV**: entrena con Monday/Tuesday/Wednesday y testea con Thursday/Friday. El test contiene días y ataques no vistos en entrenamiento; es la prueba de generalización más realista, y evita la inflación por patrones duplicados de los splits aleatorios.

`src/validate_leave_one_csv_out.py` implementa la validación cruzada leave-one-CSV-out completa: un fold por CSV real (8 disponibles), reservando exactamente un CSV como test y entrenando en los siete restantes 30.000 timesteps por fold, agregando de dos formas distintas (media/desv/min/max por métrica entre folds, y métricas globales desde la matriz de confusión agrupada, sumando tp/fp/fn/tn entre folds). Son agregaciones distintas y no hay que confundirlas. Esto está implementado como capacidad del pipeline; lo que queda por hacer es generar y commitear el artefacto de la batería completa, que aún no existe en el repo, así que se presenta como capacidad técnica, nunca con métricas.

`scripts/bootstrap_ci.py` pone un **intervalo de confianza bootstrap** percentil del 95% alrededor de las métricas del run principal SIN reentrenar. *En cristiano: un intervalo de confianza bootstrap es un margen de error que se obtiene "remuestreando" los datos de test miles de veces y mirando cuánto baila cada métrica; dice "el valor verdadero está casi seguro dentro de este rango".* Recupera las celdas exactas `(tn,fp,fn,tp)` desde los recalls a precisión completa multiplicados por los totales enteros de clase, las autovalida a 1e-9 contra cada métrica publicada y resamplea dentro de clase (binomial estratificado por defecto, o multinomial con `--unstratified`). Funciona sin reentrenar porque toda métrica es función de las 4 celdas. La recuperación solo es exacta a precisión float64 completa: con los valores redondeados de tabla, `round(0.99536*111529)=111012` frente al verdadero 111011, un off-by-one. Defaults: `n_boot=10000`, `boot_seed=12345`, `ci_level=0.95`, percentiles [2.5, 97.5].

El baseline supervisado es `src/baseline_random_forest.py`: un `RandomForestClassifier` con `n_estimators=200`, `max_depth=None`, `n_jobs=-1`, `class_weight='balanced'`, `random_state=42`. Usa la misma observación canónica y las mismas características escaladas que QRDQN (RF es invariante a la escala, pero el preprocesado se mantiene idéntico para que la comparación sea justa), `class_weight='balanced'` replica el tratamiento sensible al coste, y las métricas se calculan con el mismo `confusion_to_metrics`. Random Forest es un baseline razonable porque es un clasificador supervisado fuerte y estándar: si un método de RL no le gana bajo el mismo protocolo, hay que ser honesto sobre el valor añadido del RL. Corre bajo tres sweeps: split aleatorio full (igual que el run principal), split por día (igual que Check C) y leave-one-out con Wednesday reservado.

### Los números oficiales del run principal

El resultado oficial del TFG es el run `MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655`: QRDQN, MlpPolicy, perfil main-experiment, CICIDS2017 completo, split aleatorio estratificado, seed 42, device cuda en RTX 3090 Ti. Empezó el 2026-06-09T19:36:55 y completó el 2026-06-10T00:51:39, unas 5h15m, con 3.000.000 de timesteps y gamma 0.0.

- `train_shape` = [2.264.594, 152]; `test_shape` = [566.149, 152]. Aviso importante: el train son 2.264.594 filas, NO 2.830.743; ese 2.830.743 es el total de filas cargadas (train + test). Tengo que citar train = 2.264.594 y test = 566.149 explícitamente para evitar la ambigüedad.
- Balance del split: train benigno 1.818.477 / ataque 446.117; test benigno 454.620 / ataque 111.529. `test_benign_rate` 0.8030, `test_attack_rate` 0.19700. Estratificado: las tasas de train y test coinciden a 4 decimales.
- Métricas oficiales del test (566.149 filas): accuracy **0.99381**, recall de ataque **0.99536**, precision de ataque **0.97378**, F1 de ataque **0.98445**; en benigno: precision 0.998854, recall 0.993425, F1 0.996132.
- Matriz de confusión: **tn=451.631, fp=2.989, fn=518, tp=111.011**, total 566.149. Reverificada de extremo a extremo reejecutando el modelo guardado sobre el split seed-42 reproducido (`--from-model`), regenerando exactamente esas celdas. Derivados: FPR (benigno bloqueado) 0.00658, FNR (ataques perdidos) 0.00465, balanced accuracy 0.99439, MCC 0.98068. Solo se pierden 518 ataques de 111.529, y solo se bloquean mal 2.989 flujos benignos de 454.620. *En cristiano: de cada 1.000 ataques se nos cuelan unos 5, y de cada 1.000 conexiones legítimas bloqueamos por error unas 7.*
- IC del 95% (bootstrap estratificado, `n_boot=10000`, `boot_seed=12345`): recall de ataque 0.99536 con IC [0.99495, 0.99575]; F1 de ataque [0.98394, 0.98496]; accuracy [0.99360, 0.99401]; MCC [0.98004, 0.98130]. Los IC son estrechos (±0.0002–0.001), pero solo cuantifican la precisión del resampling del test para UN modelo entrenado; NO capturan la variabilidad por seed de entrenamiento, porque solo se entrenó un seed. No debo presentar los IC estrechos como evidencia de estabilidad del entrenamiento.
- `test_set_sha256` = `cb175377...035b`; referencia en `runs/cicids2017/test_partition_reference_seed42.json`. La reproducción cross-environment coincide: sklearn 1.8.0 en local da igual que sklearn 1.9.0 de RunPod, y el `mean_`/`scale_` del scaler reproducido coincide con el `scaler.joblib` commiteado del run principal.

Resultados de las comprobaciones de validación (artefactos históricos sobre conjuntos pequeños, ~10k filas, NO el test de 566.149 del run principal): Check A accuracy 0.9939 con TP/FP/FN/TN = 4772/60/1/5167; Check B (2.000 timesteps) accuracy con etiquetas barajadas 0.4773 frente a baseline mayoritario 0.5227, `leakage_detected` false (sin fuga); Check C, split duro por día, accuracy 0.84135, recall de ataque 0.52954, F1 0.62578, TP/FP/FN/TN = 154169/47414/136970/823660, train 1.668.530 / test 1.162.213, 30.000 timesteps. La caída del recall en Check C es esperable: es el escenario de generalización a días y ataques no vistos.

Comparación con Random Forest sobre el mismo split: en split aleatorio (mismo test de 566.149) RF da accuracy 0.99872 y F1 de ataque 0.99676, recall 0.99853, es decir, **RF supera marginalmente a QRDQN** en el split aleatorio. Pero en split por día RF da accuracy 0.76913 y recall de ataque 0.08135, un colapso mucho peor que el recall 0.52954 de QRDQN en Check C; y en leave-one-out (Wednesday) RF da accuracy 0.63782 y F1 0.01427. El leave-one-out es solo de RF; no hay artefacto LOO de QRDQN.

La lectura honesta de esto: que RF gane por poco en el split aleatorio NO basta para decir que QRDQN sea mejor solo por usar RL; lo que se puede afirmar es que, bajo el mismo protocolo, RF se hunde más que QRDQN en el escenario de generalización por día. Y RF tiene ventajas prácticas reales (más rápido de entrenar, más interpretable, menos maquinaria).

Sobre la procedencia: el run principal es histórico frente a las sondas previas al diseño C0x (archivadas, no oficiales, no comparables). C03 da accuracy 0.99859, recall 0.99945 y F1 0.99876, PERO sobre un test de 100.000 filas con la mezcla de clases distorsionada (tasa benigna ~0.434 frente a ~0.803 del run principal): explícitamente NO comparable, así que no debo presentar C03 como el mejor resultado. C01/C02 usaron un fp=−1.0 más suave; el run principal y C03 usan fp=−2.0. No debo generalizar la config de recompensa entre runs.

### Fase 2: inferencia sobre tráfico real del laboratorio y lo que queda por hacer

La Fase 2 es inferencia offline. El punto de entrada mantenido es `scripts/predict_real_traffic_v2.py`. El laboratorio es un entorno cerrado controlado por el operador: el `docker-compose.yaml` define dos servicios en una red bridge `labnet` marcada `internal:true` (los contenedores no alcanzan internet), `web` (nginx:alpine, sin puertos publicados) y `generator` (python:3.12-slim). El generador Docker (`lab/docker/gen_traffic.py`) usa sockets Python crudos para conexiones tipo escaneo a puertos cerrados (ráfagas con RST) y GETs HTTP (`Connection: close`, una conexión TCP nueva por petición), con defaults `--n-http 4000`, `--n-closed 2500`, `--closed-range 30000-30250`, `--jitter-ms 2`. Este generador está DEPRECADO: sus flujos resultaron inutilizables y NO es la fuente del CSV de Fase 2. No debo confundirlos: conflarlos sería un error factual.

El CSV commiteado de Fase 2 (`pcaps/lab_capture_traffic.csv`, ~2M filas de flujos) es captura real de tráfico de un laboratorio doméstico cerrado, generado a mano por consola. Las etiquetas (`truth_label`/`truth_y`/`source_label`) son etiquetas de intención del operador, no el veredicto de un detector independiente, así que una puntuación alta es una comprobación in-distribution o cerca de distribución, en un laboratorio controlado, con validez externa limitada. Nunca hay que mezclar las métricas de Fase 2 con las de CICIDS2017 interno.

El pipeline de inferencia v2 ejecuta 10 pasos ordenados: (1) lee el CSV de flujos; (2) separa las columnas de metadatos; (3) `maybe_convert_time_units` (si la mediana de `flow_duration < 1.0` trata los tiempos como segundos y multiplica las columnas que contienen duration/iat/active/idle por 1e6 para pasarlas a microsegundos, que es la unidad de CICIDS2017); (4) `map_to_canonical` produce X de forma (n,152) mapeando columnas del extractor a las 76 canónicas más la máscara de 76 vía `FLOWMETER_PY_TO_CANON`; (5) recorte percentil opcional sobre las primeras 76 dims a los p0.5/p99.5 persistidos; (6) `StandardScaler.transform`; (7) recorte z-score opcional (`--clip-z 10.0`); (8) diagnósticos de z-score; (9) carga del modelo y predicción por lotes (`batch_size=4096`, `deterministic=True`); (10) escritura de `predictions.csv`, `config.json`, `metrics.json` y opcionalmente `diagnostics.json` y un CSV sensible gitignored. El modelo se carga con `QRDQN.load`, con fallback a `DQN.load` SOLO si falla el import de sb3_contrib; cualquier otro fallo de carga se propaga. El modelo, el scaler y los percentiles se resuelven con verificación SHA-256 vía `artifact_manifest.json` salvo `--allow-unsafe-artifacts`.

El recorte z-score a 10.0 evita que valores OOD (fuera de distribución) del laboratorio (se han observado z-scores de conteos de flags TCP de hasta |z|=89) empujen las estimaciones Q a regímenes degenerados. *En cristiano: si un dato del laboratorio es rarísimo comparado con lo visto en entrenamiento, lo recorto para que no descoloque al modelo.* Hay una asimetría de preprocesado que debo señalar: el entrenamiento y la evaluación interna de CICIDS2017 usan solo `StandardScaler`, sin recorte (el train calcula y persiste los percentiles, pero ajusta y evalúa sobre características estandarizadas sin recortar). Por eso las métricas de Fase 2 con `--percentiles`/`--clip-z` NO son comparables byte a byte con las del test interno; para una comparación estricta hay que omitir ambos flags. El recorte opera solo sobre las primeras 76 dims de característica; las 76 dims de máscara se cortan y se reconcatenan sin tocar.

`pred_action` es la decisión del modelo: 0 = allow, 1 = block; `block_rate = mean(pred==1)`. Si hay `truth_y` o `truth_label` (BENIGN→0, ATTACK/MALICIOUS→1) se calculan métricas supervisadas con `confusion_to_metrics`. El `predictions.csv` es commit-safe (solo columnas de predicción); las IPs, puertos, timestamps y etiquetas van solo a `predictions_sensitive_local.csv` con `--include-sensitive-metadata`, y ese fichero está gitignored.

El comportamiento de Fase 2 varía por artefacto, así que siempre lo cito por RUN_ID. El artefacto principal, con el modelo del run principal sobre la captura de laboratorio etiquetada, es `P2v2_pred_20260610_161231_MAIN` (usó `--clip-z 10.0`): block_rate 0.252364, accuracy 0.991862, precision de ataque 0.97919, recall de ataque 0.988452, F1 de ataque 0.983801. Los artefactos solo-benignos divergen radicalmente: `P2v2_pred_20260224_004121` da block_rate 1.0 / allow 0.0, mientras que `P2v2_pred_20260408_230318` da block_rate 0.0 / allow 1.0. Nunca debo decir que la Fase 2 está "resuelta": el **desajuste de dominio** (domain shift) es el principal problema abierto. *En cristiano: desajuste de dominio es que el tráfico real del laboratorio no se parece lo bastante al de CICIDS2017, y un modelo que brilla en uno puede tropezar en el otro.*

Sobre el **bloqueo activo**: en el pipeline completo, la fase final extiende la inferencia a un lazo en tiempo real donde `pred_action==1` alimenta un punto de aplicación inline que bloquea el flujo, cerrando el ciclo capturar → extraer flujos → mapear a esquema canónico → inferir → aplicar. Lo que queda por hacer aquí es precisamente ese lazo de enforcement: hoy el modelo solo emite la etiqueta allow(0)/block(1) en `predictions.csv` y ningún punto de aplicación la consume; el bloqueo inline con iptables/nftables es el siguiente hito, a abordar como prototipo controlado solo tras tener evidencia offline estable. De la misma forma, la captura automatizada PCAP → CICFlowMeter → mapeo canónico está documentada como workflow de referencia, pero el dato commiteado se produjo por comandos manuales de consola; la automatización completa de esa cadena, junto con la batería leave-one-CSV-out completa commiteada y una calibración específica para tráfico benigno real del laboratorio, son las piezas que cierran el pipeline.

---

Nota de dosificación para los 20-30 minutos: secciones 1-2 (datos y entorno) en ~5 min; secciones 3-4 (QR-DQN y el RL del proyecto, con el punto del bandit contextual) en ~7 min, que es el núcleo conceptual; sección 5 (entrenamiento y tabla de hiperparámetros) en ~6 min; validación y números oficiales en ~7 min; Fase 2 y lo que queda por hacer en ~4 min. Los tres puntos que el tutor probablemente apretará y para los que tengo respuesta preparada: por qué gamma=0 no lo reduce a un clasificador trivial, por qué la máscara es constante en CICIDS2017, y por qué C03 no es el resultado oficial pese a tener mejor accuracy.

---

## Parte 2 · Banco de respuestas a las preguntas del tutor

### A · Comprensión general del proyecto

**A1. Explica en dos minutos cuál es el problema de investigación del TFM sin usar las palabras del título.**

El problema es decidir, de forma automática, si cada conexión de red que entra a un sistema debe dejarse pasar o cortarse, sabiendo que equivocarse en un sentido cuesta mucho más que en el otro: dejar entrar un ataque es mucho peor que cortar por error una conexión legítima. En lugar de tratar esto como un simple ejercicio de 'etiquetar bien o mal', lo planteo como un agente que toma decisiones y al que se le premia o castiga según el coste real de cada acierto y de cada error. La meta de investigación es construir y validar de extremo a extremo un sistema reproducible que aprenda esa decisión con costes asimétricos explícitos, sobre datos de red representados como resúmenes estadísticos de conexiones (flujos). Y, como segundo objetivo, probar si ese mismo sistema sobrevive cuando lo saco del conjunto de datos académico y lo enfrento a tráfico capturado de verdad en un laboratorio. En una frase: aprender a permitir o bloquear conexiones optimizando el coste de seguridad, no solo la precisión.

> **Qué comprueba / qué no decir:** El tutor comprueba si entiendes el PROBLEMA o solo repites el título. No digas 'detección de intrusiones con deep learning'; el núcleo es la decisión permit/block con costes asimétricos y la honestidad de que es un solo paso, no una defensa secuencial.


**A2. ¿Por qué el trabajo no plantea simplemente un clasificador supervisado tradicional?**

Porque un clasificador supervisado clásico se entrena para minimizar el error de clasificación tratando, por defecto, todos los errores como igual de costosos, mientras que en ciberseguridad un falso negativo (dejar pasar un ataque) es mucho más grave que un falso positivo (bloquear tráfico bueno). Yo quería que ese coste asimétrico viviera de forma explícita y configurable en la propia señal de aprendizaje: en mi caso, la recompensa penaliza el falso negativo con -5.0 frente a -2.0 del falso positivo, es decir, 2.5 veces peor. Esto se hace mediante aprendizaje por refuerzo (un agente que recibe recompensa o castigo por sus acciones) en lugar de entropía cruzada sobre etiquetas. Tengo que ser honesto: como uso gamma=0.0, en la práctica es una formulación de coste de un solo paso, técnicamente un 'bandit contextual', muy emparentada con un clasificador sensible al coste. La diferencia real es que el coste está en la recompensa y no en un peso de clase, la decisión es el argmax de un valor aprendido, y la cabeza distribucional de QRDQN modela toda la distribución del retorno, no solo su media. De hecho, comparo contra un clasificador clásico (Random Forest) precisamente para no sobrevender el RL.

> **Qué comprueba / qué no decir:** Comprueba si reconoces que con gamma=0 esto se parece muchísimo a un clasificador. NO defiendas que es 'RL secuencial de verdad'. Lo defendible es: formulación cost-sensitive explícita, no que el agente planifique en el tiempo.


**A3. ¿Qué significa que las decisiones sean binarias en este TFM?**

Significa que, ante cada flujo de red, el agente solo puede elegir entre dos acciones: 0 = PERMIT (dejar pasar) o 1 = BLOCK (bloquear). No hay acciones intermedias como 'inspeccionar más', 'poner en cuarentena', 'limitar la velocidad' o 'avisar a un humano'; el espacio de acción es Discrete(2) y cualquier acción fuera de ese rango el entorno la rechaza con error. Esto simplifica el problema a una decisión de seguridad fundamental: permitir o cortar. La etiqueta verdadera también es binaria: 0 = benigno, 1 = ataque, con la regla de que cualquier cosa que no sea 'BENIGN' cuenta como ataque. La ventaja es claridad y comparabilidad; la limitación, que una defensa real suele tener una respuesta más graduada, y eso lo reconozco como simplificación de diseño.

> **Qué comprueba / qué no decir:** El tutor sondea si confundes 'binario' con 'fácil' o si entiendes la pérdida de matiz. Conviene admitir que reducir todo a permit/block y a benigno/ataque pierde información (tipo y gravedad del ataque).


**A4. ¿Qué diferencia hay entre PERMIT y BLOCK como acciones experimentales y un bloqueo real en una red?**

En el experimento, PERMIT y BLOCK son simplemente etiquetas que el agente emite sobre un flujo ya capturado y guardado: no detienen tráfico de verdad, solo se comparan contra la etiqueta correcta para calcular la recompensa y las métricas. Es inferencia offline; el resultado se escribe en un fichero predictions.csv y nadie lo consume para actuar. Un bloqueo real implicaría un punto de aplicación 'inline' (por ejemplo, reglas de iptables o nftables) que de verdad cortara la conexión en tiempo real, con riesgo operativo: latencia, cortar tráfico legítimo de clientes reales, y la posibilidad de que el atacante se adapte. Ese lazo de enforcement está documentado como el siguiente hito pero NO está implementado todavía; hoy el modelo solo produce la decisión, sin que ningún sistema la aplique. Por eso es importante no afirmar que el sistema 'bloquea ataques en producción': aún no lo hace.

> **Qué comprueba / qué no decir:** Verifica si exageras el alcance. NO digas que ya bloquea tráfico real ni en tiempo real. La línea honesta: hoy es solo una predicción en disco; el enforcement inline es trabajo futuro.


**A5. ¿Cuál es la contribución principal del trabajo: el algoritmo, el dataset, el protocolo de evaluación o la formulación metodológica?**

La contribución principal no es el algoritmo (QRDQN ya existe) ni el dataset (CICIDS2017 es público), sino la formulación metodológica y el pipeline coherente y reproducible que la sostiene. Lo central es: (1) un esquema canónico fijo de 76 características de flujo más una máscara de ausencia de 76, dando un vector de observación estable de 152 dimensiones que no cambia aunque cambie el dataset; (2) plantear la decisión permit/block como aprendizaje de valor sobre una matriz de coste asimétrica explícita; y (3) un protocolo de evaluación serio que va más allá de la accuracy: validación directa, prueba anti-fuga con etiquetas barajadas, generalización dura por día/CSV, comparación con un baseline Random Forest bajo el mismo protocolo, e intervalos de confianza bootstrap. El algoritmo y el dataset son piezas; el valor está en cómo los conecto y los valido con trazabilidad por RUN_ID. Si tuviera que elegir una sola, diría la formulación metodológica, con el protocolo de evaluación como segundo pilar.

> **Qué comprueba / qué no decir:** Comprueba que no vendes 'inventé un algoritmo nuevo'. NO atribuyas novedad a QRDQN ni a CICIDS2017. Lo defendible es el diseño/integración del pipeline y el rigor de la evaluación.



### B · Ciberseguridad y datos

**B6. ¿Qué es un flujo de red y qué información resume?**

Un flujo de red es el resumen estadístico de una conversación entre dos máquinas: agrupa todos los paquetes que pertenecen a la misma conexión (mismo origen, destino, puertos y protocolo) y los condensa en un conjunto de números que describen su comportamiento agregado, no su contenido. En este proyecto cada flujo se reduce a 76 características numéricas: duración de la conexión, número y tamaño de paquetes en cada sentido (ida y vuelta), tasas de bytes y paquetes por segundo, tiempos entre llegadas de paquetes, recuento de banderas TCP (SYN, ACK, FIN, RST, etc.), longitudes de cabecera, estadísticas de ventana TCP y tiempos de actividad/inactividad. Es decir, resume la 'forma' y el 'ritmo' de la comunicación. Lo que NO incluye es la carga útil (el contenido real de los mensajes), ni identificadores como IPs, puertos o marcas de tiempo, que se descartan a propósito por la política anti-fuga.

> **Qué comprueba / qué no decir:** Verifica que sabes qué guarda y qué NO un flujo. No digas que un flujo lee el contenido de los paquetes; eso sería inspección profunda. El flujo son metadatos/estadísticas agregadas.


**B7. ¿Por qué se usan flujos en lugar de inspección profunda de paquetes?**

Por tres razones prácticas. Primera, escalabilidad: un resumen estadístico por conexión es muchísimo más ligero de procesar que inspeccionar el contenido de cada paquete a gran velocidad. Segunda, privacidad y cifrado: la inspección profunda (DPI) mira dentro de la carga útil, lo que choca con la privacidad y, sobre todo, hoy la mayoría del tráfico va cifrado (HTTPS), así que el contenido no es legible; los flujos siguen funcionando porque describen el comportamiento, no el contenido. Tercera, reproducibilidad y portabilidad: las características de flujo se extraen con herramientas estándar como CICFlowMeter o Zeek tanto de un dataset como de tráfico real capturado, lo que me permite usar el mismo esquema de entrada en el dataset y en el laboratorio. De hecho, mi criterio para elegir las 76 características exige que sean extraíbles de PCAP real, no peculiaridades del dataset. La contrapartida: al no mirar el contenido, se pierden señales que un análisis profundo sí vería.

> **Qué comprueba / qué no decir:** Sondea si entiendes la relación con el cifrado y con la portabilidad a tráfico real. No vendas los flujos como superiores en todo; admite que renuncias a información de contenido.


**B8. ¿Qué problemas conocidos tiene CICIDS2017?**

CICIDS2017 es un referente, pero arrastra problemas documentados que asumo abiertamente. Tiene duplicados y registros muy repetidos, lo que puede inflar artificialmente los resultados si un split aleatorio pone copias casi idénticas en train y en test (por eso incluí un análisis de duplicados de solo lectura). Tiene errores y artefactos en la extracción de características hechos con CICFlowMeter, y un fuerte desbalanceo de clases (en mi test, ~80% benigno frente a ~20% ataque). Además, ciertos atributos pueden actuar como atajos hacia la etiqueta: el caso típico es el puerto de destino, que correlaciona con tipos concretos de ataque; por eso lo elimino explícitamente, junto con IPs, Flow ID y timestamps. Y, siendo de 2017, es tráfico de laboratorio antiguo que no representa una red empresarial moderna. Por todo esto, una métrica alta en CICIDS2017 hay que leerla con cautela y por eso añado las validaciones duras y la comparación con Random Forest.

> **Qué comprueba / qué no decir:** Comprueba que conoces las críticas reales al dataset (duplicados, fuga por puerto, desbalanceo, antigüedad) y no lo presentas como dato perfecto. No digas que CICIDS2017 'es representativo de redes reales'.


**B9. ¿Por qué un resultado alto en CICIDS2017 no demuestra que el modelo funcione en una empresa real?**

Porque una accuracy del 0.99381 y un recall de ataque del 0.99536 se obtienen sobre un dataset de laboratorio de 2017, con un split aleatorio estratificado donde train y test comparten la misma distribución, e incluso pueden compartir patrones casi duplicados. Eso mide cómo de bien aprende dentro de ese universo concreto, no cómo generaliza fuera. La prueba de ello es mi propio Check C (split duro por día/CSV, con ataques no vistos): el recall de ataque cae de 0.99536 a 0.52954 y la accuracy a 0.84135. Y al llevarlo a tráfico real de laboratorio, el comportamiento se vuelve muy sensible a la configuración: hay artefactos que bloquean el 100% del tráfico benigno y otros que permiten el 100%. Una empresa real tiene tráfico distinto, ataques nuevos y una distribución cambiante (domain shift). Por tanto el número alto demuestra competencia en el dominio de entrenamiento, no validez externa en producción.

> **Qué comprueba / qué no decir:** Verifica si caes en sobreafirmar. Usa tus propios números de caída (Check C: recall 0.53) como prueba de honestidad. No presentes 0.99 como evidencia de que sirve en producción.


**B10. ¿Qué variables podrían producir fuga de información y por qué?**

Fuga de información es cuando el modelo aprende a 'hacer trampa' usando una variable que delata la etiqueta sin reflejar el comportamiento real del ataque. Las principales en este dominio son: el puerto de destino (ciertos ataques usan puertos fijos, así que el puerto sería casi un sinónimo de la etiqueta), las direcciones IP (de origen y destino: en un dataset de laboratorio el atacante suele tener IP fija, y el modelo aprendería 'esta IP = ataque'), las marcas de tiempo / timestamps (los ataques se lanzaron en franjas horarias concretas, así que la hora delata la clase) y el Flow ID (un identificador único que no generaliza). Por eso mi política anti-fuga descarta explícitamente IPs, puertos de origen y destino, timestamps y Flow ID antes de entrenar, usando una lista exacta de nombres más reglas de subcadena. La idea es forzar al agente a aprender la 'forma' del flujo y no atajos triviales. Mi Check B confirma que no hay fuga: con etiquetas barajadas la accuracy se queda en 0.4773, cerca del baseline de clase mayoritaria 0.5227, en lugar de dispararse.

> **Qué comprueba / qué no decir:** El tutor del listado 'especialmente útiles' insiste: ¿dónde se cuela fuga aunque quites lo obvio? Menciona también duplicados entre splits y, sutilmente, que el puerto no es solo 'identificador' sino proxy de etiqueta. No digas que basta con quitar IPs.



### C · Preprocesamiento y esquema canónico

**C11. ¿Qué significa que el TFM use un esquema canónico de 76 variables?**

Significa que, en lugar de aceptar las columnas tal cual vengan de cada fuente de datos, fijo de antemano una lista cerrada y ordenada de 76 características numéricas que describen un flujo de red (su nombre exacto es FEATURES_CANON en src/canonical_schema.py). 'Canónico' quiere decir 'forma de referencia única': todo dato, venga de CICIDS2017, de NSL-KDD o del laboratorio, se traduce siempre a esas mismas 76 variables en el mismo orden. Las 76 no son arbitrarias: las elegí con cinco criterios deliberados: que existan en CICIDS2017, que se puedan extraer de tráfico real con herramientas como CICFlowMeter o Zeek, que no sean identificadores ni delaten la etiqueta (nada de IPs, puertos o timestamps), que sean estadísticas numéricas de flujo y que sean estables. La ventaja práctica es la comparabilidad: el modelo siempre ve un vector con el mismo significado en cada posición, así que un modelo entrenado con un dataset puede aplicarse a otro sin reescribir el preprocesado.

> **Qué comprueba / qué no decir:** El tutor comprueba si entiendes que 'canónico' = formato común reutilizable entre datasets, no una lista cualquiera. No digas que son simplemente 'las columnas del dataset'; el punto es que es un esquema diseñado con criterios anti-fuga y de portabilidad.


**C12. ¿Por qué se añade una máscara de ausencia?**

La máscara es un segundo bloque de 76 valores (0 o 1) que acompaña a las 76 características, formando un vector de observación total de 152 dimensiones (obs = [x_1..x_76, m_1..m_76], NUM_OBSERVATION_FEATURES = 76*2). Cada m_i indica si esa característica estaba realmente presente y era un número válido en la fila de origen (m_i=1) o si tuve que rellenarla porque faltaba, era NaN o infinita (m_i=0, rellenada con 0.0). El problema que resuelve es esta ambigüedad: si imputo un dato ausente con 0.0, el modelo no sabe distinguir 'esta variable valía cero de verdad' de 'esta variable no existía y la inventé'; la máscara le dice cuál de las dos cosas es, es decir, marca de qué se puede fiar. Tengo que ser honesto: sobre CICIDS2017 nativo el mapeo cubre las 76 variables, así que la máscara vale 1 en todas las filas, es constante y no aporta nada al resultado interno de CICIDS2017. Solo se vuelve informativa cuando hay desajuste de dominio, por ejemplo con NSL-KDD, cuyo mapeo solo cubre 3 de las 76 variables y deja la máscara casi toda a 0, o en la Fase 2 con tráfico del laboratorio.

> **Qué comprueba / qué no decir:** El tutor (pregunta 'útil' nº 7) prueba si sabes que la máscara NO es un detalle menor pero que en CICIDS2017 es constante. Si dices que 'mejora el resultado en CICIDS2017' mientes; lo defendible es que es una decisión de diseño para portabilidad entre dominios, inerte en el dataset principal.


**C13. Si una variable no existe en el tráfico de laboratorio, ¿qué ocurre con ella en el vector de entrada?**

Esa variable no desaparece del vector: el esquema canónico siempre tiene 76 posiciones, así que esa posición se rellena (se imputa) con el valor por defecto 0.0 (DEFAULT_IMPUTATION_VALUE), y a la vez su casilla de máscara se pone a 0 para dejar constancia de que ese valor es inventado, no medido. Así el vector mantiene su forma fija de 152 dimensiones y el modelo puede procesarlo sin errores, pero queda señalado que esa característica no es fiable. La imputación es por celda: una misma columna puede estar presente en unas filas y ausente en otras, y la máscara lo registra fila a fila, no de forma global. La consecuencia es que cuantas más variables falten en el laboratorio, más ceros 'falsos' verá el modelo y más se aleja la entrada de lo que vio en entrenamiento, lo que conecta con el problema de desajuste de dominio que aún tengo abierto en la Fase 2.

> **Qué comprueba / qué no decir:** Comprueba si entiendes el mecanismo imputación + máscara y que la dimensión es fija. No digas que 'se elimina la variable' ni que 'el vector cambia de tamaño': siempre son 152 dimensiones; lo que cambia es el contenido (0.0) y la marca de la máscara (0).


**C14. ¿Por qué el escalado debe ajustarse solo con datos de entrenamiento?**

Escalar (uso un StandardScaler de sklearn, que resta la media y divide por la desviación de cada característica, el llamado z-score) requiere calcular esas medias y desviaciones a partir de algún conjunto de datos. Si las calculo usando también el test, estoy dejando que información del test influya en el preprocesado, y eso es fuga de información (data leakage): el modelo se beneficia indirectamente de datos que se supone que no ha visto, y las métricas salen artificialmente buenas e irreales. Por eso ajusto (fit) el scaler SOLO sobre X_train y luego me limito a aplicarlo (transform) al test con esos mismos parámetros. En el proyecto esto está cuidado al detalle: el escalado se gestiona dentro de train_rl_defender.py, se ajusta un StandardScaler fresco sobre X_train y se persiste con joblib; además verifiqué que la media y la escala del scaler reproducido coinciden con el scaler.joblib del run oficial. El test debe tratarse como datos nuevos que llegan en producción, donde obviamente no podrías 'mirar el futuro' para calibrar.

> **Qué comprueba / qué no decir:** El tutor verifica que entiendes 'fit en train, transform en test' como prevención de fuga. No mezcles esto con el split por grupos; aquí el punto es estadístico (medias/desviaciones), no temporal.


**C15. ¿Qué pasaría si se normalizan los datos antes de separar entrenamiento y prueba?**

Sería un error metodológico clásico: al calcular las medias y desviaciones sobre el conjunto completo (train + test juntos), el preprocesado del train queda 'contaminado' con estadísticas que incluyen al test. Eso es fuga de información, y su efecto típico es inflar las métricas: el modelo parece mejor de lo que sería con datos verdaderamente nuevos, porque la evaluación deja de ser independiente. En este TFM el riesgo sería grave porque mis cifras principales son muy altas (accuracy 0.99381, recall de ataque 0.99536) y una fuga así me haría dudar de si son reales. Por eso el orden correcto, y el que sigo, es: primero separar (split estratificado 80/20, seed 42), luego ajustar el scaler solo con el train y después transformar el test; nunca al revés. De hecho monté comprobaciones expresas para descartar fugas, como el Check B con etiquetas barajadas, precisamente porque resultados tan altos exigen demostrar que no vienen de un atajo de este tipo.

> **Qué comprueba / qué no decir:** Quiere ver si reconoces que normalizar antes del split = fuga = métricas infladas. No lo confundas con duplicados ni con split por día; es específicamente el orden scaler/split. Vincular con tus métricas altas y los checks anti-fuga lo hace más creíble.



### D · Aprendizaje por refuerzo

**D16. ¿Qué convierte este problema en un problema de aprendizaje por refuerzo?**

Lo formulo como aprendizaje por refuerzo (RL) porque no entreno al modelo con la respuesta correcta directa (como haría un clasificador con entropía cruzada sobre etiquetas), sino con una señal de recompensa que premia o castiga sus decisiones según una matriz de costes. El entorno (RLDatasetDefenderEnv, un gym.Env de Gymnasium) presenta al agente el vector de un flujo, el agente elige una acción (dejar pasar o bloquear) y recibe una recompensa que codifica cuánto cuesta acertar o equivocarse. El agente (QR-DQN) aprende a maximizar esa recompensa acumulada eligiendo la mejor acción, que es el esquema central del RL. Ahora bien, tengo que ser honesto sobre el alcance: como gamma=0.0 y cada flujo es independiente del anterior, en la práctica es un bandit contextual (un RL de un solo paso por contexto), no un MDP secuencial con planificación a futuro. Lo que sí lo hace genuinamente RL y no clasificación es que aprendo valor sobre una matriz de coste asimétrica explícita (un falso negativo vale -5.0 frente a -2.0 de un falso positivo) y la decisión es el argmax de ese valor aprendido entre PERMIT y BLOCK.

> **Qué comprueba / qué no decir:** Es la pregunta 'útil' nº 2 y nº 6: el tutor sospecha que es 'clasificación disfrazada de RL'. NO digas que es RL secuencial pleno. La respuesta defendible es: interfaz RL real + aprendizaje de valor sensible al coste, pero honestamente un bandit contextual por gamma=0. Reconocerlo tú mismo desarma la trampa.


**D17. ¿Cuál es el estado, cuál es la acción y cuál es la recompensa?**

El estado (la observación) es el vector de 152 dimensiones de un único flujo de red: las 76 características canónicas estandarizadas más las 76 de la máscara de ausencia. La acción es binaria, Discrete(2): 0 = PERMIT (dejar pasar el flujo) o 1 = BLOCK (bloquearlo). La recompensa es una función pura de (etiqueta verdadera, acción) que implementa una matriz de costes asimétrica, con estos valores exactos: verdadero positivo (ataque bloqueado) = +1.5, falso positivo (benigno bloqueado) = -2.0, falso negativo (ataque que dejo pasar) = -5.0, y el término 'omission' (benigno permitido, el equivalente al verdadero negativo) = 0.0. Importante: no existe una clave 'tn'; el verdadero negativo se llama 'omission' y vale 0.0. La asimetría es deliberada: castigar el falso negativo con -5.0, 2.5 veces más que el falso positivo, le dice al agente que dejar pasar un ataque es mucho peor que una falsa alarma, y así trato el desbalanceo de clases de forma económica, sin remuestreo, empujando el recall de la clase de ataque (la rara).

> **Qué comprueba / qué no decir:** Verifica que sabes los tres elementos y los números exactos, y que 'tn' se llama 'omission' = 0.0. No inventes un valor de tn distinto de 0. El detalle de que el desbalanceo se trata vía recompensa asimétrica (no oversampling) es lo que demuestra comprensión real.


**D18. ¿Por qué el entorno se define como offline?**

Offline significa que el agente no interactúa con una red viva que reacciona a sus decisiones, sino que aprende sobre un conjunto de datos ya capturado, etiquetado y estático (CICIDS2017). El entorno toma ese array de flujos, lo baraja en cada reset() y va recorriéndolo flujo a flujo: cada step() le muestra una muestra, recibe la acción y devuelve la recompensa según la etiqueta ya conocida de esa muestra. No hay nada en tiempo real ni consecuencias sobre tráfico futuro. Es offline por dos razones: (1) seguridad y reproducibilidad, porque entrenar un agente que bloquea tráfico real sería peligroso y no repetible, y (2) porque CICIDS2017 ya viene grabado con sus etiquetas, así que el 'mundo' del agente es ese fichero fijo. Esto encaja con que sea un bandit contextual: la siguiente observación no depende de lo que el agente hizo antes, es simplemente la siguiente entrada del dataset permutado.

> **Qué comprueba / qué no decir:** Comprueba que distingues 'offline' (aprender de datos grabados) de 'online' (interactuar con un sistema vivo). No digas que el agente 'explora la red'; explora un dataset fijo. Conecta offline con la ausencia de transiciones reales (bandit).


**D19. ¿Qué limitación tiene tratar un dataset estático como entorno?**

La limitación de fondo es que un dataset estático no es un entorno real: no hay dinámica ni consecuencias. En un MDP de verdad la acción del agente cambia el estado siguiente (bloquear un flujo afectaría al tráfico posterior, al atacante, etc.), pero aquí la siguiente observación es una muestra independiente extraída del array permutado, totalmente no correlacionada con la acción anterior. Por eso no hay asignación temporal de crédito y el factor de descuento gamma es irrelevante (de hecho lo fijo en 0.0), lo que reduce el problema a un bandit contextual: decisiones de un solo paso. Además, el dataset fija el universo de ataques y de tráfico: el agente solo aprende lo que CICIDS2017 contiene, con sus problemas conocidos (etiquetas y artefactos del dataset), y no se adapta a amenazas nuevas ni a un adversario que cambia de táctica. Y como las etiquetas vienen dadas, el aprendizaje depende por completo de que esas etiquetas sean correctas. En resumen: gano reproducibilidad y seguridad, pero pierdo realismo, dinámica adversaria y validez externa.

> **Qué comprueba / qué no decir:** El tutor mide tu honestidad metodológica. La trampa es presumir de que es RL completo. Lo correcto: reconocer que sin transiciones reales no hay secuencialidad, gamma es irrelevante, y el realismo/validez externa quedan limitados. Esto es justo lo que las preguntas del bloque J y nº 2/6 buscan.


**D20. ¿En qué sentido esta formulación no representa una defensa autónoma real?**

En varios sentidos que reconozco abiertamente. Primero, no hay enforcement: hoy el modelo solo escribe una etiqueta allow(0)/block(1) en predictions.csv; ningún componente consume esa decisión para bloquear tráfico de verdad. El bloqueo inline real (con iptables/nftables) es un hito pendiente, no algo implementado. Segundo, no es secuencial ni adaptativa: al ser un bandit contextual con gamma=0, decide flujo a flujo sin memoria ni planificación, no reacciona a un ataque que evoluciona en el tiempo. Tercero, solo tiene dos acciones (permit/block); una defensa real tendría más respuestas graduadas (alertar, poner en cuarentena, limitar ancho de banda). Cuarto, opera offline sobre datos grabados y etiquetados por un humano, no sobre tráfico vivo: en la Fase 2, las etiquetas son la intención del operador del laboratorio, no el veredicto de un detector independiente, con validez externa limitada. Por eso nunca afirmo que la Fase 2 esté 'resuelta': el desajuste de dominio entre laboratorio y entrenamiento es el principal problema abierto. Es, honestamente, una prueba de concepto metodológica de decisión sensible al coste, no un defensor autónomo desplegable.

> **Qué comprueba / qué no decir:** Pregunta 'útil' nº 9 (más acciones) y nº 5 (qué NO puedes afirmar). El tutor quiere que no sobrevendas el sistema. No digas que 'bloquea ataques en tiempo real'. Lo defendible: faltan enforcement, secuencialidad, acciones ricas y validación en producción; es PoC, no producto.



### E · Función de recompensa y costes asimétricos

**E21. ¿Por qué un falso negativo es más grave que un falso positivo en este contexto?**

Un falso negativo significa que un ataque real entra como tráfico permitido: el agente decidió PERMIT sobre un flujo malicioso, y ese ataque puede causar daño dentro de la red sin que nadie lo detenga. Un falso positivo es bloquear tráfico benigno: molesto, porque corta a un usuario legítimo, pero recuperable y normalmente reversible. En seguridad el coste de un ataque que se cuela suele ser mucho mayor que el de una falsa alarma. Por eso en este proyecto se asume explícitamente que dejar pasar un ataque es 2.5 veces peor que una falsa alarma. Además, los ataques son la clase rara (en el test solo el 19.7% de los flujos son ataque), así que perderlos importa especialmente.

> **Qué comprueba / qué no decir:** El tutor comprueba que entiendes la asimetría de coste como decisión de diseño, no como dogma. No digas que un FP 'no importa'; importa, pero menos. Reconoce que el ratio 2.5x es una elección concreta, no una verdad universal.


**E22. ¿Cómo se refleja esa diferencia en la función de recompensa?**

La recompensa es una matriz de coste asimétrica que solo depende de (etiqueta verdadera, acción), no del historial. Los valores son: verdadero positivo (ataque bloqueado) +1.5, falso positivo (benigno bloqueado) -2.0, falso negativo (ataque permitido) -5.0 y el verdadero negativo, llamado 'omission' (benigno permitido), 0.0. El falso negativo (-5.0) es el castigo más fuerte, 2.5 veces el falso positivo (-2.0). Esa magnitud mayor empuja al agente a preferir bloquear cuando duda ante un posible ataque, subiendo el recall de la clase rara. Es importante: el desbalanceo de clases no se trata con pesos ni sobremuestreo, sino de forma económica, solo con estas magnitudes de recompensa.

> **Qué comprueba / qué no decir:** Verifica que sabes los números exactos y que el TN se llama 'omission' (no existe clave 'tn'). No inventes una penalización al FP igual o mayor que al FN. Menciona que el desbalanceo se gestiona vía recompensa, no vía resampling.


**E23. ¿Qué riesgo aparece si se penalizan demasiado los falsos negativos?**

Si castigas el falso negativo en exceso, el agente aprende que casi siempre es más seguro bloquear, así que tiende a marcar como ataque casi todo el tráfico. Eso sube el recall a costa de disparar los falsos positivos: bloqueas mucho tráfico legítimo. En el extremo se degenera en una política trivial de 'bloquear todo' que tiene recall perfecto pero es inútil porque corta la red. De hecho, en los artefactos de Fase 2 se vio justo este colapso degenerado: un run con block_rate 1.0 (bloquea absolutamente todo) y otro con block_rate 0.0 (deja pasar todo). El equilibrio que se eligió aquí (+1.5/-2.0/-5.0) busca un recall alto sin caer en bloquear indiscriminadamente, y se mide con métricas que penalizan el exceso de FP como la precisión y el MCC.

> **Qué comprueba / qué no decir:** El tutor prueba si entiendes el trade-off recall/precision. No digas solo 'más recall es mejor'. Nombra el caso degenerado de bloquear todo y conecta con que por eso hay un +1.5 al TP y un -2.0 al FP que frenan ese comportamiento.


**E24. ¿Cómo interpretarías un modelo con recall muy alto pero muchos falsos positivos?**

Sería un modelo paranoico: detecta casi todos los ataques (recall alto) pero a costa de bloquear mucho tráfico legítimo, así que su precisión sería baja. Atrapa lo malo, pero genera tantas falsas alarmas que en producción sería inviable: los operadores se saturan y los usuarios legítimos quedan cortados. El recall por sí solo engaña, porque un modelo que bloquea todo tiene recall 1.0 y es inútil. Por eso aquí no se mira solo el recall: se reporta también la precisión de ataque (0.97378 en el run oficial), el F1 (0.98445) y el MCC (0.98068), que combinan ambos lados. En nuestro caso el recall alto (0.99536) viene además acompañado de pocos FP (2.989 de 454.620 benignos, FPR 0.66%), así que no es paranoico; pero la pregunta describe el escenario que precisamente esas métricas sirven para detectar.

> **Qué comprueba / qué no decir:** Comprueba que no confundes recall con calidad global. No celebres el recall aislado. Menciona precision/F1/MCC y el caso 'bloquear todo'. Puedes citar que el run real NO sufre esto, con sus números.


**E25. ¿Qué métrica mirarías primero si el objetivo es evitar ataques no detectados?**

El recall de la clase ataque, o de forma equivalente su complementario, la tasa de falsos negativos (FNR). El recall mide qué fracción de los ataques reales se detectan; un ataque no detectado es exactamente un falso negativo. En el run oficial el recall de ataque es 0.99536 y el FNR 0.00465, es decir, solo se pierden 518 ataques de 111.529. Ahora bien, el recall solo no basta como criterio único, porque se puede inflar bloqueando todo; por eso, una vez fijado un recall alto, hay que mirar la precisión y el F1 para confirmar que no se logró a base de falsas alarmas masivas. Para evitar ataques no detectados, recall primero; pero la decisión final se toma con el conjunto recall + precision + F1/MCC.

> **Qué comprueba / qué no decir:** Trampa clásica: si dices 'accuracy' fallas, porque con clases desbalanceadas la accuracy se infla con la clase mayoritaria (benigna). Di recall/FNR de ataque primero, y aclara que no es la única métrica para no caer en bloquear todo.



### F · QR-DQN

**F26. ¿Qué es QRDQN, explicado de forma comprensible?**

QRDQN (Quantile Regression DQN) es una variante del algoritmo de aprendizaje por refuerzo DQN. Un DQN normal aprende, para cada acción posible, un único número: cuánto valor (recompensa esperada) cree que dará esa acción en promedio. QRDQN, en cambio, no aprende solo ese promedio sino toda la distribución de resultados posibles, aproximada por un conjunto de cortes llamados cuantiles (aquí 200). Imagina que en vez de decir 'esta acción da de media 1.5' dice 'esta acción da un abanico de resultados, y aquí está su forma completa'. Cada uno de esos 200 cuantiles es un trocito equiprobable de esa distribución. Para decidir qué hacer, QRDQN colapsa esa distribución a su media y elige la acción con mayor valor medio; la distribución sirve para aprender mejor y representar la incertidumbre.

> **Qué comprueba / qué no decir:** El tutor quiere ver que entiendes 'distribucional' en lenguaje llano, no que recitas el paper. No te enredes con la quantile-Huber loss salvo que pregunten. Deja claro que para ACTUAR se usa la media (argmax del Q medio).


**F27. ¿En qué se diferencia QRDQN de un DQN estándar?**

La diferencia central es qué aprende cada uno por cada par estado-acción. El DQN estándar aprende un solo valor escalar, la media de la recompensa esperada. QRDQN aprende la distribución completa de esa recompensa, representada como 200 cuantiles, y solo recupera la media cuando necesita elegir acción. Por dentro, la red de QRDQN produce action_dim x n_quantiles valores en lugar de un valor por acción, y se entrena con una pérdida específica (quantile-Huber loss) en vez del error cuadrático típico de DQN. Todo lo demás es muy parecido a DQN: es off-policy, usa un replay buffer para decorrelar las muestras, exploración epsilon-greedy y una red objetivo separada que se copia en duro cada cierto número de pasos. La selección de acción siguiente la hace la propia red objetivo por su Q medio, así que NO es el desacople tipo Double-DQN.

> **Qué comprueba / qué no decir:** Comprueba que distingues 'media' (DQN) de 'distribución' (QRDQN) y que no inventas que QRDQN hace Double-DQN. Bonus: que sabes que comparten replay buffer, target net y epsilon-greedy.


**F28. ¿Por qué podría tener sentido usar un método distribucional en este problema?**

La justificación honesta tiene dos capas. La conceptual: un método distribucional modela toda la distribución de retornos, no solo la media, lo que en seguridad encaja bien porque las decisiones llevan riesgo asimétrico (un fallo grave, el falso negativo, cuesta -5.0) y conocer la forma de la distribución, no solo el promedio, da información sobre esa incertidumbre. La práctica/empírica: la cabeza distribucional tiende a dar un aprendizaje más estable y rico que un solo escalar. Pero debo ser sincero sobre la limitación real de ESTE proyecto: como gamma=0.0, el problema es un bandit contextual de un solo paso (cada decisión es independiente), así que la ventaja distribucional aquí no se explota para planificar en el tiempo; solo aporta como mejor estimador de valor cost-sensitive, no como gestor de riesgo secuencial. En rigor, no he demostrado que QRDQN supere a un DQN simple en este montaje.

> **Qué comprueba / qué no decir:** Trampa: que vendas la parte distribucional como imprescindible. Reconoce que con gamma=0 es bandit y que no has probado que lo distribucional sea necesario aquí; di que es una elección defendible, no una superioridad demostrada.


**F29. ¿Qué hiperparámetros del agente podrían influir mucho en los resultados?**

El más estructural es gamma, el factor de descuento: aquí está fijado en 0.0, lo que colapsa el problema a un bandit contextual de un paso; cambiarlo redefiniría el problema entero. Después, la configuración de recompensa (+1.5/-2.0/-5.0/0.0) es decisiva, porque es toda la señal de aprendizaje y mueve directamente el equilibrio recall/precision. Luego también pesan: la arquitectura de red (net_arch, el run principal usa [1024,1024,512]), el learning_rate (5e-5 en el run largo), el número de timesteps (3.000.000), el tamaño del replay buffer (1M) y los parámetros de exploración (epsilon final 0.02, fracción 0.10). El número de cuantiles (200) define la resolución de la distribución. Hay que reconocer una inconsistencia: el tuner con Optuna buscó gamma en [0.95, 0.999], régimen secuencial, mientras producción usa gamma=0.0, así que esos valores tuneados no son transferibles, y la config del run principal se fijó a mano, no salió de Optuna.

> **Qué comprueba / qué no decir:** El tutor puede tirar del hilo de que tuneaste gamma alto pero usas gamma=0. Adelántate y reconócelo. No presentes la config principal como salida de Optuna: fue fijada a mano y congelada por tests.


**F30. ¿Qué evidencia necesitarías para afirmar que QRDQN aporta algo frente a Random Forest?**

Necesitaría una comparación justa y repetida bajo el mismo protocolo, y hoy no la tengo del todo. Primero, varias semillas de entrenamiento para ambos (ahora QRDQN solo se entrenó con un seed, así que los intervalos de confianza que tengo solo miden el resampling del test de UN modelo, no la variabilidad de entrenamiento). Segundo, comparar en los splits que de verdad miden generalización (por día y leave-one-CSV-out), no solo en el split aleatorio. Ahí está la pista más interesante: en split aleatorio RF gana marginalmente a QRDQN (RF accuracy 0.99872, F1 0.99676 frente a QRDQN 0.99381/0.98445), pero en split por día RF colapsa (recall de ataque 0.08135) mientras QRDQN aguanta mejor (recall 0.52954 en Check C). Para afirmar que QRDQN aporta, necesitaría confirmar con múltiples seeds que esa mayor robustez al cambio de día es consistente y estadísticamente sólida, además de completar y commitear la batería leave-one-CSV-out de QRDQN, que aún no existe.

> **Qué comprueba / qué no decir:** Trampa: presumir que QRDQN 'ya ganó'. En split aleatorio RF es mejor; solo en split por día QRDQN parece más robusto, y con un solo seed no es concluyente. Reconoce el seed único y que falta el artefacto LOO de QRDQN.



### G · Comparación con Random Forest

**G31. ¿Por qué Random Forest es un baseline razonable para este TFM?**

Un baseline es un método de referencia sencillo y bien conocido: si tu propuesta nueva no lo supera, no has demostrado que aporte nada. Random Forest (un conjunto de árboles de decisión que votan) es exactamente eso para la clasificación tabular: es robusto, rápido de entrenar, maneja bien características numéricas como las 76 estadísticas de flujo y es un estándar de facto en detección de intrusiones, así que un tribunal lo reconoce de inmediato. En mi proyecto está pensado para ser comparable de verdad: usa la misma observación canónica, las mismas características escaladas (aunque RF es invariante a la escala, mantengo el preprocesado idéntico) y class_weight='balanced', que imita el tratamiento sensible al coste de mi recompensa asimétrica. Y se evalúa con el mismo confusion_to_metrics que QRDQN. La pregunta honesta que responde es: ¿hace falta aprendizaje por refuerzo, o un clasificador clásico ya resuelve el problema?

> **Qué comprueba / qué no decir:** El tutor comprueba que entiendes para qué sirve un baseline (refutar tu propia propuesta), no que lo metiste por rellenar. NO presentes RF como un rival débil de adorno; preséntalo como un baseline fuerte y comparable que de hecho te gana en el split fácil.


**G32. ¿Qué significa una comparación bajo el mismo protocolo?**

Significa que las dos cosas que comparas solo se diferencian en lo que quieres medir, y todo lo demás se mantiene idéntico, para que la comparación sea justa. En mi caso QRDQN y Random Forest ven el mismo vector de entrada canónico, las mismas características escaladas con el mismo StandardScaler ajustado solo sobre train, la misma partición exacta de test (las 566.149 filas del split seed 42, verificadas por hash SHA-256) y las mismas métricas calculadas con el mismo código. Lo único que cambia es el modelo. Si dejara que cada modelo usara un preprocesado distinto, un test distinto o métricas distintas, cualquier diferencia de resultados podría venir del montaje y no del método, y la comparación no demostraría nada.

> **Qué comprueba / qué no decir:** Verifica que no comparas peras con manzanas. NO cites como comparación el run C03 (test de 100.000 filas con mezcla de clases distorsionada) ni la Fase 2: esos NO están bajo el mismo protocolo y no son comparables con el run principal.


**G33. Si Random Forest obtiene resultados similares o mejores que QRDQN, ¿cómo debería interpretarse?**

Hay que interpretarlo con honestidad, y es justo lo que pasa en mi caso: en el split aleatorio RF me gana marginalmente (accuracy 0,99872 y F1 de ataque 0,99676 frente a 0,99381 y 0,98445 de QRDQN). Eso significa que, para esa tarea concreta y ese reparto fácil de los datos, un clasificador clásico iguala o supera al RL; no se puede vender el RL como necesario solo por ese resultado. Pero la lectura no acaba ahí: cuando aprieto la evaluación con un split por día (entrenar con unos días y testear con días y ataques no vistos), RF se desploma a recall de ataque 0,08135 mientras QRDQN aguanta en 0,52954; y en leave-one-out (reservando Wednesday) RF cae a F1 0,01427. Así que la interpretación correcta es: RF gana en el escenario optimista, pero ambos sufren ante el desajuste de distribución, y RF lo hace mucho peor. Reconocer que el baseline a veces gana es señal de rigor, no de fracaso del trabajo.

> **Qué comprueba / qué no decir:** Comprueba que no inflas tu propio método ni escondes que el baseline te gana. NO digas que QRDQN es claramente superior: en el split fácil pierdes. Tu argumento es la robustez relativa bajo cambio de distribución, no la victoria global.


**G34. ¿Sería válido decir que QRDQN es mejor solo porque usa aprendizaje por refuerzo?**

No, eso sería una falacia: usar una técnica más sofisticada o de moda no la hace mejor por sí misma. La única evidencia válida de que un método aporta es que gane bajo el mismo protocolo y en las métricas que importan, y en mi caso QRDQN ni siquiera gana en el split aleatorio (ahí pierde frente a RF). Además, con gamma=0,0 mi QRDQN es en realidad un bandit contextual sensible al coste, una decisión de un solo paso, no un MDP secuencial con planificación, así que tampoco puedo apelar a la 'potencia del RL secuencial'. Lo defendible es que QRDQN se mantiene mejor que RF cuando hay desajuste de distribución (split por día), y eso es un dato medido, no una creencia sobre la etiqueta 'RL'.

> **Qué comprueba / qué no decir:** Esta es una trampa directa contra el bombo metodológico. NO defiendas el RL por prestigio; reconoce que la superioridad debe demostrarse empíricamente y que, de hecho, en el caso fácil el baseline te supera.


**G35. ¿Qué ventajas prácticas podría tener Random Forest frente a QRDQN?**

Varias, y son razones reales por las que en producción mucha gente elegiría RF. Es mucho más simple y barato de entrenar: mi QRDQN principal necesitó 3.000.000 de pasos y unas 5h15m en una GPU RTX 3090 Ti, mientras que un Random Forest se ajusta en CPU en minutos. Tiene menos hiperparámetros que tunear y menos formas de fallar silenciosamente. Es más interpretable: puedes mirar la importancia de las características y entender en qué se apoya, algo valioso en seguridad para auditar decisiones. Y es invariante a la escala, lo que le quita una fuente de error en preprocesado. La contrapartida, que también debo decir, es que en mi experimento RF generaliza peor ante datos no vistos (se hunde en el split por día), así que la ventaja de simplicidad hay que sopesarla contra la robustez.

> **Qué comprueba / qué no decir:** Comprueba que no eres ciego a las ventajas del rival. Da ventajas concretas (coste de cómputo, interpretabilidad, simplicidad), no genéricas; y NO ocultes que esa simplicidad viene con peor generalización en tu caso.



### H · Evaluación y validación

**H36. ¿Qué problema tiene una partición aleatoria fila a fila en datasets de tráfico?**

El problema es que el tráfico de red no es independiente fila a fila: un mismo ataque o una misma sesión genera muchos flujos casi idénticos. Si repartes las filas al azar, copias muy parecidas de un mismo evento acaban a la vez en train y en test, así que el modelo no demuestra que generaliza, simplemente reconoce patrones que ya vio (es fuga de información por duplicación). Esto infla artificialmente las métricas: por eso en mi split aleatorio tanto QRDQN como RF sacan accuracies por encima de 0,99, números que parecen casi perfectos. Mi proyecto es consciente de ello e incluye un análisis de duplicados de solo lectura (analyze_duplicates.py) que cuantifica duplicados exactos y solape entre splits sin entrenar, precisamente para no engañarse con esos números optimistas.

> **Qué comprueba / qué no decir:** Verifica que no te crees tus propios 0,99. NO presentes el split aleatorio como prueba de que el sistema funciona en el mundo real; reconoce que es el escenario optimista y que los duplicados lo inflan.


**H37. ¿Qué aporta una partición temporal o por día?**

Aporta una evaluación mucho más honesta de la capacidad de generalizar. En lugar de repartir filas al azar, separas días completos: en mi Check C entreno con Monday/Tuesday/Wednesday y testeo con Thursday/Friday, y es un group split real con rechazo de solape (train y test no pueden compartir el mismo CSV). Así el test contiene días y tipos de ataque que el modelo nunca vio, que es justo lo que pasa en producción cuando aparece tráfico nuevo. El resultado es revelador y mucho más bajo: la accuracy cae de 0,99 a 0,84135 y el recall de ataque a 0,52954. Esa caída no es un fallo del experimento, es la información valiosa: te dice cuánto del 0,99 anterior era memorización de patrones repetidos y cuánto era generalización real.

> **Qué comprueba / qué no decir:** Comprueba que entiendes por qué el número baja y que no lo escondes. NO maquilles la caída de recall del Check C; preséntala como el resultado realista y esperado, y cita el contraste con el split aleatorio.


**H38. ¿Qué evalúa dejar fuera un CSV completo?**

Es una versión aún más exigente de la idea anterior: la validación leave-one-CSV-out reserva un CSV entero (que corresponde a un día y a un conjunto concreto de ataques) como test y entrena con los demás, repitiendo el proceso para cada CSV. Evalúa si el modelo generaliza a una familia de tráfico completa que no vio en absoluto durante el entrenamiento, eliminando casi por completo la posibilidad de fuga por flujos repetidos entre train y test. En mi proyecto está implementado como capacidad del pipeline (validate_leave_one_csv_out.py, un fold por cada uno de los 8 CSV, 30.000 timesteps por fold), pero el artefacto de la batería completa para QRDQN todavía no está generado ni commiteado, así que lo presento como capacidad técnica y no con métricas. Donde sí tengo dato es en RF con Wednesday reservado, y el resultado es muy malo (accuracy 0,63782, F1 0,01427), lo que refuerza que generalizar a un día entero no visto es duro.

> **Qué comprueba / qué no decir:** Verifica que no inventas métricas LOO de QRDQN. Di explícitamente que la batería completa de QRDQN está pendiente de generar/commitear y que el único LOO con número es el de RF (Wednesday). NO atribuyas resultados LOO a QRDQN.


**H39. ¿Qué significa desplazamiento de dominio (domain shift)?**

El desplazamiento de dominio es cuando los datos sobre los que usas el modelo siguen una distribución distinta de aquellos con los que lo entrenaste: cambia el 'mundo' de los datos. En detección de intrusiones ocurre constantemente porque la red de otra empresa, otro periodo de tiempo o ataques nuevos producen flujos con estadísticas diferentes a las de CICIDS2017. Cuando hay domain shift, un modelo que parecía excelente se degrada, y es exactamente lo que veo: del split aleatorio (recall de ataque ~0,99) al split por día (recall 0,52954 en QRDQN, y un colapso a 0,08 en RF). También es la razón de ser de mi máscara de ausencia, que solo se vuelve informativa cuando el dominio cambia y faltan columnas, y del recorte z-score en Fase 2, donde he observado características con |z|=89 por estar fuera de distribución.

> **Qué comprueba / qué no decir:** Comprueba que conectas el concepto con tus propios resultados, no que recitas una definición de manual. Menciona la caída concreta de métricas entre splits como evidencia de que el domain shift no es teórico en tu trabajo.


**H40. ¿Por qué la validación con tráfico de laboratorio no equivale a una validación en producción?**

Porque mi laboratorio es un entorno cerrado y controlado, no una red real con su diversidad y su adversario activo. El tráfico de Fase 2 es una captura de un laboratorio doméstico aislado, generada a mano por consola, y sus etiquetas son de intención del operador (yo digo qué es benigno o ataque), no el veredicto de un detector independiente; por eso una puntuación alta ahí es una comprobación dentro o cerca de la distribución, con validez externa limitada. Una red de producción tiene volumen, variedad de aplicaciones, fallos, tráfico ambiguo y atacantes que se adaptan, nada de lo cual está representado en mi laboratorio. La prueba está en lo inestable que es la Fase 2: artefactos solo-benignos que dan block_rate 1,0 en un caso y 0,0 en otro. Por eso nunca digo que la Fase 2 esté 'resuelta': el desajuste de dominio es el principal problema abierto del trabajo.

> **Qué comprueba / qué no decir:** Esta pregunta busca que no sobrevendas. NO presentes los buenos números de Fase 2 (accuracy 0,991) como validación en producción; reconoce que las etiquetas son de intención propia, el entorno es cerrado y la validez externa es limitada.



### I · Discusión crítica

**I41. ¿Cuál es la principal debilidad metodológica del TFM?**

La principal debilidad es que el modelo oficial se entrenó con un solo seed de entrenamiento (seed 42) y sobre una única partición de test fija. Eso significa que no puedo separar cuánto del resultado se debe al método y cuánto a una inicialización afortunada de la red. Los intervalos de confianza que reporto (por ejemplo, recall de ataque 0.99536 con IC [0.99495, 0.99575]) son estrechos, pero solo miden el ruido de remuestrear el conjunto de test de ese único modelo; NO capturan la variabilidad entre semillas de entrenamiento. La forma rigurosa de cerrar esto sería reentrenar con varias semillas y reportar media y desviación. A esto se suma que el resultado principal sale de un split aleatorio fila a fila, que en datos de tráfico tiende a inflar las métricas por flujos casi duplicados repartidos entre train y test.

> **Qué comprueba / qué no decir:** El tutor comprueba si confundes 'IC estrecho' con 'resultado estable'. No digas que los IC demuestran robustez del entrenamiento: solo cuantifican el remuestreo del test de un único modelo entrenado con un solo seed.


**I42. ¿Qué afirmación sería excesiva o no justificada a partir de este trabajo?**

Sería excesivo decir que el sistema es un defensor de red autónomo listo para producción, o que detecta ataques 'con un 99% de fiabilidad' en el mundo real. El 99.381% de accuracy es sobre CICIDS2017 con split aleatorio, un escenario cómodo; en cuanto fuerzo generalización a días y ataques no vistos (Check C), el recall de ataque cae a 0.52954, y en laboratorio real el comportamiento es muy sensible al artefacto (hay runs que bloquean el 100% del tráfico benigno y otros que lo permiten al 100%). Tampoco puedo afirmar que QRDQN sea superior a un clasificador clásico: Random Forest lo iguala o supera en split aleatorio (accuracy 0.99872, F1 ataque 0.99676). Y como gamma=0.0, no puedo decir que el agente aprenda estrategias defensivas secuenciales: es una decisión de un solo paso por flujo. La afirmación defendible es mucho más modesta: ofrezco un pipeline reproducible con un buen resultado offline y una identificación honesta del problema abierto, el domain shift.

> **Qué comprueba / qué no decir:** El tutor busca si te crees tus propias cifras altas. Lo que NO conviene decir es 'el sistema funciona muy bien en general'; hay que acotar el resultado a su dominio exacto y reconocer el colapso fuera de él.


**I43. ¿Qué cambio harías si quisieras acercar el sistema a un escenario real?**

El cambio más importante sería entrenar y validar con particiones que respeten el tiempo (split por día o leave-one-CSV-out completo) en lugar de un split aleatorio fila a fila, porque el aleatorio reparte flujos casi idénticos entre train y test e infla el resultado; el escenario real es siempre 'predecir mañana con lo aprendido hoy'. En segundo lugar, cerrar el lazo de enforcement real: hoy el modelo solo escribe la etiqueta allow(0)/block(1) en un CSV y ningún punto de aplicación la consume; haría falta el bloqueo inline (iptables/nftables) como prototipo controlado. En tercer lugar, atacar el domain shift de la Fase 2 con calibración o reentrenamiento sobre tráfico real capturado, porque las distribuciones del laboratorio difieren mucho de CICIDS2017 (se han visto z-scores de hasta |89| en contadores de flags TCP). Y operativamente, medir latencia y throughput: bloquear en tiempo real impone restricciones que un experimento offline no captura.

> **Qué comprueba / qué no decir:** El tutor quiere ver que entiendes la diferencia entre métrica de laboratorio y operación real. No basta con decir 'más datos'; lo que prueba es si sabes que el split temporal y el enforcement real son los cambios estructurales que faltan.


**I44. ¿Qué información se pierde al convertir todas las etiquetas a benigno/ataque?**

Al binarizar (regla: todo lo que no sea BENIGN cuenta como ataque, etiqueta 1) se pierde la familia y la gravedad del ataque: un PortScan, un DDoS, un intento de fuerza bruta SSH o una infiltración acaban todos como '1', indistinguibles para el modelo y para las métricas. Esto tiene tres consecuencias concretas. Primero, no puedo saber si el recall alto se reparte por igual entre todas las familias o si el modelo es ciego a las clases raras (un ataque minoritario puede estar oculto dentro del 99.536% global de recall). Segundo, pierdo la capacidad de dar respuestas defensivas distintas según el tipo de amenaza, algo que un defensor real necesita. Tercero, no puedo diagnosticar QUÉ tipo de ataque falla en el Check C, donde el recall cae a 0.52954. La binarización simplifica el problema y lo hace tratable, pero a costa de ocultar el rendimiento por clase, que es justo lo que más importaría en seguridad.

> **Qué comprueba / qué no decir:** El tutor comprueba si has pensado en el desbalance por familia, no solo en el binario. No digas 'no se pierde nada relevante': lo grave es que el recall agregado puede esconder ceguera total ante un tipo de ataque concreto.


**I45. ¿Qué tipo de ataque o escenario podría no estar bien representado en este diseño?**

Hay varios huecos. Primero, los ataques que se detectan precisamente por los campos que elimino por anti-fuga (IPs, puerto de destino, timestamps): quito el puerto de destino a propósito para que no actúe como atajo a la etiqueta, pero eso significa que un ataque que en la práctica se reconoce por su puerto característico queda peor representado. Segundo, los ataques lentos o multi-flujo (escaneos sigilosos, exfiltración gota a gota, campañas que se despliegan a lo largo del tiempo): como el diseño es un bandit contextual con gamma=0.0, cada flujo se juzga de forma aislada, sin memoria del historial, así que no puede capturar patrones que solo emergen en la secuencia. Tercero, los ataques cifrados o que imitan tráfico legítimo a nivel de estadísticas de flujo, porque solo veo metadatos del flujo, no el contenido (inspección profunda de paquetes). Y cuarto, cualquier ataque ausente o infrarrepresentado en CICIDS2017, dataset de 2017 con familias concretas: lo nuevo (zero-day) queda fuera por construcción.

> **Qué comprueba / qué no decir:** El tutor verifica si entiendes que gamma=0 te impide modelar ataques secuenciales y que el anti-fuga tiene un coste. No presentes el anti-leakage solo como virtud: reconoce que también elimina señal legítima para ciertos ataques.



### J · Preguntas de control (detectar uso superficial de IA)

**J46. Señala una decisión metodológica del TFM con la que no estés completamente de acuerdo y justifica por qué.**

La decisión que más cuestiono es haber elegido el split aleatorio estratificado 80/20 como resultado oficial del proyecto. Lo entiendo como punto de partida y para comparar de forma justa con Random Forest bajo el mismo protocolo, pero en datos de tráfico el split aleatorio reparte flujos casi duplicados entre train y test, lo que infla las métricas y mide poca generalización real. Lo veo claramente en mis propios números: en split aleatorio QRDQN da 0.99381 de accuracy, pero en el split duro por día (Check C) el recall de ataque se hunde a 0.52954, y Random Forest pasa de 0.99853 de recall a 0.08135. Esa caída enorme es la prueba de que el split aleatorio sobreestima. Si lo rehiciera, presentaría el split por día o el leave-one-CSV-out como resultado principal, y el aleatorio como referencia secundaria, no al revés. La decisión es defendible por comparabilidad, pero no estoy del todo de acuerdo con darle el peso de 'resultado oficial'.

> **Qué comprueba / qué no decir:** Pregunta de control anti-IA: quiere una crítica propia y específica, no una genérica. No respondas con un defecto inventado; usa la tensión real entre 0.99381 (aleatorio) y 0.52954 (Check C) que ya está en tus artefactos.


**J47. ¿Qué parte del TFM te parece más débil y cómo la mejorarías?**

La parte más débil es la Fase 2, la inferencia sobre tráfico real de laboratorio. Es débil por dos motivos concretos. Uno: las etiquetas de verdad son etiquetas de intención del operador (yo digo qué tráfico generé como benigno o malicioso), no el veredicto de un detector independiente, así que un acierto alto es una comprobación casi in-distribution, con validez externa limitada. Dos: el resultado no es estable; tengo artefactos benign-only que bloquean el 100% del tráfico legítimo y otros que lo permiten al 100%, lo que demuestra una sensibilidad fuerte al preprocesado (percentile clipping, z-score clipping) y al artefacto exacto. La mejoraría de dos formas: capturando un conjunto de validación etiquetado de forma independiente y más variado, y calibrando explícitamente el modelo para el dominio del laboratorio (reajuste del scaler o fine-tuning con una pequeña muestra real), de modo que el comportamiento deje de oscilar entre extremos. Mientras eso no esté, no debo presentar la Fase 2 como resuelta.

> **Qué comprueba / qué no decir:** El tutor comprueba que no maquillas la Fase 2. No digas que la Fase 2 'ya funciona': lo honesto es que el domain shift sigue abierto y que los resultados benign-only se contradicen entre sí.


**J48. Explica una limitación que no sea solo 'faltan más datos'.**

Una limitación que no es de cantidad de datos sino de formulación: el problema está planteado como un bandit contextual con gamma=0.0, es decir, cada flujo se decide de forma aislada y el factor de descuento es irrelevante. Esto significa que, aunque uso la maquinaria de QRDQN (red objetivo, replay buffer), el término de bootstrap se anula y NO hay asignación temporal de crédito: el agente no aprende ninguna estrategia que dependa del orden o del historial de flujos. Por tanto, llamarlo 'aprendizaje por refuerzo secuencial' o 'defensa autónoma' sería incorrecto; honestamente es una formulación cost-sensitive de un solo paso. La limitación de fondo es que un defensor real opera en un entorno que SÍ reacciona a sus acciones (el atacante cambia de táctica cuando lo bloqueas), y mi entorno estático no modela esa dinámica adversaria. Eso no se arregla con más datos; requiere cambiar la formulación a un MDP real con un entorno que responda.

> **Qué comprueba / qué no decir:** Pregunta de control anti-IA: penaliza la muletilla 'faltan datos'. Debes nombrar una limitación estructural —gamma=0 / bandit, no MDP secuencial, sin adversario reactivo— y reconocerla sin disfrazarla de RL secuencial.


**J49. Si tuvieras que eliminar una sección por redundante, ¿cuál sería y por qué?**

Eliminaría o reduciría drásticamente la sección dedicada al tuning con Optuna (src/tune_hparams.py). Es la candidata clara por dos razones. Primero, no aporta nada al resultado oficial: los hiperparámetros del run principal son un perfil fijo puesto a mano y congelado por tests, no una salida de Optuna; de hecho el espacio de búsqueda de Optuna ni siquiera contiene la configuración final (busca gamma en [0.95, 0.999] mientras el run usa gamma=0.0, y net_arch [1024,1024,512] no está en el espacio). Segundo, hay una inconsistencia conceptual: el tuner busca en régimen de RL secuencial mientras el entrenador de producción es un bandit, así que esos gamma no son transferibles. Como mucho lo mencionaría como exploración inicial descartada, en una frase honesta, en lugar de darle una sección propia que sugiere una optimización sistemática que no respalda el resultado final. El resto del pipeline (datos, entorno, validación A/B/C, Fase 2) no es redundante: cada pieza sostiene una afirmación distinta.

> **Qué comprueba / qué no decir:** El tutor comprueba si conoces la incoherencia Optuna vs gamma=0. No defiendas Optuna como si hubiera producido la config oficial; lo honesto es reconocer que es exploración descartada y no transferible.


**J50. ¿Qué resultado experimental te haría cambiar la interpretación del trabajo?**

Varios resultados me harían replantear la interpretación. El más directo: si al reentrenar con varias semillas distintas las métricas variaran mucho (por ejemplo, el recall de ataque oscilando varios puntos entre seeds), tendría que admitir que el 0.99536 es en parte suerte de una semilla y no una propiedad del método. Segundo, si una validación por día o leave-one-CSV-out completa y bien hecha mostrara un colapso generalizado (como ya insinúa el recall 0.52954 del Check C), entonces el mensaje pasaría de 'buen rendimiento' a 'el modelo memoriza el dataset y no generaliza'. Tercero, si Random Forest, además de igualar a QRDQN en split aleatorio, lo igualara también en los splits duros, perdería sentido justificar el coste extra del RL distribucional. Y cuarto, en Fase 2, si con un etiquetado independiente y robusto el modelo siguiera oscilando entre bloquear todo y permitir todo, tendría que concluir que el pipeline no transfiere a tráfico real en su estado actual. En resumen, la interpretación 'tengo un buen defensor offline' aguanta solo mientras la generalización dura y la estabilidad entre seeds no la desmientan.

> **Qué comprueba / qué no decir:** El tutor verifica que tu interpretación es falsable, no fe ciega en tus números. No digas 'nada me haría cambiar de opinión'; nombra resultados concretos (varianza entre seeds, colapso en split temporal, RF ganando en split duro) que invalidarían tu tesis.



### Preguntas estrella · las más diagnósticas

**S1. Explícame con un ejemplo concreto qué sería un falso positivo y un falso negativo en tu sistema.**

Mi sistema ve un flujo de red (un resumen estadístico de una conexión) y decide PERMIT (dejar pasar, acción 0) o BLOCK (bloquear, acción 1). Un FALSO POSITIVO es bloquear tráfico que en realidad era legítimo: por ejemplo, un empleado se descarga un fichero grande del servidor interno, eso genera un flujo benigno, y el agente lo marca como ataque y lo bloquea (BLOCK sobre un BENIGN). En mi run oficial sobre el test de 566.149 flujos hubo 2.989 falsos positivos de 454.620 benignos (FPR 0.00658). Un FALSO NEGATIVO es lo contrario: un flujo que sí es un ataque (por ejemplo un escaneo de puertos o un intento de DDoS) y el agente lo deja pasar (PERMIT sobre un ATTACK); tuve 518 de 111.529 ataques (FNR 0.00465). La asimetría es deliberada: el falso negativo es el error grave porque deja entrar la amenaza, mientras que el falso positivo solo molesta a un usuario legítimo.

> **Qué comprueba / qué no decir:** El tutor comprueba que sé traducir las dos acciones (PERMIT/BLOCK) a las dos etiquetas (BENIGN/ATTACK) sin confundir el sentido. El error típico es cruzarlos. NO digo 'falso positivo = ataque detectado'; positivo en mi convención es la clase ataque (1=BLOCK), así que FP = benigno bloqueado y FN = ataque permitido. Doy los números reales (2.989 y 518) para que sea verificable.


**S2. ¿Por qué dices que usas aprendizaje por refuerzo si realmente entrenas sobre un dataset etiquetado y estático?**

Es la pregunta más honesta y la respondo de frente: técnicamente uso la maquinaria del aprendizaje por refuerzo (un agente QRDQN, un entorno Gymnasium con reset/step, recompensas, replay buffer, red objetivo), pero con gamma=0.0 esto NO es un MDP secuencial sino un bandit contextual, es decir, una decisión de un solo paso por cada flujo. Lo que aprende el agente no es entropía cruzada sobre etiquetas como un clasificador normal: aprende el VALOR de cada acción sobre una matriz de costes asimétrica explícita (falso negativo -5.0, falso positivo -2.0, acierto en ataque +1.5, benigno permitido 0.0), y decide tomando el argmax del valor entre PERMIT y BLOCK. La diferencia frente a un clasificador es dónde vive el coste: aquí está en la recompensa, no en un class_weight, y la cabeza distribucional de QRDQN modela toda la distribución del retorno, no solo su media. Así que la respuesta honesta es: uso el FRAMEWORK de RL para resolver un problema cost-sensitive de un paso, y no afirmo que sea una defensa secuencial autónoma.

> **Qué comprueba / qué no decir:** El tutor está verificando si entiendo que mi gamma=0 colapsa el RL a algo de un solo paso, o si lo vendo como RL secuencial real. La trampa es presumir de 'aprendizaje secuencial' o 'asignación temporal de crédito': NO existe, porque el siguiente flujo es independiente de mi acción anterior. Reconozco abiertamente que es un bandit contextual; eso es más fuerte que fingir lo contrario.


**S3. ¿Qué tendría que pasar para que Random Forest fuese una opción metodológicamente preferible a QRDQN?**

Random Forest YA gana a QRDQN en el split aleatorio: 0.99872 de accuracy y F1 de ataque 0.99676 frente a mi 0.99381 y 0.98445. Si el objetivo se quedara solo en clasificar bien CICIDS2017 con partición aleatoria, RF sería la opción preferible por ser más simple, más rápido de entrenar, interpretable (importancia de variables) y sin GPU. Sería metodológicamente preferible cuando: (1) el problema es realmente de un solo paso y estático, sin intención futura de añadir secuencialidad ni varias acciones; (2) no necesito modelar la distribución del retorno ni el riesgo, solo la clase; y (3) priorizo coste computacional e interpretabilidad. PERO hay un matiz decisivo: en el split por día RF colapsa con recall de ataque 0.08135 (deja pasar el 92% de los ataques no vistos) y en leave-one-out con Wednesday baja a F1 0.01427, mientras QRDQN aguanta mejor (recall 0.52954 en Check C). Así que RF es preferible para el caso fácil, pero su fragilidad ante días/ataques nuevos justifica explorar el enfoque RL como puente hacia un sistema que algún día decida secuencialmente.

> **Qué comprueba / qué no decir:** El tutor mide si soy capaz de admitir que mi baseline gana en el caso estándar, en vez de defender QRDQN ciegamente. La trampa es decir 'QRDQN es mejor porque usa RL': falso en el split aleatorio. Lo honesto es reconocer la victoria de RF ahí y apoyarme en el único terreno donde QRDQN aguanta mejor (generalización dura por día), sin exagerar ese 0.52954 como si fuera bueno; es solo 'menos malo' que el 0.08135 de RF.


**S4. ¿Dónde podría colarse fuga de información aunque hayas quitado las columnas más obvias?**

Mi política anti-fuga elimina lo evidente (Flow ID, Timestamp, todas las IPs y los puertos de origen y destino, este último porque ciertos ataques usan puertos fijos y el puerto sería un atajo a la etiqueta). Pero la fuga puede colarse de formas más sutiles. Primero, CICIDS2017 tiene duplicados exactos y casi-duplicados de flujos: en un split aleatorio fila a fila, copias casi idénticas de un mismo ataque caen a la vez en train y en test, así que el modelo 'memoriza' en lugar de generalizar, e infla la métrica sin ser fuga de columna sino fuga de partición. Segundo, hay variables que son proxies estadísticos de la etiqueta: ciertas estadísticas de flujo (tamaños, tasas, conteos de flags TCP) tienen valores muy característicos de un ataque concreto del laboratorio CICIDS2017, así que correlacionan con la etiqueta por artefacto del montaje, no por causa real. Tercero, el orden temporal: si no controlas por día, el modelo aprende el 'régimen' de un día en que solo se lanzó un ataque. Por eso tengo el split por día (Check C) y un analizador de duplicados de solo lectura: precisamente para cuantificar y mitigar la fuga que no se ve quitando columnas.

> **Qué comprueba / qué no decir:** El tutor comprueba si entiendo que la fuga no es solo 'columnas chivatas' sino también fuga por partición (duplicados train/test) y proxies estadísticos. La trampa es responder solo con la lista de columnas eliminadas y darme por satisfecho. Lo que demuestra comprensión es nombrar los duplicados de CICIDS2017 y reconocer que mi accuracy 0.99381 en split aleatorio probablemente está inflada por eso, lo cual conecta con la caída brutal en Check C.


**S5. ¿Qué NO puedes afirmar con tus resultados?**

No puedo afirmar que mi sistema funcione en una red empresarial o de producción real: solo lo he validado en CICIDS2017 (un dataset de laboratorio de 2017 con problemas conocidos) y en una captura de un laboratorio doméstico aislado y no adversarial, con validez externa limitada. No puedo afirmar estabilidad de entrenamiento: entrené UN solo seed (42), y aunque tengo intervalos de confianza estrechos por bootstrap (recall de ataque 0.99536 con IC [0.99495, 0.99575]), esos IC solo miden la precisión del remuestreo del test para ESE modelo, NO la variabilidad si reentrenase con otras semillas. No puedo afirmar que sea una defensa autónoma secuencial: con gamma=0.0 es un bandit contextual de un paso, no toma decisiones encadenadas en el tiempo. No puedo presentar C03 (accuracy 0.99859) como mi mejor resultado: es un probe previo al diseño sobre un test de 100.000 filas con mezcla de clases distorsionada, no comparable. Y no puedo afirmar que QRDQN sea superior a Random Forest en general: RF me gana en el split aleatorio.

> **Qué comprueba / qué no decir:** El tutor verifica madurez metodológica: si reconozco los límites yo mismo o si sobrevendo. La trampa es presentar los IC estrechos como prueba de robustez: NO lo son, porque un solo seed no captura variabilidad de entrenamiento. Tampoco debo decir que generaliza a tráfico real ni colar C03 como resultado oficial. Listar lo que NO puedo afirmar es justo lo que demuestra que entiendo el alcance real.


**S6. Si un tribunal te dice que esto es clasificación disfrazada de RL, ¿cómo responderías?**

Respondería: tienen razón en parte, y lo asumo abiertamente. Con gamma=0.0 mi formulación es honestamente un bandit contextual, una decisión cost-sensitive de un solo paso, no un MDP secuencial; no hay transiciones reales porque el siguiente flujo es independiente de mi acción. Pero NO es un clasificador estándar disfrazado, y la diferencia es real en tres puntos. Uno: no minimizo entropía cruzada sobre etiquetas; aprendo el valor de cada acción sobre una matriz de costes asimétrica explícita (FN -5.0, mucho peor que FP -2.0), de modo que el coste de seguridad vive en la recompensa, no escondido en un class_weight. Dos: la decisión es el argmax del valor aprendido entre PERMIT y BLOCK. Tres: la cabeza distribucional de QRDQN (200 cuantiles) modela toda la distribución del retorno, no solo la media, lo que aporta información de riesgo en la frontera de decisión. Y el valor metodológico está en haber montado el framework completo (entorno, recompensa, agente) de forma que, cambiando gamma o añadiendo más acciones, el mismo andamiaje escala hacia un problema realmente secuencial sin rehacer el pipeline.

> **Qué comprueba / qué no decir:** El tutor pone una objeción cierta para ver si me derrumbo o si la integro con honestidad técnica. La trampa es negarlo en redondo ('no, esto es RL puro') porque es indefendible con gamma=0. La respuesta ganadora reconoce que es un bandit contextual cost-sensitive y explica por qué eso aun así no es entropía cruzada: el coste en la recompensa, el argmax de valor y la cabeza distribucional. Conceder lo justo y delimitar lo que sí es propio del RL.


**S7. ¿Por qué la máscara de ausencia no es un detalle técnico menor?**

La máscara de ausencia (missingness) son las 76 dimensiones extra que completan mi observación de 152: para cada una de las 76 características, m_i=1 si el valor estaba presente y era finito en el flujo, y m_i=0 si estaba ausente, era NaN o infinito y tuve que imputarlo con 0.0. No es menor porque es lo que distingue 'esta característica vale cero' de 'esta característica no la pude medir'. En CICIDS2017 nativo la máscara es constante a 1 (el mapeo cubre las 76), así que ahí no aporta nada, y lo reconozco. Pero su razón de ser es la transición a otros dominios: con NSL-KDD, por ejemplo, el mapeo solo cubre 3 de las 76 características, así que la máscara queda casi toda a 0 y le dice al agente 'estos 73 ceros son ausencia de medición, no actividad nula'. Sin la máscara, un cero imputado por ausencia y un cero real serían indistinguibles, y el agente aprendería patrones falsos en cuanto cambie el extractor de flujos o el dataset. Es decir, es la pieza que hace el contrato de observación honesto y reutilizable entre datasets distintos, que es justo la aportación de diseño del TFG.

> **Qué comprueba / qué no decir:** El tutor comprueba dos cosas: si entiendo PARA QUÉ sirve la máscara y si soy honesto en que en CICIDS2017 NO aporta. La trampa es presumir de que la máscara mejora mi resultado oficial: no lo hace, es constante a 1 ahí. Lo correcto es admitir eso y defender su valor en el cambio de dominio (Fase 2 / NSL-KDD), donde codifica presencia de columna de origen, no de valor.


**S8. ¿Qué diferencia hay entre generalizar a otro CSV de CICIDS2017 y generalizar a tráfico real?**

Generalizar a otro CSV de CICIDS2017 (lo que mide mi Check C y la validación leave-one-CSV-out) es generalizar DENTRO del mismo laboratorio, el mismo extractor de flujos (CICFlowMeter), el mismo montaje de 2017 y los mismos tipos de ataque, solo que a días distintos: entreno con Monday/Tuesday/Wednesday y testeo en Thursday/Friday. Ahí ya sufro (el recall de ataque cae a 0.52954), pero sigo en el mismo dominio estadístico. Generalizar a tráfico real es un salto mucho mayor: cambia la red, los servicios, las versiones de software, el extractor de flujos, las unidades de tiempo, e incluso la noción de qué es 'normal'. Eso es domain shift, y se nota: en mi laboratorio doméstico he visto z-scores de conteos de flags TCP de hasta |z|=89 (valores fuera de distribución que ni aparecen en CICIDS2017), y artefactos de Fase 2 sobre tráfico solo-benigno que oscilan entre bloquear el 100% y bloquear el 0%. Por eso un buen resultado entre CSVs no demuestra robustez en producción: el primero cambia el muestreo dentro de una distribución, el segundo cambia la distribución entera. El domain shift sigue siendo mi principal problema abierto.

> **Qué comprueba / qué no decir:** El tutor verifica que no equiparo 'otro CSV' con 'el mundo real'. La trampa es decir que mi validación por día ya prueba generalización a producción: no, solo prueba generalización intra-laboratorio. Demuestro comprensión nombrando lo que cambia de verdad (extractor, unidades, distribución) y citando evidencia concreta de inestabilidad en Fase 2 (z=89, block_rate 1.0 vs 0.0), sin vender la Fase 2 como resuelta.


**S9. ¿Qué cambiaría si en vez de dos acciones tuvieras varias acciones defensivas posibles?**

Hoy mi espacio de acción es Discrete(2): solo PERMIT o BLOCK. Si tuviera varias acciones defensivas (por ejemplo: permitir, limitar la tasa, redirigir a una sandbox/cuarentena, pedir reautenticación, bloquear), cambiarían tres cosas. Primero, la matriz de recompensas dejaría de ser 4 celdas (tp/fp/fn/omission) y pasaría a ser una matriz coste-acción mucho más rica: tendría que asignar coste a respuestas intermedias, por ejemplo que limitar un ataque sea mejor que permitirlo pero peor que bloquearlo, y que poner en cuarentena a un usuario legítimo moleste menos que bloquearlo. Segundo, aquí es donde la formulación RL empezaría a ganarle de verdad a un clasificador y donde la cabeza distribucional de QRDQN sería más útil: con acciones graduales, conocer la distribución del retorno (el riesgo) y no solo su media ayuda a elegir respuestas proporcionales a la incertidumbre. Tercero, sería el momento natural de salir del bandit (gamma=0) hacia un MDP secuencial real con gamma>0, porque algunas respuestas (cuarentena, rate-limit) SÍ cambian el estado futuro de la red, y entonces la asignación temporal de crédito dejaría de ser irrelevante. Es decir, varias acciones es justo el escenario que convertiría esto de un cost-sensitive de un paso en una defensa secuencial donde el RL aporta lo que un Random Forest no puede.

> **Qué comprueba / qué no decir:** El tutor mira si entiendo por qué elegí RL pese a que con 2 acciones y gamma=0 parece un clasificador: la respuesta es que el RL es la arquitectura que escala a más acciones y a secuencialidad. La trampa es no conectar 'más acciones' con 'gamma>0 y MDP real'. Hay que mostrar que el andamiaje actual está pensado para ese salto, sin afirmar que ya lo hago: hoy son solo 2 acciones y un bandit.


**S10. Dime una decisión que hayas tomado por seguridad metodológica, aunque redujera el rendimiento aparente.**

La más clara: eliminar el puerto de destino del vector de características, aun sabiendo que subiría la accuracy si lo dejara. En CICIDS2017 varios ataques usan puertos fijos y característicos (por ejemplo escaneos o servicios concretos), así que el puerto correlaciona casi perfectamente con la etiqueta y actuaría como un atajo, un proxy de la respuesta: el modelo 'acertaría' memorizando el puerto en vez de aprender el comportamiento del flujo. Eso es fuga de información, no señal legítima, así que lo descarto a propósito junto con IPs, Flow ID y timestamps. Una segunda decisión del mismo tipo es el split por día (Check C) y el escalado ajustado SOLO sobre el train: el split por día me da un recall de ataque de 0.52954, que es feo comparado con el 0.99536 del split aleatorio, pero es la medida honesta de generalización a días y ataques no vistos, y prefiero reportar ese número incómodo antes que esconderme tras el split aleatorio inflado por duplicados. También entrené el Check B con etiquetas barajadas a propósito para demostrar que sin señal real mi accuracy cae al 0.4773 (cerca del baseline 0.5227), confirmando que no hay fuga.

> **Qué comprueba / qué no decir:** El tutor busca señal de criterio propio frente a optimizar la métrica a toda costa (y de paso detectar uso superficial de IA: quien copia respuestas no suele sacrificar rendimiento conscientemente). La trampa es dar una decisión cosmética. La respuesta fuerte cita un sacrificio cuantificado y real: quitar el puerto, o aceptar el recall 0.52954 del split por día como medida honesta en lugar del 0.99536 inflado del split aleatorio.


---

## Apéndice · Glosario «en cristiano»

**Flujo de red (network flow)**  
Es el resumen estadístico de una 'conversación' entre dos máquinas en la red: en lugar de mirar paquete a paquete, se agrupan todos los paquetes que van y vienen entre un origen y un destino y se calculan cifras como duración, número y tamaño de paquetes, velocidades, etc. El modelo no ve los paquetes en bruto, ve estos números.  
*Analogía:* Como resumir una llamada telefónica con 'duró 5 minutos, hablaron 200 frases, la mitad cada uno' en vez de transcribir cada palabra.  
*En el proyecto:* Cada fila del dataset es un flujo descrito por 76 características numéricas (FEATURES_CANON): longitudes de paquete, tasas, tiempos entre llegadas, flags TCP, etc. El agente decide PERMIT o BLOCK sobre un flujo a la vez.

**Inspección profunda de paquetes (DPI)**  
Técnica clásica de seguridad que abre y examina el contenido completo de cada paquete que circula por la red para buscar amenazas. Es potente pero costosa y choca con el cifrado.  
*Analogía:* Como un control de aduana que abre cada maleta y revisa todo lo que hay dentro, frente a solo leer la etiqueta.  
*En el proyecto:* El proyecto NO hace DPI: deliberadamente trabaja con estadísticas de flujo (datos agregados, no el contenido de los paquetes), lo que lo hace más ligero y compatible con tráfico cifrado. Es el enfoque alternativo al que se contrapone.

**Aprendizaje por refuerzo (RL)**  
Rama de la inteligencia artificial donde un programa aprende a tomar decisiones probando acciones y recibiendo premios o castigos, en vez de que le den las respuestas correctas de antemano. Aprende de las consecuencias.  
*Analogía:* Como entrenar a un perro con premios y correcciones: no le explicas las reglas, refuerzas lo que hace bien.  
*En el proyecto:* Se replantea la detección de ataques (un problema de clasificación) como un problema de RL: un agente 'defensor' aprende a permitir o bloquear tráfico según los premios/castigos que recibe.

**Agente / entorno / episodio**  
El agente es el que decide y aprende; el entorno es el mundo con el que interactúa y que le devuelve premios; un episodio es una tanda completa de interacción, de principio a fin.  
*Analogía:* Un jugador (agente) jugando una partida (episodio) a un videojuego (entorno).  
*En el proyecto:* El agente es QR-DQN; el entorno es RLDatasetDefenderEnv, que le va mostrando flujos uno a uno; un episodio recorre el dataset barajado hasta agotarlo o hasta llegar a max_steps_per_episode (min(10.000, tamaño del train)).

**Estado-acción-recompensa**  
El trío básico del aprendizaje por refuerzo: el estado es lo que el agente observa, la acción es lo que decide hacer, y la recompensa es el premio o castigo que recibe por esa decisión.  
*Analogía:* Ves el semáforo en rojo (estado), decides frenar (acción), evitas la multa (recompensa positiva).  
*En el proyecto:* Estado = las 152 características de un flujo; Acción = 0 (PERMIT/dejar pasar) o 1 (BLOCK/bloquear); Recompensa según la matriz de coste: acertar un ataque +1.5, falsa alarma -2.0, dejar pasar un ataque -5.0, dejar pasar tráfico bueno 0.0.

**Política (policy)**  
Es la 'estrategia' que ha aprendido el agente: la regla que, dado lo que observa, le dice qué acción tomar. Es el producto final del entrenamiento.  
*Analogía:* El manual de decisiones de un portero: dado quién llega a la puerta, le dejo pasar o no.  
*En el proyecto:* La política es MlpPolicy (una red neuronal). Tras entrenar, dado un flujo decide PERMIT o BLOCK eligiendo la acción de mayor valor aprendido.

**Bandit contextual**  
Un caso simplificado del aprendizaje por refuerzo donde cada decisión es independiente: ves una situación, eliges una acción, recibes el premio, y lo que hagas ahora no afecta a lo que verás después. No hay consecuencias encadenadas en el tiempo.  
*Analogía:* Una máquina tragaperras que te muestra una pista (contexto) antes de cada tirada: cada tirada es independiente de la anterior.  
*En el proyecto:* Es el encuadre real y honesto del proyecto: aunque usa toda la interfaz de RL, cada flujo es una decisión aislada (permit/block) sin relación con el siguiente. Por eso el factor de descuento gamma se fija a 0.0.

**MDP (Proceso de Decisión de Markov)**  
El modelo matemático clásico del aprendizaje por refuerzo secuencial: un mundo donde tus acciones cambian el estado siguiente, así que las decisiones de ahora tienen consecuencias futuras encadenadas.  
*Analogía:* Una partida de ajedrez: cada jugada cambia el tablero y condiciona tus opciones futuras.  
*En el proyecto:* El entorno se presenta formalmente como un MDP de Gymnasium, pero al ser un MDP de un solo paso (la siguiente observación no depende de la acción) se reduce en la práctica a un bandit contextual.

**On-policy vs off-policy**  
On-policy: el agente solo aprende de su comportamiento actual. Off-policy: puede aprender de experiencias pasadas o ajenas, almacenadas en memoria, reutilizándolas para entrenar.  
*Analogía:* On-policy es aprender solo de lo que haces hoy; off-policy es estudiar también partidas grabadas del pasado, tuyas o de otros.  
*En el proyecto:* QR-DQN es off-policy: guarda las transiciones pasadas en un replay buffer y las reutiliza muestreando minibatches para entrenar, en vez de usar solo la experiencia más reciente.

**Replay buffer (buffer de repetición)**  
Una memoria donde el agente va guardando sus experiencias pasadas (lo que vio, qué hizo y qué premio obtuvo) para luego reaprender de ellas repetidamente en vez de usarlas una sola vez y tirarlas.  
*Analogía:* Un cuaderno de jugadas pasadas que repasas una y otra vez para entrenar.  
*En el proyecto:* Es el ReplayBuffer de QR-DQN. Su capacidad (buffer_size) es 1.000.000 en el run principal y se llena durante los primeros 50.000 pasos (learning_starts) antes de empezar a entrenar.

**Muestreo de minibatches**  
En lugar de aprender de todas las experiencias a la vez (lento) o de una sola (inestable), el agente coge al azar un pequeño lote de ejemplos de su memoria en cada paso de entrenamiento.  
*Analogía:* Estudiar para un examen sacando al azar un puñado de fichas del taco en cada sesión, en vez del taco entero.  
*En el proyecto:* Cada actualización muestrea un minibatch del replay buffer. El batch_size es 2048 en el run principal. Coger ejemplos variados al azar 'decorrela' las actualizaciones y estabiliza el aprendizaje.

**Epsilon-greedy y exploración vs explotación**  
Dilema de todo agente que aprende: explotar (usar lo que ya sabe que funciona) o explorar (probar cosas nuevas por si son mejores). Epsilon-greedy lo resuelve eligiendo al azar una fracción epsilon de las veces y la mejor acción conocida el resto.  
*Analogía:* En un restaurante, casi siempre pides tu plato favorito (explotar), pero de vez en cuando pruebas algo nuevo de la carta (explorar).  
*En el proyecto:* Epsilon arranca en 1.0 (todo exploración) y baja linealmente hasta 0.02 (exploration_final_eps) en el run principal, decayendo a lo largo del 10% del entrenamiento (exploration_fraction=0.10).

**Factor de descuento gamma**  
Número entre 0 y 1 que indica cuánto le importan al agente las recompensas futuras frente a la inmediata. Cerca de 1: piensa a largo plazo; en 0: solo le importa el premio inmediato.  
*Analogía:* Un tipo de interés al revés: cuánto valoras hoy lo que cobrarás mañana. En 0 solo cuenta lo de hoy.  
*En el proyecto:* gamma=0.0 en los tres regímenes. Es la decisión central del proyecto: anula el componente de futuro y convierte QR-DQN en un aprendiz de un solo paso (bandit contextual), donde cada decisión permit/block se evalúa por su premio inmediato.

**Red objetivo (target network) y actualización dura (hard update / tau)**  
Para no perseguir un blanco móvil al entrenar, se mantiene una copia 'congelada' de la red (red objetivo) que sirve de referencia estable. Cada cierto tiempo se actualiza copiando la red que aprende. Tau controla esa copia: 1.0 = copia completa de golpe (dura).  
*Analogía:* Apuntar a una diana fija un rato y solo moverla de vez en cuando, en vez de a una que se mueve a cada disparo.  
*En el proyecto:* Hay una red objetivo (quantile_net_target) con tau=1.0 (copia dura completa) cada target_update_interval pasos (10.000 en el run principal). Da estabilidad al objetivo de aprendizaje.

**Bootstrapping (en RL)**  
Técnica donde el agente estima el valor de una situación apoyándose en sus propias estimaciones de la situación siguiente, en vez de esperar al resultado final. Aprende 'a cuenta' de lo que él mismo predice.  
*Analogía:* Estimar cuánto tardarás en llegar usando tu propia estimación del tramo siguiente, sin esperar a llegar al destino.  
*En el proyecto:* Existe la maquinaria de bootstrap, pero con gamma=0.0 se anula: el objetivo se reduce a la recompensa inmediata (target = rewards) y la red objetivo no aporta nada. El bootstrap queda como un no-op.

**DQN**  
Algoritmo de aprendizaje por refuerzo que usa una red neuronal para estimar, dado un estado, lo buena que es cada acción posible (su valor Q). Elige la acción con mayor valor estimado.  
*Analogía:* Una tabla mental aprendida de 'en esta situación, esta jugada vale tanto' que actualizas con la experiencia.  
*En el proyecto:* DQN es la base sobre la que se construye QR-DQN. El proyecto usa la versión distribucional (QR-DQN), no DQN clásico, aunque hay un fallback a DQN.load solo si falla importar sb3_contrib.

**Valor Q**  
Es la calidad estimada de tomar una acción concreta en una situación concreta: cuánto premio total espera el agente obtener si elige esa acción. La acción elegida suele ser la de mayor valor Q.  
*Analogía:* La nota que le pones a cada opción de un menú según lo satisfecho que crees que quedarás.  
*En el proyecto:* Para decidir, QR-DQN colapsa su distribución de cuantiles en un valor Q medio por acción (mean sobre cuantiles) y elige el mayor (argmax) entre PERMIT y BLOCK.

**RL distribucional**  
Variante del aprendizaje por refuerzo donde el agente, en vez de aprender solo el premio medio esperado de cada acción, aprende la distribución completa de posibles resultados (la gama de premios y sus probabilidades).  
*Analogía:* En vez de saber solo 'de media saco un 7', conocer toda la campana de notas posibles: cuánto de probable es un 4, un 9, etc.  
*En el proyecto:* QR-DQN es distribucional: modela toda la distribución de retornos de cada acción mediante cuantiles, no solo la media. Es uno de los argumentos de por qué aporta más que un clasificador simple.

**Cuantiles y n_quantiles**  
Los cuantiles son puntos que cortan una distribución en trozos de igual probabilidad para describir su forma. n_quantiles es cuántos de esos puntos usa el modelo para representar la distribución de premios.  
*Analogía:* Describir el reparto de alturas de una población con 200 marcas (percentiles) en lugar de un único promedio.  
*En el proyecto:* n_quantiles=200 en los tres regímenes (valor del paper original de QR-DQN). Cada cuantil es un 'átomo' equiprobable de probabilidad 1/200 que aproxima la distribución de retornos de una acción.

**Pérdida quantile-Huber**  
La fórmula de error que el modelo intenta minimizar al aprender la distribución de premios. Combina la Huber loss (que castiga errores grandes con más suavidad, sin disparar) con una ponderación asimétrica propia de los cuantiles.  
*Analogía:* Una regla de penalización que no se vuelve loca con los errores enormes y además pesa de forma distinta quedarse corto o pasarse.  
*En el proyecto:* Se usa quantile_huber_loss con umbral kappa fijo a 1.0 (no ajustable), sumando sobre cuantiles como en el paper. El peso de asimetría usa delta.detach() para no aportar gradiente; el gradiente fluye solo por la parte Huber.

**Adam y learning_rate**  
Adam es el método que ajusta automáticamente los pesos de la red neuronal durante el entrenamiento. El learning_rate (tasa de aprendizaje) es el tamaño de cada paso de ajuste: muy grande inestable, muy pequeño lento.  
*Analogía:* Bajar una montaña en niebla: el learning_rate es lo largo que das cada paso; Adam es ir adaptando la zancada al terreno.  
*En el proyecto:* El optimizador es Adam con learning_rate=5e-5 en el run principal (más bajo que el 1e-4 de los runs cortos, para dar estabilidad a la corrida larga de 3M de pasos).

**Gradiente y descenso de gradiente**  
El gradiente indica en qué dirección hay que mover los pesos de la red para reducir el error. El descenso de gradiente es repetir pequeños pasos en esa dirección hasta que el modelo aprende.  
*Analogía:* Buscar el punto más bajo de un valle dando pasos cuesta abajo en la dirección de máxima pendiente.  
*En el proyecto:* En cada paso: se calcula la pérdida, se hace backward (calcula el gradiente), opcionalmente se recorta su norma (max_grad_norm=10.0 en el run largo) y se da el paso de Adam. Se repite gradient_steps veces (20 en el run principal).

**Normalización / z-score / StandardScaler**  
Poner todas las características en una escala comparable. El z-score resta la media y divide por la desviación típica, de modo que cada característica queda centrada en 0 con dispersión 1. StandardScaler es la herramienta de sklearn que lo hace.  
*Analogía:* Convertir notas de asignaturas con escalas distintas a una escala común para poder compararlas de forma justa.  
*En el proyecto:* Se usa StandardScaler ajustado SOLO sobre el train y aplicado al test (z-scoring por característica). El scaler se persiste en scaler.joblib para que coincida exactamente con lo que ven entorno y modelo.

**Partición estratificada**  
Al dividir los datos en train y test, se hace de modo que la proporción de cada clase (aquí benigno vs ataque) sea la misma en ambas partes. Así el test es representativo.  
*Analogía:* Repartir una bolsa de caramelos de dos sabores en dos cajas manteniendo la misma mezcla de sabores en cada caja.  
*En el proyecto:* Split estratificado aleatorio 80/20 (test_size=0.2, random_state=42). Resultado: train y test comparten tasas a 4 decimales (test_benign_rate 0.8030, test_attack_rate 0.1970).

**Fuga de información (data leakage)**  
Error grave en el que el modelo aprende, sin querer, pistas que no debería tener (atajos que delatan la respuesta), inflando los resultados de forma engañosa y haciendo que falle en el mundo real.  
*Analogía:* Un alumno que saca un 10 porque alguien dejó el examen resuelto a la vista: el resultado no mide lo que sabe.  
*En el proyecto:* Se combate activamente: se descartan Flow ID, timestamp, IPs y puertos (el puerto de destino actuaría como chivato de la etiqueta). Además el Check B entrena con etiquetas barajadas para confirmar que no hay fuga (resultó 0.4773, sin fuga).

**Matriz de confusión**  
Tabla que resume los aciertos y errores de un clasificador cruzando lo que era de verdad con lo que predijo: cuántos acierta y de qué tipo son sus fallos.  
*Analogía:* Una tabla de 'lo que dijiste vs lo que era' que separa los aciertos de los dos tipos de error.  
*En el proyecto:* Se construye con labels=[0,1] dando el orden (tn, fp, fn, tp). Run principal: tn=451.631, fp=2.989, fn=518, tp=111.011 sobre 566.149 flujos de test.

**TP / FP / TN / FN**  
Los cuatro resultados posibles: TP (verdadero positivo, ataque bien bloqueado), FP (falso positivo, falsa alarma sobre tráfico bueno), TN (verdadero negativo, tráfico bueno bien dejado pasar), FN (falso negativo, ataque que se cuela).  
*Analogía:* Alarma de incendios: suena con fuego real (TP), suena sin fuego (FP), calla sin fuego (TN), calla con fuego (FN, el peor).  
*En el proyecto:* Convención: 1=BLOCK/ataque=positivo, 0=PERMIT/benigno=negativo. En el run principal: TP=111.011, FP=2.989, TN=451.631, FN=518. El FN (ataque permitido) es el error más penalizado (-5.0).

**Precisión / recall / F1**  
Precisión: de lo que marqué como ataque, qué porcentaje lo era de verdad (mide falsas alarmas). Recall: de todos los ataques reales, qué porcentaje detecté (mide ataques perdidos). F1: combinación equilibrada de ambas en un solo número.  
*Analogía:* Pescar con red: precisión es qué porcentaje de lo que sacas son peces (no basura); recall es qué porcentaje de los peces del lago logras sacar.  
*En el proyecto:* Run principal, clase ataque: recall 0.99536 (solo se pierden 518 de 111.529 ataques), precision 0.97378, F1 0.98445. Se enfatiza el recall de ataque porque dejar pasar un ataque es lo más costoso.

**MCC (coeficiente de correlación de Matthews)**  
Una nota global de la calidad del clasificador entre -1 y +1 que tiene en cuenta los cuatro tipos de resultado a la vez. Es fiable incluso con clases desbalanceadas: 1 es perfecto, 0 es azar.  
*Analogía:* Una nota final que pondera bien todos los aspectos del examen, no solo los fáciles, y no se deja engañar si hay muchas más preguntas de un tipo.  
*En el proyecto:* MCC=0.98068 en el run principal. Es una métrica clave aquí porque las clases están desbalanceadas (~80% benigno, ~20% ataque) y el MCC no se infla por la clase mayoritaria.

**Accuracy balanceada**  
Variante de la exactitud que promedia el acierto en cada clase por separado, en vez de globalmente. Así una clase rara cuenta tanto como la frecuente y no queda enmascarada.  
*Analogía:* Nota media de dos asignaturas calculada por separado y promediada, para que la fácil no tape el suspenso en la difícil.  
*En el proyecto:* balanced_accuracy=0.99439 en el run principal. Importa porque con 80/20 una accuracy normal podría parecer alta solo acertando lo benigno; la balanceada evita ese espejismo.

**Intervalo de confianza bootstrap**  
Un rango que indica el margen de incertidumbre de una métrica. El método bootstrap lo estima remuestreando los datos de test muchas veces con repetición y viendo cómo varía el resultado.  
*Analogía:* Repetir un sondeo electoral mil veces sobre la misma muestra (con reemplazo) para ver entre qué dos valores baila el porcentaje.  
*En el proyecto:* IC del 95% (n_boot=10000, boot_seed=12345): recall de ataque [0.99495, 0.99575], F1 [0.98394, 0.98496]. Son estrechos, pero solo miden la precisión del resampling de UN modelo; no capturan variabilidad por seed de entrenamiento (solo se entrenó un seed).

**Desplazamiento de dominio (domain shift)**  
Cuando un modelo entrenado con unos datos se enfrenta a datos de otra procedencia o naturaleza, y rinde peor porque el mundo real no se parece al de entrenamiento.  
*Analogía:* Aprender a conducir siempre en ciudad y de pronto encontrarte en una carretera de montaña nevada: las reglas aprendidas ya no encajan igual.  
*En el proyecto:* Es el principal problema abierto en la Fase 2 (tráfico real del laboratorio). Se han observado z-scores de hasta |z|=89 en datos fuera de distribución, y artefactos solo-benignos divergen totalmente (block_rate 1.0 vs 0.0). Por eso la Fase 2 no está 'resuelta'.

**Random Forest**  
Modelo clásico de machine learning que combina muchos árboles de decisión entrenados sobre porciones distintas de los datos y vota entre ellos. Es robusto y un buen punto de comparación.  
*Analogía:* Pedir opinión a un bosque de expertos (cada árbol) y quedarse con el voto mayoritario, más fiable que un único experto.  
*En el proyecto:* Es el baseline supervisado (baseline_random_forest.py): n_estimators=200, class_weight='balanced', random_state=42. En split aleatorio supera ligeramente a QR-DQN (F1 0.99676), pero en split por día colapsa (recall de ataque 0.08135 frente al 0.52954 de QR-DQN).

**CICFlowMeter**  
Herramienta que lee el tráfico de red capturado (paquetes) y lo convierte en flujos con sus estadísticas numéricas, que es el formato que el modelo entiende.  
*Analogía:* Una máquina que coge la grabación en bruto de una conversación y la convierte en una ficha resumen con cifras.  
*En el proyecto:* Es el extractor que generó los 8 CSV oficiales de CICIDS2017 y la referencia para extraer flujos de PCAP real. Las 76 características canónicas se eligieron para ser extraíbles con CICFlowMeter o Zeek.

**SHA-256 / hash**  
Una huella digital de unos datos: una cadena de letras y números que cambia por completo si los datos cambian aunque sea un bit. Sirve para verificar que algo no se ha alterado y es idéntico a la referencia.  
*Analogía:* El sello único y a prueba de falsificaciones de un documento: si cambias una coma, el sello ya no coincide.  
*En el proyecto:* Se usa para fijar y verificar la integridad: el test_set_sha256 de referencia es cb175377...035b, y el artifact_manifest.json valida modelo, scaler y percentiles con checksums SHA-256 antes de usarlos.

**Seed / semilla**  
Un número que fija el punto de partida de todo lo que tiene azar en el programa (barajados, inicializaciones), para que ejecutar dos veces dé el mismo resultado. Es la base de la reproducibilidad.  
*Analogía:* Repartir las cartas siempre desde el mismo orden de mazo: el reparto parece aleatorio pero es idéntico cada vez.  
*En el proyecto:* La seed por defecto es 42. Siembra random, numpy, torch y CUDA. El split de datos usa su propio random_state interno, de modo que la siembra global no altera la partición fija seed-42.

