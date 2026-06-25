# QRDQN en Stable Baselines3 para el TFG

## Resumen ejecutivo

La implementación de **QR-DQN** en el ecosistema de Stable Baselines vive en **SB3-Contrib**, no en el núcleo de Stable Baselines3, y convierte a DQN en un método **distribucional**: en vez de aprender un único escalar \(Q(s,a)\), aprende una **distribución aproximada de retornos** por acción usando \(N\) cuantiles. En la práctica de SB3, esa distribución se representa como un tensor de forma \((\text{batch}, N, |\mathcal A|)\), se entrena con **quantile Huber loss**, y la acción codiciosa se elige por la **media de los cuantiles**, es decir, por el retorno esperado implícito en la distribución. Por tanto, QRDQN aprende más información que DQN, pero **no es risk-sensitive por defecto**: sigue actuando según la esperanza matemática salvo que se modifique explícitamente el criterio de decisión.

Desde la perspectiva teórica, QR-DQN nace para cerrar la brecha entre la teoría distribucional de Marc G. Bellemare, Will Dabney, y Rémi Munos en C51 y una implementación entrenable por descenso de gradiente. C51 fija los soportes \(z_i\) y aprende probabilidades; QR-DQN “transpone” esa idea: fija probabilidades uniformes y aprende las **localizaciones** de los cuantiles. Esto evita la proyección categórica de C51 y elimina la necesidad de fijar manualmente cotas como \(V_{\min}\) y \(V_{\max}\).

En tu repositorio `yeaight7/TFG_CYBER_AI`, QRDQN es el algoritmo principal para un problema discreto binario de ciberdefensa sobre **CICIDS2017**, con observaciones de **152 dimensiones** y acciones **0=PERMIT**, **1=BLOCK**. La arquitectura elegida es `MlpPolicy` con `net_arch=[512, 256]`, `learning_rate=1e-4`, `gamma=0.99`, `tau=1.0`, `train_freq` de 50 o 100 y `gradient_steps` de 10 o 20 según preset. El mejor artefacto consolidado del repositorio es el run `C03_qrdqn_cicids2017_canonical_full_random_20260223_232439`, con accuracy 0.99859 y F1 de ataque 0.99876 en random split; sin embargo, la validación dura por día/CSV cae de forma importante, lo que demuestra que la generalización real es bastante más difícil que el escenario aleatorio.

La conclusión práctica para tu TFG es clara: QRDQN está bien alineado con tu problema porque el espacio de acciones es discreto y porque el aprendizaje distribucional puede capturar mejor colas, multimodalidad y asimetrías del retorno inducidas por una recompensa con falsos negativos muy penalizados. Pero, en SB3, esa ventaja depende mucho de detalles de implementación: **número de cuantiles**, rapidez del decaimiento de \(\varepsilon\), tamaño del batch, frecuencia de actualización de la red objetivo y diseño de la recompensa. En tu repo, además, algunas decisiones se heredan por defecto de SB3-Contrib y no están declaradas explícitamente, lo que conviene dejar muy claro en la memoria para evitar ambigüedades metodológicas.

## Supuestos y lagunas del repositorio

Antes de entrar en teoría e implementación, conviene fijar varios supuestos y señalar qué información **no** queda completamente cerrada en el repo.

- **La versión exacta de SB3 y SB3-Contrib no está fijada**, solo un mínimo: `stable-baselines3>=2.3` y `sb3-contrib>=2.3`. Por eso, para reproducibilidad estricta, la interpretación de defaults debe hacerse respecto a la familia de versiones **2.3+**, no a una única release inmutable. Esto importa especialmente porque en SB3-Contrib v2.3.0 cambió el default de `learning_starts` de 50 000 a 100.
- **`n_quantiles` no se fija explícitamente en tu script de entrenamiento**, así que hay que inferir que usas el default oficial de la política QRDQN, que es **200 cuantiles**. Ese dato es metodológicamente importante porque determina el tamaño de la cabeza de salida, el coste computacional y la granularidad con la que aproximas la distribución.
- **Los parámetros de exploración tampoco se fijan explícitamente** en tu repositorio. Por tanto, se heredan los defaults de QRDQN en SB3-Contrib: `exploration_fraction=0.005`, `exploration_initial_eps=1.0` y `exploration_final_eps=0.01`. Esto tiene una consecuencia práctica importante: con 100 000 timesteps, la caída lineal de \(\varepsilon\) termina aproximadamente en los **primeros 500 pasos**; con 25 000 timesteps, en los **primeros 125 pasos**. Para una tesis, este detalle merece mención explícita porque afecta mucho al régimen de exploración real.
- **El script `train_rl_defender.py` contiene una inconsistencia documental interna**: en la ayuda de `--timesteps` se habla de un default de “500k full, 5k fast”, pero el código efectivo en `main()` asigna `25_000` para `fast` y `100_000` para `full`. En una defensa oral, esto conviene reconocerlo como una discrepancia de documentación del script, no como un resultado científico.
- **Los resultados históricos no coinciden necesariamente con los defaults actuales del código**. El reward actual por defecto usa `tp=1.5`, `fp=-2.0`, `fn=-5.0`, `omission=0.0`, que coincide con C03 y MAIN, pero no con todos los runs históricos anteriores. En tu memoria conviene separar con nitidez “**defaults del código actual**” frente a “**settings del artefacto reportado**”.
- **La validación leave-one-exact-CSV-out está implementada, pero no hay un artefacto agregado comprometido en `runs/validation/`**. Por tanto, esa parte debe presentarse como “workflow implementado” y no como resultado medido cerrado.

Como hallazgo sintético del repo: tu pipeline actual trabaja con CICIDS2017, un esquema canónico de **76 features**, una máscara de missingness de otras **76**, observación total de **152 dimensiones**, y un entorno Gymnasium con acción discreta binaria. El entrenamiento se hace con `DummyVecEnv` y `Monitor`, se normalizan las observaciones con `StandardScaler`, y la evaluación usa predicción determinista con matriz de confusión y métricas de clasificación.

## Fundamentos teóricos y matemáticos

La motivación de **distributional RL** es que el retorno futuro desde un estado-acción no es un número fijo, sino una **variable aleatoria**. DQN clásico aprende solo su expectativa \(Q(s,a)=\mathbb E[Z(s,a)]\). El enfoque distribucional, en cambio, intenta aproximar la ley completa de \(Z(s,a)\), lo que permite representar colas, multimodalidad y asimetría. El paper de C51 formaliza este punto y muestra que el operador de Bellman distribucional tiene propiedades teóricas distintas del Bellman escalar, aunque la estabilidad completa en control es más sutil. QR-DQN nace exactamente sobre esa base.

En C51, la distribución se aproxima con una combinación discreta de **soportes fijos** \(z_1,\dots,z_N\) y **probabilidades aprendidas** \(q_i\). QR-DQN invierte esa parametrización: fija probabilidades uniformes \(1/N\) y aprende las **localizaciones** \(\theta_i(s,a)\). Formalmente, la aproximación queda como

\[
Z_\theta(s,a)=\frac{1}{N}\sum_{i=1}^N \delta_{\theta_i(s,a)},
\]

donde cada \(\theta_i\) representa una localización cuantil y \(\delta\) es una masa de Dirac. La intuición es que, si los \(\theta_i\) se colocan en los cuantiles adecuados, la distribución aproximada se acerca a la real en distancia de Wasserstein.

La pérdida básica de regresión cuantil para un único cuantil \(\tau\) usa la llamada **pinball loss**:

\[
\rho_\tau(u)=u\bigl(\tau-\mathbf 1_{\{u<0\}}\bigr),
\]

y el paper de QR-DQN muestra que minimizar

\[
\mathbb E_{\hat Z\sim Z}\left[\rho_\tau(\hat Z-\theta)\right]
\]

recupera el cuantil correspondiente. Para varios cuantiles fijos en los puntos medios \(\hat\tau_i\), el problema se convierte en ajustar todas las localizaciones \(\theta_i\) por descenso de gradiente estocástico. El interés clave es que esta formulación da **gradientes no sesgados** para aproximar la distribución en Wasserstein \(W_1\), algo que C51 no consigue de forma directa por su paso heurístico de proyección seguido de una KL.

Como la pinball loss no es suave en cero, el paper propone la **quantile Huber loss**, que sustituye localmente el valor absoluto por una penalización cuadrática dentro de un margen \(\kappa\). En forma simplificada:

\[
L_\kappa(u)=
\begin{cases}
\frac{1}{2}u^2,& |u|\le \kappa \\
\kappa\left(|u|-\frac{1}{2}\kappa\right),& |u|>\kappa
\end{cases}
\qquad
\rho^\kappa_\tau(u)=|\tau-\mathbf 1_{\{u<0\}}|\,L_\kappa(u).
\]

La idea práctica es muy simple: cerca del error cero se obtiene una superficie más suave y estable; lejos de cero se conserva un comportamiento tipo \(L_1\), más robusto. SB3-Contrib usa precisamente esta familia de pérdidas para entrenar su implementación.

El **Bellman update** distribucional en control sustituye la actualización escalar de DQN por un target distribucional. En forma conceptual:

\[
\mathcal T Z(s,a)\overset{D}{=} r + \gamma Z(s',a^\*),
\qquad
a^\*=\arg\max_{a'}\mathbb E[Z(s',a')].
\]

La diferencia crucial es que el bootstrap se hace sobre una **distribución** de retornos de la acción siguiente, no sobre un escalar. En SB3, la acción siguiente \(a^\*\) se obtiene por la **media de cuantiles** de la red objetivo y luego se recogen los cuantiles correspondientes a esa acción para formar el target.

La intuición de convergencia que resulta defendible en un TFG es esta: si proyectas la distribución en una familia de cuantiles fijos y aplicas Bellman distribucional seguido de la proyección cuantil adecuada, el operador proyectado conserva una propiedad contractiva útil; por eso el aprendizaje por iteraciones/SGD tiende a estabilizarse hacia un punto fijo distribucional aproximado. No hace falta sobredemostrar en la memoria: basta con explicar que QR-DQN aproxima la distribución minimizando una distancia tipo Wasserstein de manera compatible con descenso estocástico, mientras que C51 depende de una proyección categórica separada.

La comparación más útil para el tribunal es la siguiente:

| Atributo                      | DQN                                | C51                               | QRDQN                                            |
| ----------------------------- | ---------------------------------- | --------------------------------- | ------------------------------------------------ |
| Qué modela                   | Esperanza\(Q(s,a)\)                | Distribución categórica         | Distribución por cuantiles                      |
| Representación               | 1 escalar por acción              | Soportes fijos + probs aprendidas | Probs uniformes + localizaciones aprendidas      |
| Salida por acción            | 1 valor                            | \(N\) probabilidades              | \(N\) cuantiles                                  |
| Necesita\(V_{\min},V_{\max}\) | No                                 | Sí                               | No                                               |
| Pérdida                      | TD/Huber escalar                   | KL a proyección categórica      | Quantile Huber                                   |
| Política greedy              | \(\arg\max Q\)                     | \(\arg\max \mathbb E[Z]\)         | \(\arg\max \mathbb E[Z]\)                        |
| Ventaja principal             | Simplicidad                        | Distribucional con coste moderado | Distribucional más flexible, sin soportes fijos |
| Inconveniente principal       | Pierde información de dispersión | Requiere proyección y cotas      | Cabeza de salida más grande y más coste        |

La tabla sintetiza la documentación oficial de DQN y QR-DQN en SB3/SB3-Contrib junto con los artículos originales de C51 y QR-DQN.

## Lógica algorítmica e implementación en SB3

A nivel algorítmico, la lógica de QRDQN en SB3 es la de un **off-policy method** clásico: recolecta transiciones en un replay buffer, espera hasta `learning_starts`, entrena cada `train_freq` pasos durante `gradient_steps` iteraciones y actualiza la red objetivo cada `target_update_interval`. La diferencia con DQN está en el contenido del target y en la pérdida, no en el esqueleto general del entrenamiento.

```
flowchart TD
    A[obs_t] --> B[selección de acción]
    B --> C[env.step]
    C --> D[guardar transición en replay buffer]
    D --> E{num_timesteps > learning_starts}
    E -- no --> A
    E -- sí --> F{se alcanzó train_freq}
    F -- no --> A
    F -- sí --> G[muestra minibatch]
    G --> H[Z_next = target_net(next_obs)]
    H --> I[a* = argmax_a media_tau Z_next]
    I --> J[Z_target = r + gamma * (1-done) * Z_next[a*]]
    J --> K[Z_current = online_net(obs, action)]
    K --> L[quantile_huber_loss]
    L --> M[Adam / backprop]
    M --> N{target_update_interval}
    N -- sí --> O[polyak_update]
    N -- no --> A
    O --> A
```

El diagrama resume fielmente el código fuente de `sb3_contrib.qrdqn.qrdqn` y la clase base `OffPolicyAlgorithm`.

El detalle más importante de la implementación oficial es **cómo se elige la acción bootstrap**. SB3 calcula los cuantiles del siguiente estado con la red objetivo, toma la **media** sobre el eje de cuantiles, aplica `argmax` sobre acciones y luego recoge los cuantiles de esa acción ganadora para formar el target. Matemáticamente, eso significa que la política es codiciosa con respecto a la **esperanza** de la distribución aprendida. Por eso, QRDQN en SB3 no es una política aversa al riesgo ni optimista por defecto; aprende distribución, pero decide como un maximizador del retorno esperado.

```python
# Pseudocódigo fiel al entrenamiento de SB3-Contrib QRDQN
Z_next = target_net(next_obs)                 # (B, N, A)
a_star = argmax_a(mean_tau(Z_next))           # acción greedy por esperanza
Z_boot = select_action_quantiles(Z_next, a_star)  # (B, N)
Z_target = r + (1 - done) * gamma * Z_boot

Z_current = select_action_quantiles(online_net(obs), action)
loss = quantile_huber_loss(Z_current, Z_target)
```

Este pseudocódigo condensa las líneas clave del método `train()` del QRDQN oficial.

En cuanto al **replay buffer**, QRDQN hereda la lógica genérica de `OffPolicyAlgorithm`. Si no se especifica `replay_buffer_class`, la base selecciona `ReplayBuffer` para observaciones vectoriales, `DictReplayBuffer` para observaciones tipo diccionario y `NStepReplayBuffer` si el algoritmo expusiera `n_steps>1`. Pero el constructor público de QRDQN no expone `n_steps`, así que en la práctica estándar de SB3-Contrib QRDQN se trabaja con **1-step targets** y replay uniforme. Además, el código de QRDQN muestrea con `self.replay_buffer.sample(...)`, sin pesos de prioridad, de modo que **no hay prioritized replay nativo** salvo que el usuario inyecte un buffer custom.

La **red objetivo** se actualiza con `polyak_update`. Si `tau=1.0`, esto equivale a una **copia dura** cada `target_update_interval`; si `tau<1`, pasa a ser una actualización blanda tipo Polyak. En tu repo usas `tau=1.0`, así que estás en modo hard target update. Ese punto conviene defenderlo explícitamente porque mucha gente asume “Polyak” y en realidad en tu código se comporta como una copia periódica.

La **exploración** es \(\varepsilon\)-greedy. Mientras `deterministic=False`, QRDQN compara un aleatorio con `exploration_rate`; si se explora, la acción es uniforme en el espacio discreto; si no, delega en la política QRDQN. La tasa \(\varepsilon\) sigue una programación lineal entre `exploration_initial_eps` y `exploration_final_eps` a lo largo de `exploration_fraction`. Esto es importante en tu caso porque no la sobreescribes, así que te quedas con el decaimiento rápido por defecto de QRDQN.

El **optimizador** por defecto es `Adam`. Si el usuario no pasa `optimizer_class` en `policy_kwargs`, el constructor de QRDQN inyecta `Adam` y fija además `optimizer_kwargs={"eps": 0.01 / batch_size}`. Ese detalle es poco conocido y muy defendible en una oral: en tu preset `full`, con `batch_size=2048`, el epsilon del Adam implícito es aproximadamente \(4.88\times 10^{-6}\); en `fast`, con `batch_size=512`, es aproximadamente \(1.95\times 10^{-5}\). No es solo “Adam por defecto”: el `eps` queda **acoplado al batch size**.

En relación con DQN estándar de SB3, conviene remarcar dos diferencias de implementación. Primero, **DQN** en SB3 documenta explícitamente que es una implementación “vanilla” sin Double-DQN, Dueling ni PER. Segundo, **QRDQN tampoco añade esas extensiones** por sí mismo: la arquitectura oficial es “casi idéntica a DQN” salvo la cabeza \(|A|\times N\), la pérdida quantile Huber y el cambio a Adam que ya señalaba el paper original. Además, la acción greedy del bootstrap se calcula sobre la **red objetivo**, no con una separación online/target al estilo Double DQN.

## Arquitectura de red y parámetros de la API

La política oficial `QRDQNPolicy` construye dos redes: `quantile_net` y `quantile_net_target`. Para observaciones vectoriales, el extractor por defecto es `FlattenExtractor` y la arquitectura MLP por defecto es `[64, 64]`; para imágenes usa `NatureCNN` y, en ese caso, `net_arch=[]`, porque el extractor convolucional ya produce la representación intermedia. La capa final genera `action_dim * n_quantiles` salidas y el `forward()` las reordena a forma `(-1, n_quantiles, n_actions)`.

```
flowchart LR
    O[Observación x] --> F[FlattenExtractor]
    F --> H1[MLP hidden]
    H1 --> H2[MLP hidden]
    H2 --> L[Linear a A*N]
    L --> R[reshape a (batch, N, A)]
    R --> M[media sobre N]
    M --> G[argmax acción]
```

Ese esquema resume exactamente cómo la política QRDQN transforma la observación en cuantiles por acción y, a partir de ellos, en valores esperados por acción para la política greedy.

En tu repo, la arquitectura concreta es más grande que la default: `MlpPolicy` con `net_arch=[512, 256]`. Como el entorno `RLDatasetDefenderEnv` tiene observaciones de 152 dimensiones y acciones discretas binarias, y como `n_quantiles` no se especifica, el cabezal de salida es de **400** neuronas \((2 \text{ acciones} \times 200 \text{ cuantiles})\). Por tanto, la forma de salida es \((B, 200, 2)\).

Para una MLP vectorial con extractor sin parámetros, el recuento de parámetros de la red online es:

\[
(dh_1+h_1) + (h_1h_2+h_2) + (h_2AN + AN),
\]

donde \(d\) es la dimensión de observación, \(A\) el número de acciones y \(N\) el número de cuantiles. La red objetivo duplica exactamente ese tamaño, aunque no se optimiza por gradiente. Aplicado a tu caso:

- \(d=152\)
- \(A=2\)
- \(N=200\)
- \(h_1=512\)
- \(h_2=256\)

da un total de **312 464 parámetros** en la red online y **624 928** si cuentas online + target. El desglose es: 78 336 en la primera capa, 131 328 en la segunda y 102 800 en la salida.

Para fijar intuición, en un control discreto pequeño con \(d=4\), \(A=2\), `net_arch=[64,64]` y \(N=200\), la red online tendría **30 480 parámetros**, en otro caso vectorial discreto algo mayor con \(d=8\), \(A=4\), mismos defaults y \(N=200\), tendría **56 736**. La conclusión relevante es que, en QRDQN, el coste crece **linealmente con \(A \times N\)**; por eso el número de cuantiles importa bastante más que en DQN estándar.

### Parámetros del constructor `QRDQN`

En la API pública de `QRDQN`, los únicos argumentos realmente **obligatorios** son `policy` y `env`. El resto son opcionales con defaults oficiales. La tabla siguiente resume su semántica y, sobre todo, si aparecen de forma explícita o implícita en tu repo.

| Parámetro                  | Default oficial | Papel                                                                  | Valores típicos / efecto                                             | En tu repo                                            |
| --------------------------- | --------------: | ---------------------------------------------------------------------- | --------------------------------------------------------------------- | ----------------------------------------------------- |
| `policy`                  |              — | tipo de política (`MlpPolicy`, `CnnPolicy`, `MultiInputPolicy`) | depende de observación                                               | **Sí**: `"MlpPolicy"`                        |
| `env`                     |              — | entorno Gym/Gymnasium o VecEnv                                         | obligatorio                                                           | **Sí**: `DummyVecEnv(Monitor(...))`          |
| `learning_rate`           |        `5e-5` | ritmo de aprendizaje; puede ser constante o schedule                   | más alto acelera, pero arriesga inestabilidad                        | **Sí**: `1e-4`                               |
| `buffer_size`             |   `1_000_000` | tamaño máximo del replay buffer                                      | buffers grandes estabilizan, pero consumen RAM                        | **Sí**: función de `timesteps`, cap en 200k |
| `learning_starts`         |         `100` | warm-up antes de entrenar                                              | demasiado bajo: targets ruidosos; demasiado alto: retrasa aprendizaje | **Implícito**: default 100                     |
| `batch_size`              |          `32` | tamaño de minibatch                                                   | mayor batch reduce varianza, aumenta coste/memoria                    | **Sí**: `512` / `2048`                     |
| `tau`                     |         `1.0` | soft update de target;`1.0` = hard copy                              | `<1` blando; `1` duro                                             | **Sí**: `1.0`                                |
| `gamma`                   |        `0.99` | descuento                                                              | más alto: horizonte largo; más bajo: más miope                     | **Sí**: `0.99`                               |
| `train_freq`              |           `4` | frecuencia de updates                                                  | controla cadencia entre datos y gradientes                            | **Sí**: `50` / `100`                       |
| `gradient_steps`          |           `1` | nº de pasos de gradiente por ciclo                                    | más alto = más compute por igual experiencia                        | **Sí**: `10` / `20`                        |
| `replay_buffer_class`     |        `None` | clase de buffer                                                        | custom buffer si se quiere extender                                   | **No**                                          |
| `replay_buffer_kwargs`    |        `None` | kwargs del buffer                                                      | p.ej.`handle_timeout_termination`                                   | **No**                                          |
| `optimize_memory_usage`   |       `False` | variante memory-efficient                                              | ahorra memoria, añade complejidad                                    | **No**                                          |
| `target_update_interval`  |       `10000` | cada cuántos pasos se actualiza target                                | pequeño = más reactivo; grande = más estable pero más lento       | **Sí**: `1000` / `10000`                   |
| `exploration_fraction`    |       `0.005` | fracción del entrenamiento donde cae\(\varepsilon\)                   | más grande = exploración prolongada                                 | **Implícito**                                  |
| `exploration_initial_eps` |         `1.0` | \(\varepsilon\) inicial                                                | normalmente alto                                                      | **Implícito**                                  |
| `exploration_final_eps`   |        `0.01` | \(\varepsilon\) final                                                  | más alto = más exploración residual                                | **Implícito**                                  |
| `max_grad_norm`           |        `None` | clipping de gradiente                                                  | puede mejorar estabilidad                                             | **Implícito**: sin clipping                    |
| `stats_window_size`       |         `100` | ventana para logging                                                   | afecta solo a métricas agregadas                                     | **Implícito**                                  |
| `tensorboard_log`         |        `None` | directorio TensorBoard                                                 | útil para monitorización                                            | **Sí**                                         |
| `policy_kwargs`           |        `None` | kwargs internos de la política                                        | donde se pasa `net_arch`, `n_quantiles`, etc.                     | **Sí**                                         |
| `verbose`                 |           `0` | nivel de log                                                           | 0,1,2                                                                 | **Sí**: `1`                                  |
| `seed`                    |        `None` | semilla global                                                         | clave para reproducibilidad local                                     | **Sí**: `42` por defecto CLI                 |
| `device`                  |      `"auto"` | CPU / CUDA                                                             | afecta velocidad y reproducibilidad                                   | **Sí**: CPU o CUDA detectado                   |
| `_init_setup_model`       |        `True` | construye red al instanciar                                            | rara vez se toca                                                      | **Implícito**                                  |

Los defaults del constructor y su semántica provienen de la documentación oficial y del código fuente de `QRDQN`; la columna del repo procede de `requirements.txt`, `train_rl_defender.py` y los `config.json` comprometidos.

### Parámetros internos de `QRDQNPolicy`

Aunque el usuario normalmente no instancia `QRDQNPolicy` a mano, para una memoria rigurosa conviene explicar sus kwargs porque `policy_kwargs` termina propagándose aquí.

| Parámetro                    | Default oficial                                                                | Papel                                                                   | En tu repo                              |
| :---------------------------- | :----------------------------------------------------------------------------- | ----------------------------------------------------------------------- | --------------------------------------- |
| `n_quantiles`               | `200`                                                                        | nº de cuantiles por acción; afecta resolución distribucional y coste | **Implícito**                    |
| `net_arch`                  | `[64,64]` en MLP, `[]` con `NatureCNN`                                   | profundidad/anchura MLP                                                 | **Sí**: `[512,256]`            |
| `activation_fn`             | `ReLU`                                                                       | no linealidad                                                           | **Implícito**                    |
| `features_extractor_class`  | `FlattenExtractor`                                                           | extractor para observación vectorial                                   | **Implícito**                    |
| `features_extractor_kwargs` | `None`                                                                       | configuración del extractor                                            | **No**                            |
| `normalize_images`          | `True`                                                                       | relevante en observaciones de imagen                                    | **Implícito**, irrelevante aquí |
| `optimizer_class`           | `Adam`                                                                       | optimizador                                                             | **Implícito**                    |
| `optimizer_kwargs`          | `None`, pero QRDQN inyecta `eps=0.01/batch_size` si no se pasa optimizador | detalle fino de estabilidad                                             | **Implícito**                    |

La política oficial crea dos redes independientes del mismo tamaño, copia pesos de online a target en la construcción y define la acción greedy por la **media** de cuantiles.

### Fragmento relevante de tu repositorio

```python
model = QRDQN(
    "MlpPolicy",
    vec_env,
    seed=seed,
    policy_kwargs=dict(net_arch=[512, 256]),
    learning_rate=lr,
    buffer_size=min(200_000, max(total_timesteps, 10_000)),
    batch_size=batch_size,
    gradient_steps=gradient_steps,
    gamma=0.99,
    tau=1.0,
    train_freq=train_freq,
    target_update_interval=target_update_interval,
    verbose=1,
    device=device,
    tensorboard_log=tb_log_dir,
)
```

Este bloque resume con fidelidad qué hiperparámetros fijas de forma explícita y cuáles dejas a default en tu entrenamiento principal.

## Consideraciones prácticas, evaluación y reproducibilidad

Desde un punto de vista práctico, QRDQN suele ser una buena elección frente a DQN cuando el problema tiene **acciones discretas** y sospechas que el retorno no está bien descrito por una media: entornos con colas de riesgo, recompensas muy asimétricas, multimodalidad o necesidad de inspeccionar incertidumbre aleatoria intrínseca. Frente a C51, QRDQN resulta especialmente cómodo cuando no quieres elegir manualmente cotas de soporte, porque la aproximación por cuantiles **no necesita** fijar \(V_{\min}\) y \(V_{\max}\).

En SB3, sin embargo, esa superioridad no es automática. La propia guía oficial recomienda no esperar “silver bullets”, hacer tuning cuantitativo, evaluar en un entorno separado y aumentar presupuesto si la señal es insuficiente. También insiste en depurar primero el entorno, diseñar recompensas informativas y validar muy bien terminaciones y wrappers. Todo eso te afecta directamente porque tu problema de ciberdefensa usa un entorno custom, reward shaping fuerte y validaciones fuera del split aleatorio.

En tu repo, el entorno tiene una recompensa muy asimétrica: `tp=1.5`, `fp=-2.0` por defecto actual y `fn=-5.0`, lo que induce una preferencia fuerte por evitar falsos negativos sin ignorar el coste de bloquear benignos. Eso tiene coherencia de dominio, pero también explica por qué una política puede tender a **bloquear demasiado** si el resto del pipeline no controla bien el shift de distribución. De hecho, el propio histórico de Phase 2 muestra que un artefacto temprano benign-only bloqueaba todo, mientras otro posterior benign-only lo permitía todo, señal clara de sensibilidad a configuración y dominio.

El gran resultado positivo de tu repo es el rendimiento en random split: el run `C03_qrdqn_cicids2017_canonical_full_random_20260223_232439` reporta accuracy 0.99859 y F1 de ataque 0.99876. Pero el hallazgo científicamente más valioso es negativo y, por eso mismo, más creíble: en la validación dura por día/CSV (Check C) la accuracy baja a 0.84135 y el recall de ataque a 0.52954. Esa diferencia te permite defender algo muy sólido: **el modelo aprende muy bien dentro del régimen i.i.d. aproximado del split aleatorio, pero generalizar entre días y capturas es mucho más difícil**.

Para el tuning, los rangos que ya exploras con Optuna en tu repo son razonables y, de hecho, merecen aparecer en la memoria: `learning_rate` entre \(10^{-5}\) y \(10^{-3}\) en log-space, `batch_size` en \(\{256,512,1024,2048\}\), `gradient_steps` en \(\{10,50,100\}\), `net_arch` entre `[256,128]`, `[512,256]` y `[256,256]`, `gamma` entre 0.95 y 0.999, y `train_freq` en \(\{50,100,200\}\). Eso ya constituye una exploración metodológicamente defendible, porque cubre capacidad de red, horizonte temporal, intensidad de actualización y rapidez de aprendizaje.

Dicho eso, si hubiera que señalar **los primeros dos hiperparámetros críticos** a revisar en tu canalización concreta, serían estos. Primero, `exploration_fraction`, porque al heredarse el default 0.005 puede que estés explorando demasiado poco durante casi todo el entrenamiento. Segundo, `n_quantiles`, porque 200 es un default razonable en Atari pero quizá no sea óptimo para un problema tabular/vectorial de clasificación secuencial con dos acciones y recompensas relativamente compactas. Ni tu script principal ni tus configs comprometidos lo exponen todavía de forma explícita.

En coste computacional, QRDQN es más caro que DQN sobre todo por la cabeza \(A\times N\) y por la pérdida entre conjuntos de cuantiles. En tu problema eso no es dramático porque \(A=2\), pero con muchos actions o imágenes el coste escala deprisa. A cambio, el paper original de QR-DQN reporta mejor rendimiento que C51 y mejoras sustanciales sobre el estado del arte de su momento en Atari, incluyendo un aumento mediano del 33 % sobre C51 al usar Huber quantile regression. Para el TFG, la formulación prudente es: **hay evidencia fuerte de mejora en benchmarks discretos clásicos, pero no debe extrapolarse sin más a ciberseguridad tabular**.

En evaluación, la guía oficial de SB3 recomienda separar claramente entrenamiento y test y recuerda que `Monitor` afecta a la contabilidad de reward/longitud de episodio. Tu pipeline además hace algo acertado para una tesis aplicada: complementar las rewards RL con **métricas de clasificación**. En `evaluate_model()` calculas matriz de confusión, accuracy, precisión, recall y F1 tanto para ataque como para benigno. Eso es especialmente apropiado en ciberdefensa, porque una reward media alta puede enmascarar trade-offs operativos que sí aparecen en confusion matrix y recall de ataque.

En reproducibilidad, SB3 es claro: incluso con la misma semilla no garantiza reproducibilidad completa entre plataformas, releases de PyTorch o CPU/GPU. La recomendación mínima es fijar `seed`, sembrar también el entorno si se reasigna con `set_env()`, y congelar versiones. Tu repo ya hace parte del trabajo: expone `--seed`, usa `vec_env.seed(seed)`, guarda `config.json`, `metrics.json`, `scaler.joblib` y percentiles. Lo que falta para un cierre experimental más fuerte es **pin version exacto** y, si es posible, exportar `pip freeze` o un lockfile por run.

Para visualizar la salida distribucional en una defensa, conviene mostrar no solo la media por acción, sino también algunos cuantiles de referencia. Un ejemplo conceptual sería este:

```text
Acción PERMIT:
q05 = -3.2    q50 = 0.1    q95 = 1.7    media ≈ -0.2

Acción BLOCK:
q05 = -0.9    q50 = 0.8    q95 = 1.1    media ≈  0.7

Decisión greedy de SB3: BLOCK
Interpretación: BLOCK tiene mejor retorno esperado y menor dispersión.
```

Ese tipo de gráfico ayuda mucho a explicar la diferencia entre “aprender una distribución” y “decidir por la media de esa distribución”.

Si quisieras extraer esas curvas de forma programática para una figura del TFG, una utilidad mínima podría ser:

```python
import torch
import numpy as np

def get_action_quantiles(model, obs_np):
    obs_t = torch.as_tensor(obs_np[None], device=model.device, dtype=torch.float32)
    with torch.no_grad():
        z = model.quantile_net(obs_t)      # (1, N, A)
    z = z.squeeze(0).cpu().numpy()         # (N, A)
    q_mean = z.mean(axis=0)                # (A,)
    return z, q_mean
```

La posibilidad de acceder a `quantile_net`, la forma `(batch, N, A)` y la selección greedy por `mean(dim=1)` se desprenden del propio código fuente de SB3-Contrib.

## Preguntas de tribunal y respuestas breves

- **¿Qué aprende QRDQN que DQN no aprende?**DQN aprende solo el retorno esperado por acción. QRDQN aprende una aproximación de la **distribución completa de retornos** mediante \(N\) cuantiles por acción. Eso permite inspeccionar dispersión, colas y multimodalidad, aunque la política greedy por defecto siga usando la media.
- **¿En qué se diferencia QRDQN de C51 en una frase?**C51 fija los soportes y aprende probabilidades; QRDQN fija probabilidades uniformes y aprende las localizaciones de los cuantiles. Por eso QRDQN no necesita \(V_{\min}\) ni \(V_{\max}\).
- **Si QRDQN aprende distribución, por qué dices que no es “risk-sensitive”?**Porque en SB3 la acción se elige con `argmax` de la **media** de cuantiles. La distribución se aprende y se puede analizar, pero la política por defecto sigue maximizando esperanza.
- **¿Tu implementación usa prioritized replay, Double DQN o dueling?**No de forma nativa. El QRDQN de SB3-Contrib usa el esqueleto off-policy estándar con `ReplayBuffer` y muestreo normal; además, la selección bootstrap se hace sobre la target net, no en esquema Double DQN. Tu repo tampoco inyecta un buffer custom ni una arquitectura dueling.
- **¿Cuál es el hiperparámetro más “silencioso” de tu implementación?**Probablemente `exploration_fraction`, porque no se fija en el repo y el default de QRDQN es 0.005. Eso implica que \(\varepsilon\) cae muy deprisa, lo que puede reducir exploración real más de lo intuitivo.
- **¿Qué gain te aporta `n_quantiles`?**Controla la resolución con la que aproximas la distribución. Más cuantiles significan mayor detalle en colas y forma, pero también más coste y una cabeza de salida más grande. En SB3 el default es 200 y en tu repo se está usando inferidamente ese valor.
- **¿Por qué no basta con reportar accuracy de random split?**Porque tu propio repositorio muestra que el rendimiento en split aleatorio es casi perfecto, pero cae de forma notable en el split duro por día/CSV. Eso indica que el verdadero reto es la **generalización fuera de distribución**, no el ajuste en condiciones i.i.d. aproximadas.
- **¿Qué aspecto de tu repo hace más creíble el trabajo?**Que no te quedas en un único score: guardas artefactos por `RUN_ID`, diferencias defaults actuales de settings históricos, usas validaciones anti-leakage y por día/CSV, y documentas explícitamente las lagunas donde todavía no hay artefacto comprometido. Eso mejora la trazabilidad experimental.
- **¿Cuál sería una mejora inmediata para la memoria del TFG?**
  Fijar en texto y, si es posible, en código los parámetros implícitos heredados de SB3-Contrib: versión exacta, `n_quantiles`, exploración y `optimizer_kwargs`. Eso evita reproducibilidades ambiguas y muestra dominio real de la implementación.
