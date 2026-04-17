# 02 — Datos, esquema canónico y preprocesado

## 1) Datasets y su papel

## CICIDS2017 (principal)

- es la base moderna del proyecto
- se usan sus 8 CSV oficiales (`src/load_cicids2017.py`)
- define el espacio de observación actual

## NSL-KDD (histórico)

- sirve como benchmark histórico
- mantiene valor didáctico para evolución del proyecto
- su mapeo al esquema canónico es muy parcial (`NSL_KDD_TO_CANON`)

## 2) Decisión arquitectónica clave: esquema canónico

Archivo: `src/canonical_schema.py`

Se define una lista fija `FEATURES_CANON` de **76 features flow-based**. Esto garantiza que cada posición del vector tenga siempre el mismo significado.

### Observación final

- 76 valores de features
- 76 valores de máscara de missingness
- total: **152 dimensiones**

## 3) Máscara de missingness (semántica)

- `1 = presente/válida`
- `0 = ausente o imputada`

Esto ayuda a no confundir “dato real” con “dato rellenado”.

## 4) Política anti-leakage

`load_cicids2017.py` elimina columnas propensas a leakage:

- IPs
- timestamps
- Flow ID
- puertos directos como proxy de etiqueta

Razonamiento: si el modelo aprende atajos espurios, las métricas offline pueden inflarse artificialmente y fallar en generalización real.

## 5) Preprocesado en CICIDS2017

Pipeline principal en `load_cicids2017.py`:

1. carga por chunks
2. localización robusta de columna de etiqueta
3. coerción numérica
4. `inf -> NaN -> fillna(0)` en features
5. mapeo a canónico + máscara
6. split (`random` o `day`)
7. `StandardScaler` (fit en train, transform en test)

## 6) Modos de split y por qué importan

## `random`

- estratificado 80/20
- útil para iterar rápido
- puede sobreestimar generalización

## `day`

- train y test en grupos de días/CSV distintos
- más exigente y más cercano a escenario real

## `exact CSV`

- usado en leave-one-exact-CSV-out
- deja fuera un CSV real completo por fold

## 7) Puntos finos que conviene dominar

- En CICIDS2017, los NaN se rellenan antes de mapear; la máscara captura sobre todo disponibilidad/validez tras mapping, no “histórico crudo” completo de missingness por celda.
- El `StandardScaler` se aplica a toda la observación (incluida máscara), así que la red recibe máscara escalada, no bits puros 0/1.

