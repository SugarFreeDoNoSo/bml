# Atención, MLP y composición causal de capa — 2026-11-03

Esta entrega repara las tres APIs públicas que seguían usando operaciones
provisionales. No completa el pipeline distribuido histórico ni el objetivo
global de las specs. Resultados finales, comandos y hashes se registran en
`operation-layer-2026-11-03.json`.

## Atención y MLP

`compile_attention_fragment` conserva su firma y declara una cabeza con un
token KV. Los constructores configurables añaden ancho, contexto y bounds
explícitos; K/V son matrices row-major `[context, head_dim]`. Ejecutan cada dot,
la escala, selección del máximo, softmax y suma ponderada. La variante con
scratch conserva exponentiales y evita recalcular scores por componente:
trabajo O(context*head_dim), 2324 instrucciones, 27888 bytes, 89 constantes y
pico de pila 37. C97 contrasta atención firmada no uniforme, ceros, direcciones,
pila, almacenamiento y preflight de todos los BML.

La variante general sin scratch intercambia memoria temporal por recomputación
O(context*head_dim²); puede exceder 32 KiB y usa tres bucles anidados. El grafo
validado histórico conserva su límite de dos niveles y la rechaza. La capa usa
la variante con scratch, y el wrapper de un solo token tiene una especialización
completa que cumple ambos límites. No se atribuye GQA a estas APIs.

`compile_mlp_fragment` ahora usa todas las matrices gate/up/down y declara FFN
cuadrado. Su constructor configurable acepta embedding y FFN independientes.
El plan con scratch realiza gate/up una vez, calcula `gate*sigmoid(gate)*up` y
proyecta down: O(embedding*FFN). `operations()`/`into_operations()` entrega cuatro
bucles completos con constantes y metadatos propios; la operación concatenada
no es una garantía de fragmentación. El wrapper sin scratch recomputa las
proyecciones por salida y conserva tres niveles de bucle: sirve al ejecutor de
operaciones, pero no pasa el límite histórico de dos niveles de `ValidatedGraph`.

C98 contrasta matrices rectangulares, modifica sus elementos, verifica dominios,
shapes, overflow, layout, memoria/pila y aritmética nativa usada solo para
direcciones. El compilador evalúa únicamente sub-DAGs cerrados de configuración
con `Basis::Bml.apply_real`, conservando los bits del DAG original. Nunca pliega
activaciones ni pesos aprendidos. Los perfiles AOT anteriores y sus listas de
literales permanecen separados de esta optimización.

## Capa completa y memoria

`LayerConfig`, `LayerLayout` y `LayerWeights` declaran dimensiones, posición,
epsilon, frecuencias, límites, stride de slot y direcciones de los nueve tensores.
La construcción comprueba overflow, capacidad y ausencia de solapamiento entre
entradas/pesos y salidas/scratch/KV. Quince slots temporales y dos slots KV son
explícitos; cada slot admite el prefijo completo y el vector FFN. Las matrices
aprendidas pueden cruzar límites de slot porque sus bases son direcciones
absolutas, no números de slot.

La secuencia ejecuta RMS, Q/K/V, RoPE para Q/K, escritura de la fila KV actual,
atención al prefijo, proyección de salida, residual original, segundo RMS,
gate/up/SwiGLU/down y segundo residual. Conserva filas KV anteriores y no
sobrescribe la entrada. Esta caché de operaciones usa f64, distinta del contrato
F32 del motor de modelo. El llamador conserva la sesión y solo debe avanzar su
posición después de validar el resultado; el ejecutor IEEE no es transaccional.

El wrapper histórico usa una cabeza, posición cero, FFN cuadrado, epsilon 1e-5,
RoPE base 10000 y stride igual a embedding. Convierte `input_slot` a base
absoluta; reserva los siguientes 17 slots y escribe en `input_slot+18`. Su antiguo
argumento de constante de columnas ya no es necesario: cada operación posee
stride y configuración. Los llamadores que dependieran de los antiguos ocho slots deben
usar el nuevo layout explícito o reservar el espacio documentado.

C99 ejecuta dos posiciones con pesos firmados, epsilon no estándar y FFN
rectangular contra ecuaciones densas independientes; verifica cada BML mediante
el ejecutor comprobado. También comprueba configuraciones de 0/1/2/4/8 workers,
los nueve tensores, memoria ajena, prefijo KV y layouts inválidos. La construcción
de anchos 4/6 y 2048/5632 produce 17 fragmentos de igual número de instrucciones,
todos dentro de 32768 bytes. Esta prueba no ejecuta un modelo de ese tamaño.

RoPE vectorial bajó de 89160 a 25020 bytes mediante reutilización de nodos
idénticos y liberación de temporales. C93 verifica igualdad bit a bit con el DAG
original, varias posiciones, signos, ceros y frecuencias reiniciadas por cabeza.
No se reasocian las operaciones aritméticas.

## Selección auxiliar y persistencia

`RpnOp::FMax` selecciona entre dos valores finitos. Los evaluadores IEEE producen
NaN ante operandos no finitos; la ruta comprobada devuelve error. C95/C96 cubren
referencia, fragmentos, loops, runtime, planificación, SHM y codecs. El tag nuevo
es 16; los contenedores con selección usan versión binaria 4. Los bytes históricos
v2/v3 se conservan y las versiones anteriores rechazan el tag nuevo.

La versión binaria 4 no debe confundirse con el perfil numérico causal v4.
Los perfiles simbólicos anteriores continúan rechazando FMax. Los entrypoints
comprobados de selección y operaciones generales son explícitos; el segundo
comprueba direcciones enteras exactas, rangos, datos finitos y cada BML. Permite
FAdd/FMul finitos sin afirmar que cualquier bytecode arbitrario los use solo para
índices; esa propiedad se verifica en los compiladores de atención y MLP.

## Verificación final

El gate `./scripts/check.sh origin/master`, con base
`a1e1ae9919ea8999b52f475d2423f239dbf1eb0e`, pasó: 875 pruebas Rust
(17 ignoradas), 86 Python, auditoría histórica sin discrepancias, 99/99
contratos registrados y cero diagnósticos Clippy nuevos (60 actuales frente a
baseline 192). También pasó la compilación de todo el workspace con todas las
features; no se presenta como ejecución de tests con todas ellas. OpenSpec
estricto valida 11/11 elementos.

La auditoría final del worker release registra 2533 bytes del dispatcher,
19 llamadas resueltas y dos tablas de salto, sin llamadas RPC/socket/syscall en
ese cuerpo. El informe `operation-layer-hot-loop-2026-11-03.json` conserva
binario, toolchain, fuentes y disassembly; no acredita PMU ni residencia L1.
Se conservan los intentos fallidos: dos matches de helpers de pruebas faltaban
al incorporar FMax y Clippy solicitó `size_of_val` para un slice. La versión
final pasó después de corregirlos, sin regenerar baselines.

## Alcance pendiente

Siguen abiertos el compilador/worker/CLI distribuido histórico y la auditoría
final de placeholders, modelos v4 mayores y prompts adicionales, ARM/NEON real,
PMU/L1, estrés de hardware y los gates globales de cobertura y mutación. Los
tamaños de bytecode y del dispatcher no demuestran residencia L1 ni velocidad
de inferencia. No se archiva `fix-placeholders`.
