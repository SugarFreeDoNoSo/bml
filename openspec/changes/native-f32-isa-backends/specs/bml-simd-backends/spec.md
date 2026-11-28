## ADDED Requirements

### Requirement: Proyecciones F32 nativas con especialización optativa
Las proyecciones F32 de cabeza y FFN en CLI y ELF nativo SHALL ofrecer un control
escalar y variantes SIMD optativas sin cambiar el default. SHALL conservar una
ruta x86-64 convencional y verificar capacidad de CPU y soporte del sistema
operativo antes de ejecutar instrucciones especializadas.

#### Scenario: Fallback y selección efectiva
- **WHEN** la variante solicitada no está disponible en CPU o sistema operativo
- **THEN** la selección automática usa el control compatible y una solicitud forzada devuelve un error explícito
- **AND** se registran variante solicitada y variante efectivamente ejecutada en CLI y ELF.

### Requirement: Identidad comprobable de AVX y AVX2 en inferencia
La evaluación SHALL separar escalar, AVX y una variante AVX2 sólo si una operación
efectivamente requiere AVX2. SHALL documentar instrucciones, selección y trabajo
numérico comprobados. Comparar limbs enteros o ejecutar operaciones elementales
F64 no SHALL contarse como estas proyecciones F32.

#### Scenario: CPU con AVX2 que ejecuta sólo AVX
- **WHEN** la CPU admite AVX2 pero las instrucciones verificadas de la proyección sólo requieren AVX
- **THEN** el resultado se identifica como AVX y no como aceleración AVX2
- **AND** la decisión sobre packing/gather u otra necesidad real de AVX2 se conserva con evidencia.

### Requirement: Reducción exacta por filas independientes
La ruta exacta SHALL preservar el orden de columnas de la referencia escalar
Rust 1.98.1 en cada una de cuatro filas independientes, acumulación F64 desde
`-0.0`, ambos operandos F32 convertidos a F64 antes del producto, multiplicación
y suma separadas y cast final F32. SHALL acreditar paridad
bit a bit, sin FMA implícito ni reasociación horizontal de la reducción.

#### Scenario: Ceros con signo, cancelación y colas
- **WHEN** los inputs contienen signed zero, cancelación sensible, subnormales o cantidades de filas/columnas no múltiplos de cuatro
- **THEN** los bits F32, orden y errores aplicables coinciden con el control escalar
- **AND** pruebas negativas distinguen semilla incorrecta, reasociación y FMA cuando cambian el resultado.

### Requirement: Coste y límites de pesos preparados
Cada disposición SIMD SHALL declarar sus cotas de memoria y trabajo y medir
packing, copia, asignación, reutilización e invalidación. SHALL separar coste
total y ejecución con pesos preparados; el ahorro del kernel no SHALL ocultar
el coste de preparación ni justificar presupuestos omitidos.

#### Scenario: Packing sin beneficio neto
- **WHEN** una variante requiere reorganizar pesos y no mejora el coste total de la carga medida
- **THEN** se conserva ese resultado con tiempo y memoria de preparación y ejecución separados, sin prometer una mejora universal.

### Requirement: Benchmark de inferencia aislado por ISA
La comparación escalar/AVX/AVX2 SHALL fijar GGUF/hash, prompts/IDs, BOS/plantilla,
contexto, núcleo/afinidad y presupuesto CPU. SHALL verificar logits, KV e IDs y
reportar selección e instrucciones efectivas, TTFT/prefill, decode sostenido,
dispersión y memoria con orden alternado y repeticiones.

#### Scenario: Contraste con stock llama.cpp
- **WHEN** se incluye stock llama.cpp con su revisión, flags e ISA propios
- **THEN** se presenta separado de la matriz que aísla ISA dentro de BML
- **AND** no se atribuye toda la diferencia a AVX2 ni se cuentan BOS, prompt o EOS como tokens nuevos de decode.

### Requirement: FMA y BMLX mantienen contratos separados
FMA SHALL ser un experimento optativo separado con política numérica previa y
equivalencia justificada o divergencias registradas. La extensión a BMLX simbólico SHALL decidirse por
medición cuando 8.6 esté operativa, preservando operaciones BML, orden, dominio
y cotas locales/globales. La evidencia nativa no SHALL acreditar BMLX.

#### Scenario: Ruta simbólica aún bloqueada
- **WHEN** CLI o ELF nativo tienen resultados pero la ejecución simbólica continúa bloqueada
- **THEN** la extensión SIMD de BMLX queda pendiente con su bloqueo concreto, sin extrapolación de rendimiento ni cierre de 8.6.

#### Scenario: Experimento FMA con bits distintos
- **WHEN** FMA cambia los bits respecto de la referencia exacta
- **THEN** se reporta la divergencia según su contrato separado y no se califica como paridad bit a bit de la ruta sin FMA.
