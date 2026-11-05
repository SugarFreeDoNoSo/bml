# bml-simd-backends Specification

## Purpose
Definir capacidades, selección segura y equivalencia de los kernels SIMD para el trabajo declarado, con evidencia por arquitectura.

## Requirements

### Requirement: Referencia escalar y selección segura de backend
Cada kernel SHALL disponer de una referencia escalar y seleccionar su backend
por arquitectura y capacidades habilitadas de la CPU cuando corresponda. AVX2
SHALL dirigirse a x86/x86-64 y NEON a ARM compatible; una instrucción específica
no SHALL ejecutarse sin garantizar su disponibilidad.

#### Scenario: Disponibilidad y contrato del backend
- **WHEN** se configura o solicita un backend
- **THEN** la compilación condicional,
los requisitos de CPU, las funciones `unsafe` y sus invariantes SHALL documentarse.
Un backend no disponible SHALL usar fallback escalar o devolver un error explícito
al solicitarlo de forma forzada. SVE/SVE2 SHALL quedar fuera del alcance inicial.

#### Scenario: CPU sin AVX2
- **WHEN** la selección automática se ejecuta en una CPU sin AVX2
- **THEN** usa un backend compatible sin ejecutar instrucciones AVX2
- **AND** solicitar AVX2 forzado devuelve un error explícito.

### Requirement: Trabajo vectorizado declarado
Cada kernel SIMD SHALL identificar si vectoriza manipulación de códigos,
construcción de expresiones o evaluación numérica; SHALL describir instrucciones,
disposición, tamaños de códigos, gestión de colas y errores. Un kernel numérico
SHALL declarar cómo calcula `exp2`/`log2`, su precisión y rango. Sumar o
multiplicar los bits del código no SHALL presentarse como aritmética BML.
El ancho vectorial por sí solo no SHALL atribuir una aceleración.

#### Scenario: Manipular códigos no acredita evaluación
- **WHEN** un backend vectoriza validación o comparación de códigos
- **THEN** solo se atribuye esa capacidad y no se afirma que evalúa BML en SIMD.

### Requirement: Equivalencia con la referencia
Cada backend SHALL preservar contrato estructural, orden, presupuestos, salida
y primer error determinista de la referencia escalar para el mismo trabajo.
Las rutas numéricas SHALL declarar previamente su precisión y política de errores;
SHALL comparar también dominio, rango y colas no múltiplos del ancho. No SHALL
reasociarse una reducción sin prueba y política explícita. Puede procesarse en
paralelo salidas independientes conservando el orden interno de cada reducción.

#### Scenario: Cola y primera falla
- **WHEN** hay elementos válidos e inválidos y una cantidad no múltiplo del ancho SIMD
- **THEN** el backend mantiene los mismos resultados y primer error que el escalar
- **AND** no entrega resultados parciales ni omite presupuestos.

#### Scenario: Equivalencia estructural
- **WHEN** se construye la misma matriz BML en escalar, AVX2 y NEON disponibles
- **THEN** los códigos coinciden exactamente salvo que una transformación distinta tenga un contrato y prueba explícitos.

### Requirement: Evidencia por arquitectura
AVX2 y NEON SHALL validarse y medirse en hardware objetivo real antes de afirmar
ventajas. La compilación cruzada no SHALL contar como ejecución ni evidencia de
rendimiento. Si no hay hardware, se SHALL registrar como pendiente/no ejecutado.

#### Scenario: Compilación ARM sin ejecución
- **WHEN** el backend NEON compila para ARM pero no se ejecutó en ARM
- **THEN** la ejecución y el rendimiento quedan pendientes sin atribuirles resultados.
