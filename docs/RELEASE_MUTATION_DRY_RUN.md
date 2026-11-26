# Release 1.0: ensayo del contrato de ejecución y agregación

Este ensayo de OpenSpec 6.2 consume un plan `release-1.0` ya preparado y su
digest externo. No vuelve a listar candidatos. Sólo describe tickets, valida
resultados de fixtures y agrega esos fixtures. No ejecuta procesos, genera
shell, despacha Actions ni habilita el runner histórico para Release.

`tools/release_mutation_dry_run.py` ofrece tres subcomandos:

| Etapa | Entrada y comprobación | Salida |
| --- | --- | --- |
| `describe` | Perfil, base, HEAD y digest externo del plan explícitos; productor del preflight identificado | 64 tickets declarativos, con ejecución deshabilitada |
| `seal-fixture` | Shard esperado, ticket reconstruido del plan y envelope explícito de fixture | Resultado seco ligado al ticket y al digest del fixture |
| `merge-fixtures` | Exactamente 64 resultados secos únicos; misma base, HEAD, plan, presupuesto, partición e identidad | Conteos anidados de fixtures; nunca un reporte de campaña |

Cada ticket fija el digest del archivo y payload del plan, fuentes, diff,
candidatos de su shard y adaptador. Declara baseline de los paquetes mutados,
pruebas de mutantes sobre el workspace y Cargo `--locked`. La validación de
fixtures reutiliza `validate_outcomes`: exige baseline exitoso, identidad exacta,
fases y argv compatibles, cuentas coherentes y resultados completos. Los
`MissedMutant`, `Timeout` y `Unviable` se conservan en los conteos de fixtures.

El agregado usa un esquema distinto, sin `outcomes`, `total_mutants` ni
`end_time` en su nivel superior. Así no puede confundirse con un reporte válido
para `quality.py mutations`, aunque los fixtures declaren todos los candidatos
detectados. Los consumidores históricos de manifiestos también lo rechazan.
Nunca convertir estos fixtures al formato de evidencia de campaña.

## Uso del plan preservado

Ejemplo de descripción, con variables que identifican el plan y recibo externos
ya conservados y un destino nuevo:

```sh
python3 tools/release_mutation_dry_run.py describe --profile release-1.0 \
  --revision "$BML_RELEASE_HEAD" \
  --base deec3aa8d2516ae19e335a5af77e6c38f8a854d8 \
  --plan "$BML_PREFLIGHT_PLAN" --plan-file-sha256 "$BML_PREFLIGHT_SHA" \
  --budget-profile documented-30-180 --output dry-description.json
```

Para `seal-fixture` se añaden `--shard`, `--ticket` y `--fixture`. El fixture
debe tener exactamente `schema=bml-release-outcomes-fixture-v1`, el
`ticket_sha256` y los datos sintéticos de `outcomes`. `merge-fixtures` recibe
64 argumentos `--result`. Todos los modos exigen los mismos argumentos de
intención y presupuesto, y escriben exclusivamente archivos nuevos. No existe
un modo `run`, `execute` o `dispatch`.

Este ensayo autentica un plan preservado, no vuelve a acreditar un checkout
vivo ni demuestra que se respetaron recursos o duraciones reales. No reemplaza
la verificación de fuentes previa a una ejecución futura.

## Presupuestos: documentación, operación y medición pendiente

| Perfil de descriptor obligatorio | Test / build, segundos | Procedencia |
| --- | --- | --- |
| `documented-30-180` | 30 / 180 | Comando de entrega de `BML_1_0.md` |
| `operational-360-600` | 360 / 600 | Runner y workflow históricos |

No hay perfil predeterminado en el ensayo. Elegir un descriptor de fixture no
cambia defaults históricos ni decide el presupuesto de una campaña Release.
Faltan tiempos de baseline, compilación y tests, memoria máxima y evidencia de
suficiencia sobre el alcance y plataforma que se vayan a ejecutar.

La configuración histórica de 64 shards fija 300 minutos por runner y
330 minutos por job, hasta ocho jobs simultáneos, prepare de 30 minutos y
aggregate de 15. De esos límites se derivan, por intento:

- 320 horas-runner nominales para los 64 procesos de 300 minutos.
- Hasta 321 h 4 min incluyendo sus 60 segundos de terminación, dentro de los
  límites de job; no sumar esa terminación nuevamente a las cifras siguientes.
- 352 h 45 min de presupuesto total de jobs: `64 × 330 + 30 + 15` minutos.
- 44 h 45 min de envolvente condicional con ocho slots continuamente
  disponibles y sin colas. `max-parallel: 8` no garantiza esos slots; no existe
  una cota de tiempo total que incluya colas y demoras externas. La retención
  de artefactos es de 72 horas y puede vencerse antes de poder agregar.

Son límites de configuración del workflow histórico, no mediciones ni una
predicción de finalización de Release. No se conoce un costo monetario máximo:
faltan tarifa efectiva, minutos incluidos, condiciones de facturación y cargos
de almacenamiento. `21.165 × tarifa efectiva por minuto` sólo describe un
subtotal nominal condicionado; no es una factura máxima verificada.

## Decisiones antes de cualquier ejecución

| Frontera | Validado en seco | Pendiente para ejecución |
| --- | --- | --- |
| Plan → executor | Tickets derivados del plan, intención y partición exactas | Verificar checkout vivo y admitir explícitamente el perfil Release en una ruta revisada; no relabelar su base como histórica |
| Executor → agregado | Identidad, fases y resultados de fixtures con el validador heredado | Resultados reales, baselines, logs, fuentes verificadas y sellos del mismo intento |
| Actions | Formatos históricos intactos; no dispatch ni archivos de workflow modificados | Decidir y revisar el transporte del digest confiable y el perfil de ejecución; el workflow actual sigue usando `40b3f…` |
| Presupuesto | Dos descriptores explícitos, sin mezcla | Seleccionar una política respaldada por mediciones; comprobar slots, retención y facturación |
| Cierre de 6.2 | Coherencia del contrato seco | Campaña aplicable completa, cero missed y timeouts, features, cobertura y demás controles sobre el SHA correspondiente |

La unión histórica incompleta 58/64 no sustituye esa evidencia. Los resultados
de este ensayo no permiten marcar 6.2 como aprobada.
