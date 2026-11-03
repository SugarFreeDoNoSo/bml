# Laboratorio BML

Cuaderno local con una interfaz HTML, doce ejemplos Rust editables y salida por
celda. El contenido del cuaderno está en `notebook.json` y los programas originales
en `examples/`. No requiere Jupyter, Node.js ni dependencias de Python.

## Abrir el cuaderno

Desde la raíz del repositorio, con Python 3.10+ y Cargo en `PATH`:

```sh
python3 sandbox/server.py
```

Abre **http://127.0.0.1:8765**. El servidor usa el toolchain fijado por
`rust-toolchain.toml`; debe estar instalado con sus componentes. La primera
ejecución compila las dependencias y puede descargarlas si no están en la caché de
Cargo. Las siguientes reutilizan la compilación.

Si necesitas otro puerto o más tiempo para compilar:

```sh
python3 sandbox/server.py --port 8766 --timeout 300
```

1. Elige un ejemplo en el índice y modifica su código.
2. Pulsa **Ejecutar** o `Ctrl/Cmd+Enter` dentro del editor. La ejecución guarda
   primero la celda y muestra stdout, stderr, duración y código de salida.
3. **Guardar cambios** permite conservar una edición sin ejecutarla.
4. **Restaurar original** descarta la copia local de esa celda, previa confirmación.

Cada celda es un programa completo con su propio `fn main()`. Las variables no se
comparten entre celdas y se ejecuta una sola a la vez. Los errores de compilación
se muestran junto al ejemplo y puedes corregir el código y volver a ejecutarlo.
Los cambios sin guardar se advierten al salir de la página. La salida no persiste
al recargar; el código guardado sí.

## Editor Rust

El código tiene resaltado local de palabras clave, tipos, funciones, macros,
cadenas, números y comentarios, sin cargar bibliotecas desde un CDN.
**Analizar Rust** consulta diagnósticos de rust-analyzer; **Completar** o
`Ctrl+Espacio` muestra sugerencias en el cursor; **Información del símbolo**
consulta sus tipos y documentación. Analizar no guarda ni ejecuta la celda.

Instala los componentes en el toolchain fijado antes de iniciar el servidor:

```sh
rustup component add --toolchain 1.98.1 rust-analyzer rust-src
```

El servidor inicia rust-analyzer al recibir la primera consulta y crea un
workspace de análisis en `sandbox/.local/ide/`, con un binario por celda y las
dependencias BML locales. La primera indexación puede tardar; si indica que aún
está indexando, vuelve a pulsar Analizar. No se ejecutan build scripts, macros
procedurales ni `cargo check` desde el analizador; los diagnósticos no sustituyen
la compilación de **Ejecutar**. Cargo metadata usa modo offline, por lo que las
dependencias deben estar previamente disponibles en la caché.

El resaltado funciona en el navegador. El análisis utiliza el servidor nativo
del toolchain, no WASM: GitHub Pages por sí solo no puede ejecutar este backend.
El tablero de Pages es independiente del cuaderno local.

## Ejemplos incluidos

| Celdas | Funcionalidades |
| --- | --- |
| 01–03 | Codificación compacta BML, síntesis desde `f64` y aritmética simbólica |
| 04–05 | Variables, bases BML/EML y errores de dominio/rango |
| 06–09 | Formas, vistas, kernels IEEE, matrices y aplicación de bases a tensores |
| 10–11 | Codecs F32/F16/Q4_0/Q8_0 y lectura de un GGUF sintético como tensor |
| 12 | DAG, RPN y serialización del compilador histórico |

Los ejemplos no necesitan descargar un modelo ni ejecutar inferencia. Los kernels
nativos usan aritmética IEEE; la evaluación simbólica y `apply_basis` comprueban
el dominio real. El ejemplo GGUF muestra la inversión de dimensiones entre GGML
y row-major sin cambiar el orden de los valores.

## Archivos de trabajo

```text
sandbox/
  notebook.json          # Orden y descripción de las celdas
  examples/*.rs          # Ejemplos originales versionados
  index.html             # Interfaz del cuaderno
  server.py              # Editor, persistencia y ejecución local
  Cargo.toml, Cargo.lock # Dependencias del ejecutor, workspace independiente
  .local/                # Ignorado por Git
    cells/<id>.rs        # Ediciones guardadas por celda
    runner.rs            # Copia de la última celda ejecutada
    target/              # Caché y binario compilado
```

Los programas se ejecutan con `sandbox/.local/` como directorio de trabajo; sus
archivos relativos también quedan ignorados. Los originales nunca se sobrescriben
desde la interfaz. El ejecutor es un workspace independiente para que una celda
experimental rota no afecte a los tests de la biblioteca. Cargo usa `--locked` y
las dependencias locales del repositorio.

El servidor solo escucha en loopback. **Ejecuta código Rust con tus permisos de
usuario; no es un contenedor de aislamiento.** Está pensado para tus experimentos
locales. Para acceder de forma remota usa la [entrada protegida con Nginx y
Cloudflare Access](../../project-gateway/README.md), restringida al correo del
propietario. Ejecuta únicamente código que hayas revisado. La entrada
interactiva por stdin está desactivada. El límite de tiempo por defecto es de
120 segundos, incluida la compilación, y se retienen hasta 128 KiB por canal de
salida. En Linux/macOS se termina el grupo de procesos al finalizar o agotar el
tiempo; en Windows se termina únicamente el proceso directo.

## Verificación del servidor

```sh
python3 -m unittest discover -s tools -p 'test_sandbox.py'
```

Estas pruebas comprueban persistencia, acceso HTTP y límites del ejecutor sin
necesitar Cargo. Para comprobar las funcionalidades Rust, ejecuta las doce celdas
desde el cuaderno. El cierre con `Ctrl+C` detiene el servidor.
