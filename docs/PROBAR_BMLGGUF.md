# Probar BMLGGUF en otra máquina

La CLI puede convertir, inspeccionar y ejecutar contenedores `.bmlgguf`
autónomos. La generación está validada con un fixture pequeño de Llama;
la conversión y la inferencia simbólica de modelos reales grandes siguen
teniendo límites numéricos y de recursos.

## Descargar y comprobar la ejecución

Desde el clon del repositorio:

```sh
git switch master
git pull --ff-only origin master
cargo test -p bml-cli --test prepared_token_output --locked
```

Se necesita Rust mediante rustup. `rust-toolchain.toml` fija la versión y los
componentes del proyecto. El test crea un modelo pequeño, lo convierte, elimina
su GGUF original y genera dos tokens desde el contenedor. También comprueba
argumentos, límites de materialización e integridad. No requiere descargar un
modelo ni copiar los archivos locales de la máquina de desarrollo.

## Ejecutar un archivo propio

```sh
cargo build -p bml-cli --release --locked
./target/release/bml-cli inspect-weights -m modelo.bmlgguf --verify-all
./target/release/bml-cli run -m modelo.bmlgguf -p "AB" -n 2 --temp 0 \
  --weight-evaluation checked-exact-f32 --max-evaluated-weights 10000
```

El presupuesto `10000` sirve para fixtures pequeños: limita la cantidad de
pesos materializados, no el consumo total de RAM. Debe elegirse según el modelo
y la memoria disponible. Aumentarlo no resuelve los desbordamientos del cálculo
simbólico. El archivo del modelo debe existir en la otra máquina; los modelos
descargados y los artefactos generados bajo `target/` no se distribuyen con Git.

`checked-exact-f32` verifica que cada peso se representa exactamente en F32.
Las activaciones y el KV siguen el perfil numérico documentado; no son una
inferencia completa con racionales exactos. La ejecución desde `.bmlgguf` no
necesita el GGUF fuente y no admite combinar estas opciones con `--bml` o
`--weights`.

Formato, conversión y límites: [BML 1.0](BML_1_0.md).
