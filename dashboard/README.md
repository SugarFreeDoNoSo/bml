# Tablero del proyecto

Sitio estático generado desde `openspec/changes/*/tasks.md`. Las casillas
completadas, pendientes y diferidas se cuentan por separado. Los porcentajes
describen tareas documentadas; no certifican corrección ni rendimiento.
La prioridad editorial de `index.html` debe revisarse cuando cambie el roadmap.

Generar y previsualizar, sin dependencias adicionales:

```sh
python3 tools/build_dashboard.py
python3 -m http.server 8080 --directory target/dashboard
```

Abrir http://localhost:8080. La búsqueda y el filtro de estado funcionan en el
navegador; la lectura de los planes también funciona sin JavaScript.

El workflow `pages.yml` reconstruye y publica al actualizar los planes, el
roadmap, la evidencia o los archivos del tablero en `master`. También permite
ejecución manual. GitHub Pages debe usar **GitHub Actions** como origen.
Los cambios locales sin commit no aparecen en el sitio publicado.
