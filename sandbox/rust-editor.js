"use strict";

// Render text only through escaped HTML; source is never interpreted as markup.
window.RustEditor = (() => {
  const escape = text => text.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const keywords = new Set('as async await break const continue crate dyn else enum extern false fn for if impl in let loop match mod move mut pub ref return self Self static struct super trait true type unsafe use where while abstract become box do final macro override priv typeof unsized virtual yield try'.split(' '));
  const primitive = /^(?:[ui](?:8|16|32|64|128|size)|f(?:32|64)|bool|char|str)$/;
  function highlight(source) {
    let result = '', i = 0;
    while (i < source.length) {
      const rest = source.slice(i);
      let token, kind;
      if (rest.startsWith('//')) {
        token = rest.split('\n', 1)[0]; kind = 'comment';
      } else if (rest.startsWith('/*')) {
        let end = 2, depth = 1;
        while (end < rest.length && depth) {
          if (rest.startsWith('/*', end)) { depth++; end += 2; }
          else if (rest.startsWith('*/', end)) { depth--; end += 2; }
          else end++;
        }
        token = rest.slice(0, end); kind = 'comment';
      } else {
        const raw = /^(?:br|cr|r)(#*)"/.exec(rest);
        if (raw) {
          const end = rest.indexOf('"' + raw[1], raw[0].length);
          token = rest.slice(0, end < 0 ? rest.length : end + 1 + raw[1].length); kind = 'string';
        } else if ((token = /^(?:b|c)?"(?:\\[\s\S]|[^"\\])*(?:"|$)/.exec(rest)?.[0])) kind = 'string';
        else if ((token = /^(?:b)?'(?:\\(?:u\{[0-9a-fA-F_]+\}|x[0-9a-fA-F]{2}|.)|[^'\\\n])'/u.exec(rest)?.[0])) kind = 'string';
        else if ((token = /^'[a-zA-Z_][\w]*/.exec(rest)?.[0])) kind = 'lifetime';
        else if ((token = /^\b(?:0x[\da-fA-F_]+|0b[01_]+|0o[0-7_]+|\d[\d_]*(?:\.\d[\d_]*)?(?:[eE][+-]?[\d_]+)?)(?:[ui](?:8|16|32|64|128|size)|f(?:32|64))?/.exec(rest)?.[0])) kind = 'number';
        else if ((token = /^(?:r#)?[a-zA-Z_][a-zA-Z_0-9]*!?/.exec(rest)?.[0])) {
          kind = token.endsWith('!') ? 'macro' : keywords.has(token) ? 'keyword'
            : primitive.test(token) || /^[A-Z]/.test(token) ? 'type'
            : /^\s*\(/.test(rest.slice(token.length)) ? 'function' : '';
        }
      }
      if (!token) token = source[i];
      result += kind ? `<span class="rust-${kind}">${escape(token)}</span>` : escape(token);
      i += token.length;
    }
    return result + '\n';
  }
  function position(editor) {
    const before = editor.value.slice(0, editor.selectionStart);
    const lines = before.split('\n');
    return {line: lines.length - 1, character: lines.at(-1).length};
  }
  function offset(source, pos) {
    const lines = source.split('\n');
    return lines.slice(0, pos.line).reduce((n, line) => n + line.length + 1, 0) + pos.character;
  }
  function attach(cell, request) {
    const editor = cell.editor;
    const panel = document.createElement('div');
    panel.className = 'rust-tools';
    panel.innerHTML = '<div class="rust-toolbar"><button type="button" class="secondary analyze">Analizar Rust</button><button type="button" class="secondary complete">Completar</button><button type="button" class="secondary hover">Información del símbolo</button><span class="rust-status" role="status">Rust · resaltado activo</span></div><div class="rust-completions" hidden></div><pre class="rust-info" hidden></pre><ul class="rust-diagnostics" hidden></ul>';
    cell.node.querySelector('.editor').after(panel);
    const paint = document.createElement('pre');
    paint.className = 'syntax-highlight';
    paint.setAttribute('aria-hidden', 'true');
    editor.before(paint);
    editor.classList.add('highlighted');
    const status = panel.querySelector('.rust-status');
    const diagnostics = panel.querySelector('.rust-diagnostics');
    const info = panel.querySelector('.rust-info');
    const suggestions = panel.querySelector('.rust-completions');
    const sync = () => {
      paint.innerHTML = highlight(editor.value);
      paint.scrollTop = editor.scrollTop;
      paint.scrollLeft = editor.scrollLeft;
      paint.style.height = editor.offsetHeight + 'px';
    };
    editor.addEventListener('input', () => {
      sync(); diagnostics.hidden = true; info.hidden = true; suggestions.hidden = true;
      status.textContent = 'Rust · código modificado';
    });
    editor.addEventListener('scroll', sync);
    new ResizeObserver(sync).observe(editor);
    async function analyze(operation) {
      if (editor.disabled) return;
      const source = editor.value;
      const cursor = editor.selectionStart;
      const revision = ++cell.ideRevision;
      const pos = position(editor);
      status.textContent = 'Consultando rust-analyzer…';
      try {
        const result = await request(`/api/cells/${cell.id}/ide`, {source, operation, position:pos});
        if (source !== editor.value || revision !== cell.ideRevision) return;
        status.textContent = result.ready ? 'rust-analyzer · listo' : 'rust-analyzer · indexando; vuelve a analizar';
        diagnostics.replaceChildren();
        for (const item of result.diagnostics || []) {
          const li = document.createElement('li');
          li.className = item.severity === 1 ? 'diagnostic-error' : 'diagnostic-warning';
          li.textContent = `L${item.range.start.line + 1}:${item.range.start.character + 1} · ${item.message}`;
          diagnostics.append(li);
        }
        diagnostics.hidden = diagnostics.children.length === 0;
        info.hidden = true;
        suggestions.hidden = true;
        if (operation === 'hover') {
          const contents = result.result?.contents;
          const values = Array.isArray(contents) ? contents : [contents];
          info.textContent = values.filter(Boolean).map(value => typeof value === 'string' ? value : value.value).join('\n\n') || 'Sin información para esta posición.';
          info.hidden = false;
        }
        if (operation === 'completion') {
          suggestions.replaceChildren();
          const items = Array.isArray(result.result) ? result.result : result.result?.items || [];
          for (const item of items.slice(0, 30)) {
            const button = document.createElement('button');
            button.type = 'button'; button.textContent = item.label;
            button.title = item.detail || item.label;
            button.addEventListener('click', () => {
              if (editor.value !== source || editor.disabled) return;
              const edit = item.textEdit;
              const range = edit?.range || edit?.replace;
              const word = source.slice(0, cursor).match(/[\w]*$/)[0];
              const primary = range ? {start:offset(source, range.start), end:offset(source, range.end), text:edit.newText}
                : {start:cursor - word.length, end:cursor, text:item.insertText || item.label};
              const edits = [primary, ...(item.additionalTextEdits || []).map(e => ({start:offset(source,e.range.start),end:offset(source,e.range.end),text:e.newText}))];
              let updated = source;
              for (const e of edits.sort((a,b) => b.start - a.start)) updated = updated.slice(0,e.start) + e.text + updated.slice(e.end);
              editor.value = updated;
              editor.dispatchEvent(new Event('input'));
              editor.focus();
              const shift = edits.filter(e => e !== primary && e.end <= primary.start).reduce((n,e) => n + e.text.length - (e.end-e.start),0);
              editor.setSelectionRange(primary.start + primary.text.length + shift, primary.start + primary.text.length + shift);
              suggestions.hidden = true;
            });
            suggestions.append(button);
          }
          suggestions.hidden = false;
          if (!items.length) suggestions.textContent = 'Sin sugerencias en esta posición.';
        }
      } catch (error) {
        if (revision === cell.ideRevision) status.textContent = error.message;
      }
    }
    cell.ideRevision = 0;
    panel.querySelector('.analyze').addEventListener('click', () => analyze('diagnostics'));
    panel.querySelector('.complete').addEventListener('click', () => analyze('completion'));
    panel.querySelector('.hover').addEventListener('click', () => analyze('hover'));
    editor.addEventListener('keydown', event => {
      if (event.ctrlKey && event.code === 'Space') { event.preventDefault(); analyze('completion'); }
      if (event.key === 'Escape') { suggestions.hidden = true; info.hidden = true; }
    });
    sync();
    return sync;
  }
  return {attach, highlight};
})();
