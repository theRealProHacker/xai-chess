#!/usr/bin/env bash
# Bundles CodeMirror 6 into codemirror.js (one ES module, so the page loads one copy of @codemirror/state).
set -euo pipefail
cd "$(dirname "$0")"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
cat > "$tmp/entry.js" <<'JS'
export { EditorView, keymap, hoverTooltip, drawSelection, highlightActiveLine, placeholder } from '@codemirror/view';
export { EditorState, Prec } from '@codemirror/state';
export { history, defaultKeymap, historyKeymap, indentWithTab } from '@codemirror/commands';
export { autocompletion, completionKeymap, closeBrackets, closeBracketsKeymap, snippetCompletion, snippet, nextSnippetField, prevSnippetField, startCompletion, acceptCompletion, completionStatus } from '@codemirror/autocomplete';
export { bracketMatching } from '@codemirror/language';
JS
cd "$tmp"
npm init -y >/dev/null
npm i --silent @codemirror/view@6.38.6 @codemirror/state@6.5.2 @codemirror/commands@6.9.0 \
  @codemirror/autocomplete@6.19.0 @codemirror/language@6.11.3 esbuild@0.25.10
npx esbuild entry.js --bundle --format=esm --minify --outfile="$OLDPWD/codemirror.js"
