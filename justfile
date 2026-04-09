default:
    @just --list

test:
    python handler.py --test_input '{"input":{"messages":[{"role":"user","content":"Hi"}],"max_tokens":64}}'

client-sync prompt:
    python client.py "{{prompt}}"

client-stream prompt:
    python client.py --stream "{{prompt}}"

clean:
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -name '*.pyc' -delete 2>/dev/null || true
