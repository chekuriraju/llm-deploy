#!/usr/bin/env bash
# Verify the deployed app is reachable and the model responds.
#
# Supports both direct NodePort access (SERVICE_URL=http://localhost:30080)
# and port-forwarding fallback if running inside WSL/devcontainers.

set -euo pipefail

PORT_FORWARD_PID=""
cleanup() {
  if [[ -n "${PORT_FORWARD_PID}" ]]; then
    kill "${PORT_FORWARD_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo ">>> Kubernetes resources"
kubectl get svc,endpoints,pods -l app=llm-deploy

SERVICE_URL="${SERVICE_URL:-}"

if [[ -z "${SERVICE_URL}" ]]; then
  if curl -sS --connect-timeout 2 "http://localhost:30080/health" >/dev/null 2>&1; then
    SERVICE_URL="http://localhost:30080"
  else
    echo ""
    echo ">>> NodePort 30080 not directly reachable from environment; starting port-forward to 18080..."
    kubectl port-forward svc/llm-deploy 18080:8000 >/dev/null 2>&1 &
    PORT_FORWARD_PID=$!
    sleep 2
    SERVICE_URL="http://127.0.0.1:18080"
  fi
fi

echo ""
echo ">>> ${SERVICE_URL}/health"
HEALTH=$(curl -sS "${SERVICE_URL}/health")
echo "  -> ${HEALTH}"

echo ""
echo ">>> ${SERVICE_URL}/generate  (POST)"
PROMPT="Classify the sentiment: I really loved this tutorial, it made everything so easy!"
echo "  Prompt: \"${PROMPT}\""
RESP=$(curl -sS -X POST "${SERVICE_URL}/generate" \
  -H 'Content-Type: application/json' \
  -d "{\"prompt\": \"${PROMPT}\", \"max_new_tokens\": 16}")
echo "  -> ${RESP}"

echo ""
echo ">>> Last 15 lines of pod logs"
POD=$(kubectl get pod -l app=llm-deploy -o jsonpath='{.items[0].metadata.name}')
kubectl logs "${POD}" --tail=15
