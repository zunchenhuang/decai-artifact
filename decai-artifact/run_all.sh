#!/usr/bin/env bash
# Full reproduction, stage by stage, with one log per stage in logs/.
#   THREADS=8 ./run_all.sh                    training + evaluation + report (about 10 h on one core)
#   THREADS=8 ./run_all.sh --skip-training    evaluation of the bundled models + report
# Set PYLIRPA to a Python with auto_LiRPA (e.g. PYLIRPA=~/venv311/bin/python) to rerun the auto_LiRPA stages;
# otherwise their bundled results are kept.  A stage that fails is reported and the run continues.
set -uo pipefail
cd "$(dirname "$0")"; mkdir -p logs
train=(train-std train-cert train-conf train-big train-k10 train-big-conf train-ibp train-std-arch train-casestudy)
evals=(eval-std eval-cert eval-conf eval-refine eval-big eval-big-conf eval-k10 beyond soundness correctness
       example ablation-loss eval-std-arch casestudy)
stages=("${train[@]}" "${evals[@]}")
[[ "${1:-}" == "--skip-training" ]] && stages=("${evals[@]}")
if [[ -n "${PYLIRPA:-}" ]] && ! command -v "$PYLIRPA" >/dev/null; then
  echo "PYLIRPA=$PYLIRPA is not an executable Python; set it to the real path (e.g. \$(which python) in the 3.11 env)"; exit 2
fi
if ${PYLIRPA:-python3} -c "import auto_LiRPA" 2>/dev/null; then
  echo "auto_LiRPA found: $(${PYLIRPA:-python3} -c 'import auto_LiRPA, sys; print(auto_LiRPA.__version__, "on Python", sys.version.split()[0])')"
  stages+=(ablation-lirpa precision precision-neutral)
else
  echo "auto_LiRPA not found (set PYLIRPA): keeping the bundled results of ablation-lirpa, precision, precision-neutral"
fi
failed=()
for s in "${stages[@]}"; do
  t0=$(date +%s); echo -n "[$(date +%H:%M:%S)] $s ... "
  if make "$s" > "logs/$s.log" 2>&1; then echo "ok ($(( $(date +%s) - t0 )) s)"; else echo "FAILED (see logs/$s.log)"; failed+=("$s"); fi
done
make check > logs/check.log 2>&1 || failed+=(check)
make report > logs/report.log 2>&1 || failed+=(report)
make paper > logs/paper.log 2>&1 || failed+=(paper)
echo "failed stages: ${failed[*]:-none}"
echo "send back: rerun_report.zip (results, logs, regenerated tables; logs/compare_report.md summarises the changes)"
