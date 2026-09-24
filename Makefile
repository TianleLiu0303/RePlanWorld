PYTHON ?= python3
WORKSPACE_ROOT ?= $(abspath ..)
AUDIT_PYTHON ?= $(WORKSPACE_ROOT)/miniconda3/envs/lingbot-map/bin/python

.PHONY: test audit-smoke audit-mini-full audit-trainval-full audit-full

test:
	$(PYTHON) -m unittest discover -s tests -v

audit-smoke:
	$(AUDIT_PYTHON) -m replan_world.data.audit --config configs/audit.json --workspace-root $(WORKSPACE_ROOT) --max-logs-per-split 8 --output reports/phase0/data_audit.json

audit-mini-full:
	$(AUDIT_PYTHON) -m replan_world.data.audit --config configs/audit.json --workspace-root $(WORKSPACE_ROOT) --dataset mini --max-logs-per-split 0 --output reports/phase0/data_audit_mini_full.json

audit-trainval-full:
	$(AUDIT_PYTHON) -m replan_world.data.audit --config configs/audit.json --workspace-root $(WORKSPACE_ROOT) --dataset trainval --max-logs-per-split 0 --checkpoint-dir /tmp/replanworld_audit_progress --resume --output reports/phase0/data_audit_trainval_full.json

audit-full:
	$(AUDIT_PYTHON) -m replan_world.data.audit --config configs/audit.json --workspace-root $(WORKSPACE_ROOT) --max-logs-per-split 0 --checkpoint-dir /tmp/replanworld_audit_progress --resume --output reports/phase0/data_audit_full.json
