# Quark Directory Cleanup Plan

## Files to Remove (Redundant/Outdated)

### Root Directory Status Files
- `PHASE_4_COMPLETION_SUMMARY.md` → Move to docs/history/
- `PHASE_4_FINAL_STATUS.md` → Remove (redundant)
- `PHASE_5_PROGRESS_SUMMARY.md` → Move to docs/history/
- `PHASE_5_PLAN.md` → Move to docs/planning/
- `CURRENT_STATUS_SUMMARY.md` → Remove (redundant)
- `PILLAR_9_AND_PHASES_COMPLETION_SUMMARY.md` → Move to docs/history/
- `PILLAR_9_COMPLETION_SUMMARY.md` → Move to docs/history/
- `PILLARS_5_6_7_8_COMPLETION_SUMMARY.md` → Move to docs/history/
- `PHASE_3_COMPLETION_SUMMARY.md` → Move to docs/history/

### Test Files in Root Directory
- `test_pillars_simple.py` → Move to tests/
- `test_pillar_9.py` → Move to tests/
- `test_pillars_final.py` → Move to tests/

### System Files
- `.DS_Store` → Remove (macOS system file)

## Directories to Organize

### Create New Directories
- `docs/history/` → For completion summaries
- `docs/planning/` → For planning documents
- `docs/status/` → For current status documents

### Data Directories to Clean
- `reasoning_data/` → Check if needed
- `memory_db/` → Check if needed
- `safety/` → Check if needed
- `evaluation/` → Check if needed
- `metrics/` → Check if needed
- `logs/` → Check if needed

## Files to Keep in Root
- `README.md`
- `LICENSE`
- `CODE_OF_CONDUCT.md`
- `CONTRIBUTING.md`
- `CHANGELOG.md`
- `PROJECT_STRUCTURE.md`
- `META_MODEL_STATUS.md` (current status)
- `pyproject.toml`
- `setup.py`
- `MANIFEST.in`
- `main.py`
- `__init__.py`
- `docker-compose.yml`
- `Dockerfile`
- `.gitignore`

## Core Implementation Directories to Keep
- `agents/`
- `core/`
- `alignment/`
- `autonomy/`
- `governance/`
- `knowledge_graphs/`
- `memory/`
- `meta_learning/`
- `reasoning/`
- `social/`
- `web/`
- `cli/`
- `config/`
- `deployment/`
- `docs/`
- `tests/`
- `training/`
- `scripts/`

## Optional Directories (Check Content)
- `capabilities/`
- `learning/`
- `models/`
- `venv/`

## Cleanup Steps
1. Create new directory structure
2. Move files to appropriate locations
3. Remove redundant files
4. Clean up empty directories
5. Update any references to moved files
6. Create a final project status document 