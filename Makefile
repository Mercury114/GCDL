PYTHON ?= python
PYTHONPATH := src

.PHONY: test audit compile check demo

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests -v

compile:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m py_compile scripts/*.py demo/*.py src/emotion_conflict/*.py

audit:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/audit_results.py \
		--correction /path/to/correction_results.json \
		--detector /path/to/detector_results_v2.json \
		--output outputs/audit_report.json

check:
	$(PYTHON) scripts/check_submission.py

demo:
	PYTHONPATH=$(PYTHONPATH) streamlit run demo/streamlit_app.py
