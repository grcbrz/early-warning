PY ?= python

.PHONY: data interim features train dashboard all test test-network test-acceptance serve docker

data:            ## download raw inputs (~4 GB Hansen tiles; NDVI/rain via range reads)
	$(PY) scripts/fetch.py hansen
	$(PY) scripts/fetch.py ndvi
	$(PY) scripts/fetch.py precip
interim:         ## validate + aggregate to the 0.25 deg grid
	$(PY) -m congo_ews.pipeline interim
features:
	$(PY) -m congo_ews.pipeline features
train:
	$(PY) -m congo_ews.pipeline train
dashboard:
	$(PY) dashboard/build_dashboard.py
all: interim features train dashboard

test:            ## unit + integration (offline)
	$(PY) -m pytest -m "not network and not acceptance"
test-acceptance: ## spec acceptance criteria against built artifacts
	$(PY) -m pytest -m acceptance
test-network:
	$(PY) -m pytest -m network
serve:
	uvicorn congo_ews.serve.api:app --reload
docker:
	docker build -t congo-ews:0.1.0 . && docker run --rm -p 8000:8000 congo-ews:0.1.0
