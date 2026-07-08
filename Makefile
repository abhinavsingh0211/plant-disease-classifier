.PHONY: install train tune evaluate app docker clean help

help:
	@echo "make install   - install dependencies"
	@echo "make train      - train the model (MobileNetV2, two-phase)"
	@echo "make tune        - run KerasTuner hyperparameter search"
	@echo "make evaluate    - evaluate the trained model on the held-out test set"
	@echo "make app         - launch the Streamlit demo"
	@echo "make docker      - build the Streamlit app container"
	@echo "make clean       - remove caches and generated artefacts"

install:
	pip install -r requirements.txt

train:
	python -m src.train

tune:
	python -m src.tune --max-epochs 10

evaluate:
	python -m src.evaluate

app:
	streamlit run app/app.py

docker:
	docker build -t plant-disease-app .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf models/kt .pytest_cache
