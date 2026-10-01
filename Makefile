.PHONY: install generate-data test train optimize dashboard clean

install:
	pip install -e ".[dev]"

generate-data:
	python -m src.data.generate_data --n-processes 50000 --seed 42

test:
	pytest

train:
	python -m src.models.train

optimize:
	python -m src.optimization.resource_allocation

dashboard:
	streamlit run app/app.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -f data/raw/*.csv data/processed/*.csv
	rm -f models/*.joblib
	rm -f reports/figures/*.png
	rm -f reports/scenario_comparison.csv reports/optimized_staffing.csv
