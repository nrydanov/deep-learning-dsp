.PHONY: clean clear

clean:
	rm -rf results

clear:
	jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace *.ipynb

build:
	docker-compose up