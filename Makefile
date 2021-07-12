tests:
	python -m unittest discover test

install: 
	python3 -m pip install -r requirements.txt --extra-index-url=https://${NETSQUIDPYPI_USER}:${NETSQUIDPYPI_PWD}@pypi.netsquid.org
