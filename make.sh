all: build install

build:
       python3 setup.py build

install:
       python3 setup.py install --user
       cp page_extractor/page_extractor.py ~/bin/page_extractor
       chmod +x ~/bin/page_extractor
