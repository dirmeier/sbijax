.PHONY: tag
.PHONY: tests
.PHONY: lints
.PHONY: docs
.PHONY: format

PKG_VERSION=`hatch version`

tag:
	 git tag -a v${PKG_VERSION} -m v${PKG_VERSION}
	 git push --tag

tests:
	hatch run test:test

lints:
	hatch run test:lint

format:
	hatch run test:format

docs:
	cd docs && make html
