.PHONY: tag
.PHONY: tests
.PHONY: lints
.PHONY: docs

PKG_VERSION=`hatch version`

tag:
	git tag -a v${PKG_VERSION} -m v${PKG_VERSION}
	git push --tag

tests:
	hatch run test:test

lints:
	hatch run test:lint

docs:
    cd docs && make html
