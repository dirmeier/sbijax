PKG_VERSION=`hatch version`

tag:
	git tag -a v${PKG_VERSION} -m ${PKG_VERSION}
	git push --tag
