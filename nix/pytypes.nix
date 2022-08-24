{ fetchPypi, buildPythonPackage, setuptools_scm }:

buildPythonPackage rec {
  pname = "pytypes";
  version = "1.0b10";

  src = fetchPypi {
    inherit pname version;
    sha256 = "sha256-+SZAQcGNi0d5YjvJwQ2Xji0qtz0ZD+Ao+Xiz38yFJWo=";
  };

  propagatedBuildInputs = [ setuptools_scm ];
}
