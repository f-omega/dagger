{ buildPythonPackage, fetchPypi, python-lsp-server, rope, pytest }:

buildPythonPackage rec {
  pname = "pylsp-rope";
  version = "0.1.6";

  src = fetchPypi {
    inherit pname version;
    sha256 = "sha256-JypYuxgShiTIobE6em2V3TjA9FLes3wEVOGIem69qL4=";
  };

  propagatedBuildInputs = [ python-lsp-server rope pytest ];
}
