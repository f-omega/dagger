{
  description = "Python workflow library";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-22.05";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let pkgs = import nixpkgs { inherit system;
                                  config.allowUnfree = true;
                                  overlays = [ (import ./nix/overlay.nix) ]; };
      in rec {
        packages = rec {
          inherit (pkgs) pytypes;
        };

        devShell = pkgs.stdenv.mkDerivation {
          name = "dagger-shell";

          buildInputs = with pkgs; [
            (python39.withPackages (p: with p; [
              python-lsp-server pylsp-mypy mypy pylsp-rope python-lsp-black
              pytypes typeguard pytest boto3 redis
            ]))

            graphviz sqlite-interactive redis
          ];
        };
      }
    );
}
