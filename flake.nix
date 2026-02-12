{
  description = "Flake for Arcgentica";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
        };
      in {
        # Enter reproducible development shell with `nix develop`
        devShells = {
          # The full shell with all packages can be quite heavy, e.g certain CI tests benefit from having less packages.
          default = pkgs.mkShell {
            buildInputs = with pkgs; [
              ruff # Python formatter and linter.
            ];
          };
        };
      }
    );
}
