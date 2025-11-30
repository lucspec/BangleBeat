{
  description = "flake for TensorFlow magic-wand notebook";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = import nixpkgs { inherit system; };
      py = pkgs.python313;
      pypkgs = pkgs.python313Packages;
    in {
      devShell = pkgs.mkShell {
        name = "BangleBeat";
        buildInputs = with pkgs; [
          git
          ripgrep
          py
          pypkgs.ipykernel
          pypkgs.jupyter
          pypkgs.notebook
          pypkgs.pip
          pypkgs.virtualenv

          # ipynb PDF export in codium
          dblatex
          texliveFull
        ];
        
        shellHook = ''
          export TF_ENABLE_ONEDNN_OPTS=0
          
          if [ ! -d ".venv" ]; then
            python3 -m venv .venv
          fi

          source .venv/bin/activate
          pip install poetry
          rm poetry.lock && poetry lock
          poetry install
          
          echo ""
          echo "Virtual environment ready at .venv/bin/python"
          echo "VSCodium: Select .venv/bin/python as interpreter"
          echo ""
          echo "Run: jupyter lab train.ipynb"
        '';
      };
    }
  );
}
