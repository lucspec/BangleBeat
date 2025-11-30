{
  description = "Python development environment with pandas and matplotlib";

  inputs = {
    nixpkgs.url = "https://github.com/NixOS/nixpkgs/archive/117cc7f94e8072499b0a7aa4c52084fa4e11cc9b.tar.gz";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python312;
        pythonEnv = python.withPackages (ps: with ps; [
          ipykernel
          jupyter
          keras
          matplotlib
          notebook
          numpy
          onnx
          pandas
          pip
          scipy
          seaborn
          scikit-learn
          skl2onnx
          tensorflow
        ]);

        baseBuildInputs = [ 
          pythonEnv
        ];

        commonShellHook =  ''
            echo "Python environment with pandas and matplotlib loaded"
            echo "Python version: $(python --version)"
            echo "Available packages:"
            pip list
          '';
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = baseBuildInputs ++ [ pkgs.sqlitebrowser ];
          shellHook = commonShellHook;
        };
        
        devShells.user = pkgs.mkShell {
          buildInputs = baseBuildInputs;
          shellHook = commonShellHook;
        };
      }
    );
}
